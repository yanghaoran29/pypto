/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file expand_mx_packed_quant_pass.cpp
 * @brief Expand ``tile.tquant_mx(..., layout=MX_A_ZZ|MX_B_NN)`` into per-box flat
 *        quant + continuous ZZ/NN scale packing (B also INT8-transposes to [K,N]).
 *
 * Preferred lowering matches the onboard-proven rearrange recipe: resolve the
 * source ``tile.load``, then per-box ``tile.load`` + flat ``tquant_mx`` +
 * ``tile.store`` into the consumer GM buffers. Vec ``slice``/``assemble`` packing
 * is only a fallback when stores are not visible and is less reliable on A5.
 */

#include <any>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

constexpr int64_t kMxPackTileM = 16;
constexpr int64_t kMxPackTileK = 64;
constexpr int64_t kMxPackGroup = 32;
constexpr int64_t kMxPackBoxRows = kMxPackTileM * (kMxPackTileK / kMxPackGroup);  // 32

std::optional<TensorLayout> GetMxPackLayout(const CallPtr& call) {
  if (!call || !IsOp(call, "tile.tquant_mx")) {
    return std::nullopt;
  }
  for (const auto& [key, value] : call->kwargs_) {
    if (key != "layout") {
      continue;
    }
    auto layout = AnyCast<TensorLayout>(value, "kwarg key: layout");
    CHECK(layout == TensorLayout::MX_A_ZZ || layout == TensorLayout::MX_B_NN)
        << "tile.tquant_mx layout must be MX_A_ZZ or MX_B_NN (ND/None are not allowed), got "
        << TensorLayoutToString(layout);
    return layout;
  }
  return std::nullopt;
}

ExprPtr MakeIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MakeShape2(int64_t d0, int64_t d1, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(d0, span), MakeIndex(d1, span)}, span);
}

const Var* GetVarIdentity(const ExprPtr& expr) {
  if (auto var = As<Var>(expr)) return var.get();
  if (auto iter_arg = As<IterArg>(expr)) return iter_arg.get();
  return nullptr;
}

bool ConstOffset2(const ExprPtr& offsets, int64_t* r, int64_t* c) {
  auto tup = As<MakeTuple>(offsets);
  if (!tup || tup->elements_.size() != 2) return false;
  auto a = As<ConstInt>(tup->elements_[0]);
  auto b = As<ConstInt>(tup->elements_[1]);
  if (!a || !b) return false;
  *r = a->value_;
  *c = b->value_;
  return true;
}

struct ResolvedTileLoad {
  ExprPtr tensor;
  int64_t row0 = 0;
  int64_t col0 = 0;
  std::vector<std::pair<std::string, std::any>> kwargs;
};

std::optional<ResolvedTileLoad> ResolveTileLoad(ExprPtr expr,
                                                const std::unordered_map<const Var*, ExprPtr>& defs) {
  for (int depth = 0; depth < 8 && expr; ++depth) {
    if (auto call = As<Call>(expr); call && IsOp(call, "tile.load")) {
      CHECK(call->args_.size() >= 4) << "tile.load expects tensor, offsets, shapes, valid_shape";
      int64_t r = 0;
      int64_t c = 0;
      CHECK(ConstOffset2(call->args_[1], &r, &c))
          << "ExpandMxPackedQuant requires tile.load offsets to be constant 2-D tuples";
      ResolvedTileLoad out;
      out.tensor = call->args_[0];
      out.row0 = r;
      out.col0 = c;
      out.kwargs = call->kwargs_;
      return out;
    }
    if (const Var* v = GetVarIdentity(expr)) {
      auto it = defs.find(v);
      if (it == defs.end()) return std::nullopt;
      expr = it->second;
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

class ExpandBuilder {
 public:
  ExpandBuilder(std::string base_name, std::size_t& temp_counter)
      : base_name_(std::move(base_name)), temp_counter_(temp_counter) {}

  ExprPtr Bind(const std::string& qualifier, const ExprPtr& expr, const Span& span) {
    auto var = std::make_shared<Var>(MakeTempName(qualifier), expr->GetType(), span);
    stmts_.push_back(std::make_shared<AssignStmt>(var, expr, span));
    return var;
  }

  std::vector<StmtPtr> TakeStmts() { return std::move(stmts_); }

 private:
  std::string MakeTempName(const std::string& qualifier) {
    return auto_name::BuildName(auto_name::GetBaseName(base_name_), qualifier, "tmp",
                                static_cast<int>(temp_counter_++));
  }

  std::string base_name_;
  std::size_t& temp_counter_;
  std::vector<StmtPtr> stmts_;
};

ExprPtr LoadBox(const ResolvedTileLoad& ld, int64_t row, int64_t col, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto shape = MakeShape2(kMxPackTileM, kMxPackTileK, span);
  return b.Bind(
      "box",
      reg.Create("tile.load", {ld.tensor, MakeShape2(ld.row0 + row, ld.col0 + col, span), shape, shape},
                 ld.kwargs, span),
      span);
}

ExprPtr SliceBox(const ExprPtr& src, int64_t row, int64_t col, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  return b.Bind(
      "box",
      reg.Create("tile.slice",
                 {src, MakeShape2(kMxPackTileM, kMxPackTileK, span), MakeShape2(row, col, span)}, {}, span),
      span);
}

struct BoxQuant {
  ExprPtr q_mk;  // [16, 64] FP8
  ExprPtr s;     // [1, 32] E8M0
};

BoxQuant QuantizeBox(const ExprPtr& box, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto box32 = b.Bind(
      "box32", reg.Create("tile.reshape", {box, MakeShape2(kMxPackBoxRows, kMxPackGroup, span)}, {}, span),
      span);
  std::string mode = "mxfp8_e4m3";
  auto pair = b.Bind("tq", reg.Create("tile.tquant_mx", {box32}, {{"mode", mode}}, span), span);
  auto q_box = b.Bind("qb", std::make_shared<TupleGetItemExpr>(pair, 0, span), span);
  auto s_box = b.Bind("sb", std::make_shared<TupleGetItemExpr>(pair, 1, span), span);
  auto q_mk = b.Bind(
      "qmk", reg.Create("tile.reshape", {q_box, MakeShape2(kMxPackTileM, kMxPackTileK, span)}, {}, span),
      span);
  return {q_mk, s_box};
}

ExprPtr CreateMxScaleU8Buffer(ExpandBuilder& b, int64_t groups, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto groups_dim = MakeIndex(groups, span);
  auto one = MakeIndex(1, span);
  // Plain ND UINT8 (no MX fractal): assemble packs bytes like consecutive GM stores.
  TileView scale_u8_view;
  scale_u8_view.valid_shape = {one, groups_dim};
  scale_u8_view.blayout = TileLayout::row_major;
  scale_u8_view.slayout = TileLayout::none_box;
  auto scale_u8_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::UINT8,
                                                  std::nullopt, scale_u8_view, MemorySpace::Vec);
  auto scale_shape = MakeShape2(1, groups, span);
  auto raw_create = As<Call>(reg.Create(
      "tile.create", {scale_shape}, {{"dtype", DataType::UINT8}, {"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw_create, span) << "Internal error: MX scale tile.create did not produce a Call";
  auto typed = std::make_shared<Call>(raw_create->op_, raw_create->args_, raw_create->kwargs_,
                                      raw_create->attrs_, scale_u8_type, span);
  return b.Bind("s_u8", typed, span);
}

ExprPtr CreatePlainU8Buffer(ExpandBuilder& b, int64_t rows, int64_t cols, const std::string& name,
                            const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto r = MakeIndex(rows, span);
  auto c = MakeIndex(cols, span);
  TileView view;
  view.valid_shape = {r, c};
  view.blayout = TileLayout::row_major;
  view.slayout = TileLayout::none_box;
  auto type = std::make_shared<TileType>(std::vector<ExprPtr>{r, c}, DataType::UINT8, std::nullopt, view,
                                         MemorySpace::Vec);
  auto shape = MakeShape2(rows, cols, span);
  auto raw = As<Call>(reg.Create("tile.create", {shape},
                                 {{"dtype", DataType::UINT8}, {"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw, span) << "Internal error: quant tile.create did not produce a Call";
  auto typed = std::make_shared<Call>(raw->op_, raw->args_, raw->kwargs_, raw->attrs_, type, span);
  return b.Bind(name, typed, span);
}

ExprPtr StoreTile(const ExprPtr& tile, int64_t row, int64_t col, const ExprPtr& tensor, ExpandBuilder& b,
                  const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  return b.Bind("st", reg.Create("tile.store", {tile, MakeShape2(row, col, span), tensor}, {}, span), span);
}

ExprPtr LoadMxScaleTile(const ExprPtr& tensor, int64_t row, int64_t col, int64_t groups, ExpandBuilder& b,
                        const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto one = MakeIndex(1, span);
  auto groups_dim = MakeIndex(groups, span);
  auto shape = MakeShape2(1, groups, span);
  auto raw = As<Call>(reg.Create("tile.load", {tensor, MakeShape2(row, col, span), shape, shape},
                                 {{"target_memory", MemorySpace::Vec}}, span));
  INTERNAL_CHECK_SPAN(raw, span) << "Internal error: MX scale tile.load did not produce a Call";

  // The GM tensor stores ordinary bytes, so tile.load cannot infer that this is
  // an MX scale result.  Restore tquant_mx's public fractal-32 scale type on the
  // Vec reload; consumers and round-trip verification rely on that metadata.
  TileView scale_view;
  scale_view.valid_shape = {one, groups_dim};
  scale_view.blayout = TileLayout::row_major;
  scale_view.slayout = TileLayout::none_box;
  scale_view.fractal = tile_view_semantics::kMXScaleFractal;
  auto scale_type = std::make_shared<TileType>(std::vector<ExprPtr>{one, groups_dim}, DataType::FP8E8M0,
                                               std::nullopt, scale_view, MemorySpace::Vec);
  auto typed = std::make_shared<Call>(raw->op_, raw->args_, raw->kwargs_, raw->attrs_, scale_type, span);
  return b.Bind("s", typed, span);
}

struct StoreSite {
  ExprPtr tensor;
  int64_t row0 = 0;
  int64_t col0 = 0;
  const Var* result_var = nullptr;  // LHS of the store assign
};

struct MxPackSite {
  const Var* quant_var = nullptr;
  TensorLayout layout = TensorLayout::ND;
  ExprPtr src;
  std::optional<StoreSite> q_store;
  std::optional<StoreSite> s_store;
};

struct ExpansionResult {
  ExprPtr quant;  // tile result (loaded back or assembled)
  ExprPtr scale;
  ExprPtr q_store_result;  // final stored tensor (if store-fused)
  ExprPtr s_store_result;
  ExprPtr scratch_result;  // final [N,K] scratch tensor (B only)
  std::vector<ExprPtr> reuse_barriers;
};

struct FusedStoreReplacement {
  ExprPtr tensor_result;
  std::vector<ExprPtr> tile_keepalives;
};

FusedStoreReplacement MakeFusedStoreReplacement(const ExprPtr& tensor_result, const ExprPtr& stored_tile,
                                                const std::vector<ExprPtr>& reuse_barriers) {
  auto keepalives = reuse_barriers;
  keepalives.push_back(stored_tile);
  return {tensor_result, std::move(keepalives)};
}

ExpansionResult ExpandMxAZzStoreFused(const ResolvedTileLoad& ld, int64_t m, int64_t k,
                                      const StoreSite& q_store, const StoreSite& s_store, ExpandBuilder& b,
                                      const Span& span) {
  const int64_t mb = m / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;
  ExprPtr q_t = q_store.tensor;
  ExprPtr s_t = s_store.tensor;
  std::vector<ExprPtr> reuse_barriers;
  for (int64_t mi = 0; mi < mb; ++mi) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      auto box = LoadBox(ld, mi * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      reuse_barriers.insert(reuse_barriers.end(), {box, qq.q_mk, qq.s});
      q_t = StoreTile(qq.q_mk, q_store.row0 + mi * kMxPackTileM, q_store.col0 + ki * kMxPackTileK, q_t, b,
                      span);
      s_t = StoreTile(qq.s, s_store.row0, s_store.col0 + (mi * kb + ki) * kMxPackBoxRows, s_t, b, span);
    }
  }
  auto& reg = OpRegistry::GetInstance();
  auto q_shape = MakeShape2(m, k, span);
  auto q_tile =
      b.Bind("q",
             reg.Create("tile.load", {q_t, MakeShape2(q_store.row0, q_store.col0, span), q_shape, q_shape},
                        {{"target_memory", MemorySpace::Vec}}, span),
             span);
  auto s_tile = LoadMxScaleTile(s_t, s_store.row0, s_store.col0, m * k / kMxPackGroup, b, span);
  return {q_tile, s_tile, q_t, s_t, nullptr, std::move(reuse_barriers)};
}

ExpansionResult ExpandMxBNnStoreFused(const ResolvedTileLoad& ld, int64_t n, int64_t k,
                                      const StoreSite& q_store, const StoreSite& s_store,
                                      const ExprPtr& scratch_nk, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  const int64_t nb = n / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;

  // Prefer an Out [N,K] scratch tensor (same recipe as rearrange_ab). Fall back to
  // Vec FP8 assemble when the caller did not provide one (InCore has no tensor.create).
  ExprPtr nk_t = scratch_nk;
  if (!nk_t) {
    auto nk_rows = MakeIndex(n, span);
    auto nk_cols = MakeIndex(k, span);
    TileView nk_view;
    nk_view.valid_shape = {nk_rows, nk_cols};
    nk_view.blayout = TileLayout::row_major;
    nk_view.slayout = TileLayout::none_box;
    auto nk_type = std::make_shared<TileType>(std::vector<ExprPtr>{nk_rows, nk_cols}, DataType::FP8E4M3FN,
                                              std::nullopt, nk_view, MemorySpace::Vec);
    auto nk_shape = MakeShape2(n, k, span);
    auto raw_nk =
        As<Call>(reg.Create("tile.create", {nk_shape},
                            {{"dtype", DataType::FP8E4M3FN}, {"target_memory", MemorySpace::Vec}}, span));
    INTERNAL_CHECK_SPAN(raw_nk, span) << "Internal error: B quant tile.create did not produce a Call";
    auto typed_nk =
        std::make_shared<Call>(raw_nk->op_, raw_nk->args_, raw_nk->kwargs_, raw_nk->attrs_, nk_type, span);
    nk_t = b.Bind("nk_q", typed_nk, span);
  }

  ExprPtr s_t = s_store.tensor;
  const bool scratch_is_tensor = scratch_nk != nullptr;
  std::vector<ExprPtr> reuse_barriers;
  for (int64_t ni = 0; ni < nb; ++ni) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      auto box = LoadBox(ld, ni * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      reuse_barriers.insert(reuse_barriers.end(), {box, qq.q_mk, qq.s});
      if (scratch_is_tensor) {
        nk_t = StoreTile(qq.q_mk, ni * kMxPackTileM, ki * kMxPackTileK, nk_t, b, span);
      } else {
        nk_t = b.Bind(
            "nk_q",
            reg.Create("tile.assemble",
                       {nk_t, qq.q_mk, MakeShape2(ni * kMxPackTileM, ki * kMxPackTileK, span)}, {}, span),
            span);
      }
      s_t = StoreTile(qq.s, s_store.row0, s_store.col0 + (ni * kb + ki) * kMxPackBoxRows, s_t, b, span);
    }
  }

  ExprPtr nk_q = nk_t;
  if (scratch_is_tensor) {
    auto nk_shape = MakeShape2(n, k, span);
    nk_q = b.Bind("nk_q",
                  reg.Create("tile.load", {nk_t, MakeShape2(0, 0, span), nk_shape, nk_shape},
                             {{"target_memory", MemorySpace::Vec}}, span),
                  span);
  }
  auto nk_i8 =
      b.Bind("nk_i8", reg.Create("tile.reinterpret_view", {nk_q}, {{"dtype", DataType::INT8}}, span), span);
  auto kn_i8 = b.Bind(
      "kn_i8", reg.Create("tile.transpose", {nk_i8, MakeIndex(0, span), MakeIndex(1, span)}, {}, span), span);
  auto kn_q = b.Bind(
      "kn_q", reg.Create("tile.reinterpret_view", {kn_i8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  ExprPtr q_t = StoreTile(kn_q, q_store.row0, q_store.col0, q_store.tensor, b, span);

  auto kn_shape = MakeShape2(k, n, span);
  auto q_tile =
      b.Bind("q",
             reg.Create("tile.load", {q_t, MakeShape2(q_store.row0, q_store.col0, span), kn_shape, kn_shape},
                        {{"target_memory", MemorySpace::Vec}}, span),
             span);
  auto s_tile = LoadMxScaleTile(s_t, s_store.row0, s_store.col0, (k / kMxPackGroup) * n, b, span);
  return {q_tile, s_tile, q_t, s_t, scratch_is_tensor ? nk_t : ExprPtr{}, std::move(reuse_barriers)};
}

ExprPtr FindBScratchNk(const FunctionPtr& func, int64_t n, int64_t k, const ExprPtr& exclude_q_store) {
  if (!func) return nullptr;
  const Var* exclude = GetVarIdentity(exclude_q_store);
  for (size_t i = 0; i < func->params_.size(); ++i) {
    if (func->param_directions_[i] == ParamDirection::In) continue;
    const auto& param = func->params_[i];
    if (exclude && param.get() == exclude) continue;
    auto tensor_type = AsTensorTypeLike(param->GetType());
    if (!tensor_type || tensor_type->dtype_ != DataType::FP8E4M3FN) continue;
    if (tensor_type->shape_.size() != 2) continue;
    auto s0 = As<ConstInt>(tensor_type->shape_[0]);
    auto s1 = As<ConstInt>(tensor_type->shape_[1]);
    if (s0 && s1 && s0->value_ == n && s1->value_ == k) {
      return param;
    }
  }
  return nullptr;
}

std::pair<ExprPtr, ExprPtr> ExpandMxAZzAssemble(const ExprPtr& src, const std::optional<ResolvedTileLoad>& ld,
                                                int64_t m, int64_t k, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  const int64_t mb = m / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;
  const int64_t groups = m * k / kMxPackGroup;

  auto q_u8 = CreatePlainU8Buffer(b, m, k, "q_u8", span);
  auto s_u8 = CreateMxScaleU8Buffer(b, groups, span);

  for (int64_t mi = 0; mi < mb; ++mi) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      ExprPtr box = ld ? LoadBox(*ld, mi * kMxPackTileM, ki * kMxPackTileK, b, span)
                       : SliceBox(src, mi * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      auto q_mk_u8 = b.Bind(
          "qmk_u8", reg.Create("tile.reinterpret_view", {qq.q_mk}, {{"dtype", DataType::UINT8}}, span), span);
      q_u8 = b.Bind(
          "q_u8",
          reg.Create("tile.assemble", {q_u8, q_mk_u8, MakeShape2(mi * kMxPackTileM, ki * kMxPackTileK, span)},
                     {}, span),
          span);
      auto s_box_u8 = b.Bind(
          "sb_u8", reg.Create("tile.reinterpret_view", {qq.s}, {{"dtype", DataType::UINT8}}, span), span);
      s_u8 =
          b.Bind("s_u8",
                 reg.Create("tile.assemble",
                            {s_u8, s_box_u8, MakeShape2(0, (mi * kb + ki) * kMxPackBoxRows, span)}, {}, span),
                 span);
    }
  }
  auto q_acc =
      b.Bind("q", reg.Create("tile.reinterpret_view", {q_u8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  auto s_acc =
      b.Bind("s", reg.Create("tile.reinterpret_view", {s_u8}, {{"dtype", DataType::FP8E8M0}}, span), span);
  return {q_acc, s_acc};
}

std::pair<ExprPtr, ExprPtr> ExpandMxBNnAssemble(const ExprPtr& src, const std::optional<ResolvedTileLoad>& ld,
                                                int64_t n, int64_t k, ExpandBuilder& b, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  const int64_t nb = n / kMxPackTileM;
  const int64_t kb = k / kMxPackTileK;
  const int64_t groups = (k / kMxPackGroup) * n;

  auto nk_u8 = CreatePlainU8Buffer(b, n, k, "nk_u8", span);
  auto s_u8 = CreateMxScaleU8Buffer(b, groups, span);

  for (int64_t ni = 0; ni < nb; ++ni) {
    for (int64_t ki = 0; ki < kb; ++ki) {
      ExprPtr box = ld ? LoadBox(*ld, ni * kMxPackTileM, ki * kMxPackTileK, b, span)
                       : SliceBox(src, ni * kMxPackTileM, ki * kMxPackTileK, b, span);
      auto qq = QuantizeBox(box, b, span);
      auto q_nk_u8 = b.Bind(
          "qnk_u8", reg.Create("tile.reinterpret_view", {qq.q_mk}, {{"dtype", DataType::UINT8}}, span), span);
      nk_u8 = b.Bind(
          "nk_u8",
          reg.Create("tile.assemble",
                     {nk_u8, q_nk_u8, MakeShape2(ni * kMxPackTileM, ki * kMxPackTileK, span)}, {}, span),
          span);
      auto s_box_u8 = b.Bind(
          "sb_u8", reg.Create("tile.reinterpret_view", {qq.s}, {{"dtype", DataType::UINT8}}, span), span);
      s_u8 =
          b.Bind("s_u8",
                 reg.Create("tile.assemble",
                            {s_u8, s_box_u8, MakeShape2(0, (ni * kb + ki) * kMxPackBoxRows, span)}, {}, span),
                 span);
    }
  }

  auto nk_q = b.Bind(
      "nk_q", reg.Create("tile.reinterpret_view", {nk_u8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  auto nk_i8 =
      b.Bind("nk_i8", reg.Create("tile.reinterpret_view", {nk_q}, {{"dtype", DataType::INT8}}, span), span);
  auto kn_i8 = b.Bind(
      "kn_i8", reg.Create("tile.transpose", {nk_i8, MakeIndex(0, span), MakeIndex(1, span)}, {}, span), span);
  auto kn_q = b.Bind(
      "kn_q", reg.Create("tile.reinterpret_view", {kn_i8}, {{"dtype", DataType::FP8E4M3FN}}, span), span);
  auto s_acc =
      b.Bind("s", reg.Create("tile.reinterpret_view", {s_u8}, {{"dtype", DataType::FP8E8M0}}, span), span);
  return {kn_q, s_acc};
}

void CheckPackShape(TensorLayout layout, int64_t d0, int64_t d1, const Span& span) {
  const char* name = layout == TensorLayout::MX_A_ZZ ? "MX_A_ZZ" : "MX_B_NN";
  CHECK_SPAN(d0 % kMxPackTileM == 0 && d1 % kMxPackTileK == 0, span)
      << "tile.tquant_mx(layout=" << name << ") requires dim0%" << kMxPackTileM << "==0 and dim1%"
      << kMxPackTileK << "==0, got [" << d0 << ", " << d1 << "]";
}

class ExpandMxPackedQuantMutator : public IRMutator {
 public:
  explicit ExpandMxPackedQuantMutator(FunctionPtr func) : func_(std::move(func)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    if (auto it = var_remap_.find(op.get()); it != var_remap_.end()) {
      return it->second;
    }
    return IRMutator::VisitExpr_(op);
  }

  ExprPtr VisitExpr_(const TupleGetItemExprPtr& op) override {
    if (const Var* tuple_var = GetVarIdentity(op->tuple_)) {
      if (auto it = tuple_outputs_.find(tuple_var); it != tuple_outputs_.end()) {
        INTERNAL_CHECK_SPAN(op->index_ >= 0 && op->index_ < 2, op->span_)
            << "Internal error: expanded MX packed quant tuple index out of range: " << op->index_;
        return it->second[static_cast<size_t>(op->index_)];
      }
    }
    return IRMutator::VisitExpr_(op);
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    // Remap stores that were already emitted by a fused expand.  Keep a bare
    // tile alias at the original store position: packed A and B quantization can
    // share one InCore function, and A's per-box load / quant buffers otherwise
    // die before B starts.  MemoryReuse may then legally place B's FP32 load /
    // quant output on those A buffers even though PTO's asynchronous pipes still
    // carry work for the distinct SSA allocations.  The aliases have no execution
    // memory access; they only express the real lifetime boundary to MemoryReuse.
    if (auto it = store_results_.find(op->var_.get()); it != store_results_.end()) {
      std::vector<StmtPtr> stmts;
      stmts.reserve(it->second.tile_keepalives.size() + 1);
      for (const auto& tile : it->second.tile_keepalives) {
        auto keepalive_name = auto_name::BuildName(auto_name::GetBaseName(op->var_->name_hint_), "keep",
                                                   "tmp", static_cast<int>(temp_counter_++));
        auto keepalive = std::make_shared<Var>(std::move(keepalive_name), tile->GetType(), op->span_);
        stmts.push_back(std::make_shared<AssignStmt>(keepalive, tile, op->span_));
      }
      stmts.push_back(std::make_shared<AssignStmt>(op->var_, it->second.tensor_result, op->span_));
      def_map_[op->var_.get()] = it->second.tensor_result;
      return std::make_shared<SeqStmts>(std::move(stmts), op->span_);
    }

    auto call = As<Call>(op->value_);
    auto layout = GetMxPackLayout(call);
    if (!layout.has_value()) {
      if (!call) {
        if (const Var* source = GetVarIdentity(op->value_)) {
          if (auto it = tuple_outputs_.find(source); it != tuple_outputs_.end()) {
            tuple_outputs_[op->var_.get()] = it->second;
            auto tuple_value =
                std::make_shared<MakeTuple>(std::vector<ExprPtr>{it->second[0], it->second[1]}, op->span_);
            auto stmt = std::make_shared<AssignStmt>(op->var_, tuple_value, op->span_);
            def_map_[op->var_.get()] = tuple_value;
            return stmt;
          }
        }
      }
      auto new_stmt = IRMutator::VisitStmt_(op);
      if (auto as = As<AssignStmt>(new_stmt)) {
        def_map_[as->var_.get()] = as->value_;
      }
      return new_stmt;
    }

    const Span& span = op->span_;
    auto src = call->args_[0];
    auto src_type = As<TileType>(src->GetType());
    INTERNAL_CHECK_SPAN(src_type && src_type->shape_.size() == 2, span)
        << "Internal error: MX packed quant expand requires 2D src";
    auto d0 = As<ConstInt>(src_type->shape_[0]);
    auto d1 = As<ConstInt>(src_type->shape_[1]);
    INTERNAL_CHECK_SPAN(d0 && d1, span) << "Internal error: MX packed quant expand requires static shape";
    CheckPackShape(*layout, d0->value_, d1->value_, span);

    auto resolved = ResolveTileLoad(src, def_map_);
    ExpandBuilder builder(op->var_->name_hint_, temp_counter_);

    // Prefer store-fused expand when both consumer stores were recorded.
    auto site_it = pack_sites_.find(op->var_.get());
    const bool can_fuse = site_it != pack_sites_.end() && site_it->second.q_store &&
                          site_it->second.s_store && resolved.has_value();

    ExpansionResult outs;
    if (can_fuse) {
      const auto& site = site_it->second;
      if (*layout == TensorLayout::MX_A_ZZ) {
        outs = ExpandMxAZzStoreFused(*resolved, d0->value_, d1->value_, *site.q_store, *site.s_store, builder,
                                     span);
      } else {
        auto scratch = FindBScratchNk(func_, d0->value_, d1->value_, site.q_store->tensor);
        outs = ExpandMxBNnStoreFused(*resolved, d0->value_, d1->value_, *site.q_store, *site.s_store, scratch,
                                     builder, span);
      }
      if (site.q_store->result_var) {
        store_results_[site.q_store->result_var] =
            MakeFusedStoreReplacement(outs.q_store_result, outs.quant, outs.reuse_barriers);
      }
      if (site.s_store->result_var) {
        store_results_[site.s_store->result_var] =
            MakeFusedStoreReplacement(outs.s_store_result, outs.scale, outs.reuse_barriers);
      }
      if (outs.scratch_result) {
        if (auto scratch = FindBScratchNk(func_, d0->value_, d1->value_, site.q_store->tensor)) {
          if (const Var* sv = GetVarIdentity(scratch)) {
            var_remap_[sv] = outs.scratch_result;
          }
        }
      }
    } else {
      auto pair = (*layout == TensorLayout::MX_A_ZZ)
                      ? ExpandMxAZzAssemble(src, resolved, d0->value_, d1->value_, builder, span)
                      : ExpandMxBNnAssemble(src, resolved, d0->value_, d1->value_, builder, span);
      outs.quant = pair.first;
      outs.scale = pair.second;
    }

    tuple_outputs_[op->var_.get()] = {outs.quant, outs.scale};
    auto stmts = builder.TakeStmts();
    auto tuple_val = std::make_shared<MakeTuple>(std::vector<ExprPtr>{outs.quant, outs.scale}, span);
    stmts.push_back(std::make_shared<AssignStmt>(op->var_, tuple_val, span));
    def_map_[op->var_.get()] = tuple_val;
    return std::make_shared<SeqStmts>(std::move(stmts), span);
  }

  void SetPackSites(std::unordered_map<const Var*, MxPackSite> sites) { pack_sites_ = std::move(sites); }

 private:
  FunctionPtr func_;
  std::size_t temp_counter_ = 0;
  std::unordered_map<const Var*, std::array<ExprPtr, 2>> tuple_outputs_;
  std::unordered_map<const Var*, ExprPtr> def_map_;
  std::unordered_map<const Var*, MxPackSite> pack_sites_;
  std::unordered_map<const Var*, FusedStoreReplacement> store_results_;
  std::unordered_map<const Var*, ExprPtr> var_remap_;
};

class CollectMxPackSites : public IRVisitor {
 public:
  void VisitStmt_(const AssignStmtPtr& op) override {
    def_map_[op->var_.get()] = op->value_;

    if (auto layout = GetMxPackLayout(As<Call>(op->value_))) {
      MxPackSite site;
      site.quant_var = op->var_.get();
      site.layout = *layout;
      site.src = As<Call>(op->value_)->args_[0];
      sites_.emplace(op->var_.get(), std::move(site));
      IRVisitor::VisitStmt_(op);
      return;
    }

    auto call = As<Call>(op->value_);
    if (call && IsOp(call, "tile.store") && call->args_.size() >= 3) {
      TryAttachStore(op, call);
    }
    IRVisitor::VisitStmt_(op);
  }

  std::unordered_map<const Var*, MxPackSite> TakeSites() { return std::move(sites_); }

 private:
  void TryAttachStore(const AssignStmtPtr& op, const CallPtr& store_call) {
    ExprPtr tile = store_call->args_[0];
    int index = -1;
    const Var* quant_var = nullptr;

    if (auto get = As<TupleGetItemExpr>(tile)) {
      quant_var = GetVarIdentity(get->tuple_);
      index = get->index_;
    } else if (const Var* tv = GetVarIdentity(tile)) {
      // Follow aliases: q = tuple[0]
      auto it = def_map_.find(tv);
      if (it != def_map_.end()) {
        if (auto get = As<TupleGetItemExpr>(it->second)) {
          quant_var = GetVarIdentity(get->tuple_);
          index = get->index_;
        }
      }
    }
    if (!quant_var || (index != 0 && index != 1)) return;
    auto sit = sites_.find(quant_var);
    if (sit == sites_.end()) return;

    int64_t r = 0;
    int64_t c = 0;
    if (!ConstOffset2(store_call->args_[1], &r, &c)) return;
    StoreSite ss;
    ss.tensor = store_call->args_[2];
    ss.row0 = r;
    ss.col0 = c;
    ss.result_var = op->var_.get();
    if (index == 0) {
      sit->second.q_store = ss;
    } else {
      sit->second.s_store = ss;
    }
  }

  std::unordered_map<const Var*, ExprPtr> def_map_;
  std::unordered_map<const Var*, MxPackSite> sites_;
};

}  // namespace

Pass ExpandMxPackedQuant() {
  auto pass_func = [](const FunctionPtr& func) -> FunctionPtr {
    if (!func || !func->body_) {
      return func;
    }
    if (!IsInCoreType(func->func_type_)) {
      return func;
    }
    CollectMxPackSites collector;
    collector.VisitStmt(func->body_);
    auto sites = collector.TakeSites();
    if (sites.empty()) {
      return func;
    }
    ExpandMxPackedQuantMutator mutator(func);
    mutator.SetPackSites(std::move(sites));
    auto new_body = mutator.VisitStmt(func->body_);
    if (new_body.get() == func->body_.get()) {
      return func;
    }
    return std::make_shared<Function>(func->name_, func->params_, func->param_directions_,
                                      func->return_types_, new_body, func->span_, func->func_type_,
                                      func->level_, func->role_, func->attrs_);
  };
  return CreateFunctionPass(pass_func, "ExpandMxPackedQuant", kExpandMxPackedQuantProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
