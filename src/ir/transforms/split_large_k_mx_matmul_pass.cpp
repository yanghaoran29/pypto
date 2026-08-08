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
 * @file split_large_k_mx_matmul_pass.cpp
 * @brief Split large-K MX matmul into K=64 chunks with matmul_mx / matmul_mx_acc.
 *
 * Runs immediately before InsertMxScaleAddr so each chunk gets its own scale
 * address bindings. Operands are already memory-space resolved; tile.slice
 * inherits Left/LeftScale/Right/RightScale via OpRegistry.
 *
 * Chunk size matches the hardware MX pack tile (K=64, 2 scale groups).
 * Packed quant for K>64 is rejected by ExpandMxPackedQuant (kb==1); large-K
 * matmul is handled here instead. Phase0 showed host concat of per-chunk
 * packed scales is not byte-identical to a full-K MX_A_ZZ pack.
 */

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace {

constexpr int64_t kMxChunkK = 64;
constexpr int64_t kMxGroupK = 32;
constexpr int64_t kMxChunkGroups = kMxChunkK / kMxGroupK;  // 2

ExprPtr MakeIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MakeShape2(int64_t d0, int64_t d1, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(d0, span), MakeIndex(d1, span)}, span);
}

enum class MxMatmulKind { kBase, kAcc, kBias };

std::optional<MxMatmulKind> ClassifyMxMatmul(const CallPtr& call) {
  if (IsOp(call, "tile.matmul_mx")) return MxMatmulKind::kBase;
  if (IsOp(call, "tile.matmul_mx_acc")) return MxMatmulKind::kAcc;
  if (IsOp(call, "tile.matmul_mx_bias")) return MxMatmulKind::kBias;
  return std::nullopt;
}

struct MxOperands {
  ExprPtr acc;    // only for kAcc
  ExprPtr lhs;
  ExprPtr lhs_scale;
  ExprPtr rhs;
  ExprPtr rhs_scale;
  ExprPtr bias;  // only for kBias
  int64_t m = 0;
  int64_t k = 0;
  int64_t n = 0;
};

std::optional<MxOperands> ParseMxOperands(const CallPtr& call, MxMatmulKind kind) {
  MxOperands ops;
  size_t lhs_i = 0;
  if (kind == MxMatmulKind::kAcc) {
    if (call->args_.size() != 5) return std::nullopt;
    ops.acc = call->args_[0];
    lhs_i = 1;
  } else if (kind == MxMatmulKind::kBias) {
    if (call->args_.size() != 5) return std::nullopt;
    ops.bias = call->args_[4];
    lhs_i = 0;
  } else if (call->args_.size() != 4) {
    return std::nullopt;
  }

  ops.lhs = call->args_[lhs_i];
  ops.lhs_scale = call->args_[lhs_i + 1];
  ops.rhs = call->args_[lhs_i + 2];
  ops.rhs_scale = call->args_[lhs_i + 3];

  auto lhs_ty = As<TileType>(ops.lhs->GetType());
  auto rhs_ty = As<TileType>(ops.rhs->GetType());
  if (!lhs_ty || !rhs_ty || lhs_ty->shape_.size() != 2 || rhs_ty->shape_.size() != 2) {
    return std::nullopt;
  }
  auto m = As<ConstInt>(lhs_ty->shape_[0]);
  auto k = As<ConstInt>(lhs_ty->shape_[1]);
  auto n = As<ConstInt>(rhs_ty->shape_[1]);
  if (!m || !k || !n) return std::nullopt;
  ops.m = m->value_;
  ops.k = k->value_;
  ops.n = n->value_;
  return ops;
}

bool NeedsSplit(const MxOperands& ops) { return ops.k > kMxChunkK && (ops.k % kMxChunkK) == 0; }

AssignStmtPtr BindSlice(const std::string& name_hint, const ExprPtr& src, int64_t d0, int64_t d1,
                        int64_t off0, int64_t off1, const Span& span) {
  auto& reg = OpRegistry::GetInstance();
  auto call =
      reg.Create("tile.slice", {src, MakeShape2(d0, d1, span), MakeShape2(off0, off1, span)}, {}, span);
  auto var = std::make_shared<Var>(name_hint, call->GetType(), span);
  return std::make_shared<AssignStmt>(var, call, span);
}

class SplitLargeKMxMatmulMutator : public IRMutator {
 public:
  StmtPtr RewriteBody(const StmtPtr& body) {
    if (As<SeqStmts>(body)) return VisitStmt(body);

    auto visited = VisitStmt(body);
    auto assign = As<AssignStmt>(visited);
    auto call = assign ? As<Call>(assign->value_) : nullptr;
    if (!assign || !call) return visited;

    std::vector<StmtPtr> stmts;
    if (!TryRewriteAssign(assign, call, stmts)) return visited;
    return SeqStmts::Flatten(std::move(stmts), body->span_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    bool changed = false;
    std::vector<StmtPtr> new_stmts;
    new_stmts.reserve(op->stmts_.size());

    for (const auto& stmt : op->stmts_) {
      auto rewritten = IRMutator::VisitStmt(stmt);
      if (rewritten.get() != stmt.get()) changed = true;

      auto assign = As<AssignStmt>(rewritten);
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (assign && call) {
        std::vector<StmtPtr> expansion;
        if (TryRewriteAssign(assign, call, expansion)) {
          changed = true;
          new_stmts.insert(new_stmts.end(), expansion.begin(), expansion.end());
          continue;
        }
      }
      new_stmts.push_back(rewritten);
    }

    if (!changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto new_then = RewriteBody(op->then_body_);
    std::optional<StmtPtr> new_else = op->else_body_;
    if (op->else_body_.has_value()) new_else = RewriteBody(op->else_body_.value());
    const bool then_changed = new_then.get() != op->then_body_.get();
    const bool else_changed = op->else_body_.has_value() && new_else->get() != op->else_body_->get();
    if (!then_changed && !else_changed) return op;
    auto result = MutableCopy(op);
    result->then_body_ = std::move(new_then);
    result->else_body_ = std::move(new_else);
    return result;
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override { return VisitLoop(op, op->body_); }
  StmtPtr VisitStmt_(const WhileStmtPtr& op) override { return VisitLoop(op, op->body_); }

 private:
  template <typename LoopPtr>
  StmtPtr VisitLoop(const LoopPtr& op, const StmtPtr& body) {
    auto new_body = RewriteBody(body);
    if (new_body.get() == body.get()) return op;
    auto result = MutableCopy(op);
    result->body_ = std::move(new_body);
    return result;
  }

  bool TryRewriteAssign(const AssignStmtPtr& assign, const CallPtr& call, std::vector<StmtPtr>& out) {
    auto kind = ClassifyMxMatmul(call);
    if (!kind.has_value()) return false;
    auto ops = ParseMxOperands(call, *kind);
    if (!ops.has_value() || !NeedsSplit(*ops)) return false;

    const Span& span = call->span_;
    const std::string base = assign->var_->name_hint_;
    auto& reg = OpRegistry::GetInstance();
    const int64_t chunks = ops->k / kMxChunkK;

    ExprPtr acc = ops->acc;
    for (int64_t ci = 0; ci < chunks; ++ci) {
      const int64_t k0 = ci * kMxChunkK;
      const int64_t g0 = ci * kMxChunkGroups;
      const std::string prefix = base + "_k" + std::to_string(ci);

      auto lhs = BindSlice(prefix + "_lhs", ops->lhs, ops->m, kMxChunkK, 0, k0, span);
      auto lhs_s = BindSlice(prefix + "_ls", ops->lhs_scale, ops->m, kMxChunkGroups, 0, g0, span);
      auto rhs = BindSlice(prefix + "_rhs", ops->rhs, kMxChunkK, ops->n, k0, 0, span);
      auto rhs_s = BindSlice(prefix + "_rs", ops->rhs_scale, kMxChunkGroups, ops->n, g0, 0, span);
      out.push_back(lhs);
      out.push_back(lhs_s);
      out.push_back(rhs);
      out.push_back(rhs_s);

      CallPtr mx_call;
      if (ci == 0 && *kind == MxMatmulKind::kBase) {
        mx_call = reg.Create("tile.matmul_mx", {lhs->var_, lhs_s->var_, rhs->var_, rhs_s->var_}, {}, span);
      } else if (ci == 0 && *kind == MxMatmulKind::kBias) {
        mx_call = reg.Create("tile.matmul_mx_bias",
                             {lhs->var_, lhs_s->var_, rhs->var_, rhs_s->var_, ops->bias}, {}, span);
      } else {
        INTERNAL_CHECK_SPAN(acc != nullptr, span)
            << "Internal error: SplitLargeKMxMatmul missing accumulator for chunk " << ci;
        mx_call = reg.Create("tile.matmul_mx_acc",
                             {acc, lhs->var_, lhs_s->var_, rhs->var_, rhs_s->var_}, {}, span);
      }

      // Final chunk keeps the original SSA name so downstream uses stay valid.
      const bool is_last = (ci + 1 == chunks);
      VarPtr out_var =
          is_last ? assign->var_ : std::make_shared<Var>(prefix + "_acc", mx_call->GetType(), span);
      out.push_back(std::make_shared<AssignStmt>(out_var, mx_call, span));
      acc = out_var;
    }
    return true;
  }
};

FunctionPtr TransformSplitLargeKMxMatmul(const FunctionPtr& func) {
  SplitLargeKMxMatmulMutator mutator;
  auto new_body = mutator.RewriteBody(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto updated = MutableCopy(func);
  updated->body_ = new_body;
  return updated;
}

}  // namespace

namespace pass {

Pass SplitLargeKMxMatmul() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
    for (const auto& [gvar, func] : program->functions_) {
      if (IsInCoreType(func->func_type_)) {
        new_functions[gvar] = TransformSplitLargeKMxMatmul(func);
      } else {
        new_functions[gvar] = func;
      }
    }
    return std::make_shared<Program>(std::move(new_functions), program->name_, program->span_);
  };
  return CreateProgramPass(pass_func, "SplitLargeKMxMatmul", kSplitLargeKMxMatmulProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
