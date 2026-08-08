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
 * @file legalize_mixed_mx_scale_via_gm_pass.cpp
 * @brief Force mixed-kernel MX E8M0 scales through GM instead of V2C.
 *
 * Runs immediately after ExpandMxPackedQuant. Rewrites
 * ``tile.tpush_to_aic`` / ``tile.tpop_from_aiv`` of ``FP8E8M0`` (A-scale) into
 * ``tile.store`` + ``tensor.view(MX_A_ZZ)`` + ``tile.load(Mat)``, reusing the
 * packed ZZ layout already produced by ExpandMxPackedQuant / ``quant_mx``.
 * FP8 data V2C traffic is left unchanged.
 */

#include <algorithm>
#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;

constexpr const char* kMxScaleGmName = "__mx_a_scale_gm";

ExprPtr MakeIndex(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr MakeShape2(int64_t d0, int64_t d1, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{MakeIndex(d0, span), MakeIndex(d1, span)}, span);
}

std::optional<int64_t> ConstPipeId(const CallPtr& call) {
  for (const auto& [key, value] : call->kwargs_) {
    if (key != "id") continue;
    try {
      return AnyCast<int64_t>(value, "kwarg key: id");
    } catch (const TypeError&) {
    }
    try {
      return static_cast<int64_t>(AnyCast<int>(value, "kwarg key: id"));
    } catch (const TypeError&) {
    }
    try {
      auto expr = AnyCast<ExprPtr>(value, "kwarg key: id");
      if (auto c = As<ConstInt>(expr)) return c->value_;
    } catch (const TypeError&) {
    }
  }
  return 0;  // default frontend pipe id
}

const TileType* AsTileType(const TypePtr& type) {
  if (!type) return nullptr;
  return dynamic_cast<const TileType*>(type.get());
}

bool IsFp8E8M0Tile(const ExprPtr& expr) {
  auto* tt = AsTileType(expr->GetType());
  return tt && tt->dtype_ == DataType::FP8E8M0;
}

std::optional<int64_t> StaticDim(const ExprPtr& dim) {
  if (auto c = As<ConstInt>(dim)) return c->value_;
  return std::nullopt;
}

std::optional<std::pair<int64_t, int64_t>> StaticShape2(const TileType* tt) {
  if (!tt || tt->shape_.size() != 2) return std::nullopt;
  auto d0 = StaticDim(tt->shape_[0]);
  auto d1 = StaticDim(tt->shape_[1]);
  if (!d0 || !d1) return std::nullopt;
  return std::make_pair(*d0, *d1);
}

struct ScalePushSite {
  int64_t pipe_id = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t nelems = 0;
};

struct ScalePopSite {
  int64_t pipe_id = 0;
  int64_t rows = 0;  // logical M for MX_A_ZZ view
  int64_t cols = 0;  // logical K/32
  int64_t nelems = 0;
  const Var* result_var = nullptr;
};

void WalkStmts(const std::vector<StmtPtr>& stmts, const std::function<void(const StmtPtr&)>& fn) {
  for (const auto& stmt : stmts) {
    fn(stmt);
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      WalkStmts(FlattenBody(for_stmt->body_), fn);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      WalkStmts(FlattenBody(if_stmt->then_body_), fn);
      if (if_stmt->else_body_.has_value()) WalkStmts(FlattenBody(*if_stmt->else_body_), fn);
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      WalkStmts(FlattenBody(while_stmt->body_), fn);
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      WalkStmts(FlattenBody(scope->body_), fn);
    }
  }
}

std::vector<ScalePushSite> CollectScalePushes(const FunctionPtr& func) {
  std::vector<ScalePushSite> sites;
  if (!func || !func->body_ || func->func_type_ != FunctionType::AIV) return sites;
  WalkStmts(FlattenBody(func->body_), [&](const StmtPtr& stmt) {
    ExprPtr expr;
    if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      expr = eval->expr_;
    } else if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      expr = assign->value_;
    }
    auto call = As<Call>(expr);
    if (!call || !IsOp(call, "tile.tpush_to_aic") || call->args_.empty()) return;
    if (!IsFp8E8M0Tile(call->args_[0])) return;
    auto* tt = AsTileType(call->args_[0]->GetType());
    auto shape = StaticShape2(tt);
    CHECK(shape) << "LegalizeMixedMxScaleViaGm requires static E8M0 tpush shape in " << func->name_;
    ScalePushSite site;
    site.pipe_id = ConstPipeId(call).value_or(0);
    site.rows = shape->first;
    site.cols = shape->second;
    site.nelems = site.rows * site.cols;
    // Packed ZZ from ExpandMxPackedQuant / quant_mx is [1, M*K/32].
    if (site.rows == 1) {
      // already packed
    } else {
      // Logical Mat [M, K/32] pushed as E8M0 — store as packed [1, M*K/32].
      site.nelems = site.rows * site.cols;
    }
    sites.push_back(site);
  });
  return sites;
}

std::vector<ScalePopSite> CollectScalePops(const FunctionPtr& func) {
  std::vector<ScalePopSite> sites;
  if (!func || !func->body_ || func->func_type_ != FunctionType::AIC) return sites;
  WalkStmts(FlattenBody(func->body_), [&](const StmtPtr& stmt) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    if (!assign) return;
    auto call = As<Call>(assign->value_);
    if (!call || !IsOp(call, "tile.tpop_from_aiv")) return;
    auto* tt = AsTileType(assign->var_->GetType());
    if (!tt || tt->dtype_ != DataType::FP8E8M0) return;
    auto shape = StaticShape2(tt);
    CHECK(shape) << "LegalizeMixedMxScaleViaGm requires static E8M0 tpop shape in " << func->name_;
    ScalePopSite site;
    site.pipe_id = ConstPipeId(call).value_or(0);
    site.rows = shape->first;
    site.cols = shape->second;
    site.nelems = site.rows * site.cols;
    site.result_var = assign->var_.get();
    sites.push_back(site);
  });
  return sites;
}

bool HasMxScaleGmParam(const FunctionPtr& func) {
  for (const auto& p : func->params_) {
    if (p->name_hint_ == kMxScaleGmName) return true;
  }
  return false;
}

FunctionPtr AddMxScaleGmParam(const FunctionPtr& func, int64_t nelems, ParamDirection dir) {
  auto gm_type =
      std::make_shared<TensorType>(std::vector<int64_t>{1, nelems}, DataType::FP8E8M0, std::nullopt, std::nullopt);
  auto gm_var = std::make_shared<Var>(kMxScaleGmName, gm_type, func->span_);
  auto new_params = func->params_;
  new_params.push_back(gm_var);
  auto new_dirs = func->param_directions_;
  new_dirs.push_back(dir);
  auto result = MutableCopy(func);
  result->params_ = std::move(new_params);
  result->param_directions_ = std::move(new_dirs);
  return result;
}

VarPtr FindMxScaleGmParam(const FunctionPtr& func) {
  for (const auto& p : func->params_) {
    if (p->name_hint_ == kMxScaleGmName) return p;
  }
  return nullptr;
}

CallPtr CreateMxScaleTensorCreate(int64_t nelems, const Span& span) {
  auto shape = MakeShape2(1, nelems, span);
  return OpRegistry::GetInstance().Create(
      "tensor.create", {shape},
      {{"dtype", std::any(DataType::FP8E8M0)}, {"layout", std::any(TensorLayout::ND)}}, span);
}

/// Rewrite AIV body: E8M0 tpush_to_aic → tile.store into gm_param.
class AivScaleToGmMutator : public IRMutator {
 public:
  AivScaleToGmMutator(VarPtr gm_param, std::unordered_set<int64_t> pipe_ids)
      : gm_param_(std::move(gm_param)), pipe_ids_(std::move(pipe_ids)) {}

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (!call || !IsOp(call, "tile.tpush_to_aic") || call->args_.empty()) {
      return IRMutator::VisitStmt_(op);
    }
    if (!IsFp8E8M0Tile(call->args_[0])) return IRMutator::VisitStmt_(op);
    auto pid = ConstPipeId(call).value_or(0);
    if (!pipe_ids_.count(pid)) return IRMutator::VisitStmt_(op);

    auto& reg = OpRegistry::GetInstance();
    auto span = op->span_;
    auto store = reg.Create("tile.store", {call->args_[0], MakeShape2(0, 0, span), gm_param_}, {}, span);
    auto store_var =
        std::make_shared<Var>("a_s_gm_st_" + std::to_string(tmp_++), gm_param_->GetType(), span);
    return std::make_shared<AssignStmt>(store_var, store, span);
  }

 private:
  VarPtr gm_param_;
  std::unordered_set<int64_t> pipe_ids_;
  int tmp_ = 0;
};

/// Rewrite AIC body: E8M0 tpop_from_aiv → view(MX_A_ZZ)+load(Mat); drop matching tfree.
class AicScaleFromGmMutator : public IRMutator {
 public:
  AicScaleFromGmMutator(VarPtr gm_param, const std::vector<ScalePopSite>& pops)
      : gm_param_(std::move(gm_param)) {
    for (const auto& p : pops) {
      pipe_ids_.insert(p.pipe_id);
      if (p.result_var) pop_vars_.insert(p.result_var);
      // Prefer last pop shape (all chunks share geometry).
      rows_ = p.rows;
      cols_ = p.cols;
    }
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (call && IsOp(call, "tile.tpop_from_aiv")) {
      auto* tt = AsTileType(op->var_->GetType());
      if (tt && tt->dtype_ == DataType::FP8E8M0) {
        auto pid = ConstPipeId(call).value_or(0);
        if (pipe_ids_.count(pid)) {
          return BuildLoadSeq(op);
        }
      }
    }
    return IRMutator::VisitStmt_(op);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto call = As<Call>(op->expr_);
    if (call && IsOp(call, "system.tfree_to_aiv") && !call->args_.empty()) {
      if (auto v = As<Var>(call->args_[0]); v && pop_vars_.count(v.get())) {
        // Drop tfree for scale tiles that no longer come from tpop.
        return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
      }
    }
    return IRMutator::VisitStmt_(op);
  }

 private:
  StmtPtr BuildLoadSeq(const AssignStmtPtr& op) {
    auto& reg = OpRegistry::GetInstance();
    auto span = op->span_;
    std::vector<StmtPtr> stmts;

    // a_s_mx = tensor.view(gm, [rows, cols], layout=MX_A_ZZ)
    auto view_call = reg.Create("tensor.view", {gm_param_, MakeShape2(rows_, cols_, span)},
                                {{"layout", std::any(TensorLayout::MX_A_ZZ)}}, span);
    auto view_var = std::make_shared<Var>("a_s_mx_" + std::to_string(tmp_++), view_call->GetType(), span);
    stmts.push_back(std::make_shared<AssignStmt>(view_var, view_call, span));

    // scale_mat = tile.load(a_s_mx, [0,0], [rows,cols], target_memory=Mat)
    auto shape = MakeShape2(rows_, cols_, span);
    auto load_call = reg.Create("tile.load", {view_var, MakeShape2(0, 0, span), shape, shape},
                                {{"target_memory", std::any(MemorySpace::Mat)}}, span);
    // Preserve the original LHS var so downstream move(LeftScale) keeps working.
    stmts.push_back(std::make_shared<AssignStmt>(op->var_, load_call, span));
    return SeqStmts::Flatten(std::move(stmts), span);
  }

  VarPtr gm_param_;
  std::unordered_set<int64_t> pipe_ids_;
  std::unordered_set<const Var*> pop_vars_;
  int64_t rows_ = 0;
  int64_t cols_ = 0;
  int tmp_ = 0;
};

StmtPtr CloneScopeWithBody(const ScopeStmtPtr& scope, const StmtPtr& body) {
  if (auto incore = As<InCoreScopeStmt>(scope)) {
    return std::make_shared<InCoreScopeStmt>(incore->split_, incore->name_hint_, body, incore->span_,
                                             incore->leading_comments_, incore->attrs_);
  }
  if (auto cluster = As<ClusterScopeStmt>(scope)) {
    return std::make_shared<ClusterScopeStmt>(cluster->name_hint_, body, cluster->span_,
                                              cluster->leading_comments_, cluster->attrs_);
  }
  if (auto hierarchy = As<HierarchyScopeStmt>(scope)) {
    return std::make_shared<HierarchyScopeStmt>(hierarchy->level_, hierarchy->role_, hierarchy->name_hint_,
                                                body, hierarchy->span_, hierarchy->leading_comments_,
                                                hierarchy->attrs_);
  }
  if (auto spmd = As<SpmdScopeStmt>(scope)) {
    return std::make_shared<SpmdScopeStmt>(spmd->core_num_, spmd->sync_start_, spmd->name_hint_, body,
                                           spmd->span_, spmd->leading_comments_, spmd->attrs_);
  }
  if (auto runtime = As<RuntimeScopeStmt>(scope)) {
    return std::make_shared<RuntimeScopeStmt>(runtime->manual_, runtime->name_hint_, body, runtime->span_,
                                              runtime->leading_comments_, runtime->attrs_);
  }
  return scope;
}

void BuildCallGraph(const std::vector<FunctionPtr>& functions,
                    std::unordered_map<std::string, std::unordered_set<std::string>>& callers,
                    std::unordered_map<std::string, std::unordered_set<std::string>>& callees) {
  std::unordered_set<std::string> names;
  for (const auto& f : functions) names.insert(f->name_);
  for (const auto& func : functions) {
    if (!func->body_) continue;
    auto note_call = [&](const ExprPtr& expr) {
      if (!expr) return;
      OpPtr op;
      if (auto call = As<Call>(expr)) op = call->op_;
      else if (auto submit = As<Submit>(expr)) op = submit->op_;
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(op);
      if (gv && names.count(gv->name_)) {
        callees[func->name_].insert(gv->name_);
        callers[gv->name_].insert(func->name_);
      }
    };
    WalkStmts(FlattenBody(func->body_), [&](const StmtPtr& stmt) {
      if (auto a = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
        note_call(a->value_);
      } else if (auto e = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
        note_call(e->expr_);
      } else if (auto r = std::dynamic_pointer_cast<const ReturnStmt>(stmt)) {
        for (const auto& v : r->value_) note_call(v);
      }
    });
  }
}

StmtPtr RewriteCallsAppendGmArg(const StmtPtr& body, const std::unordered_set<std::string>& modified,
                                const VarPtr& gm_param) {
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any = false;
  auto try_rw = [&](const ExprPtr& expr) -> ExprPtr {
    if (!expr) return nullptr;
    if (auto call = std::dynamic_pointer_cast<const Call>(expr)) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(call->op_);
      if (!gv || !modified.count(gv->name_)) return nullptr;
      auto nc = MutableCopy(call);
      nc->args_.push_back(gm_param);
      return nc;
    }
    if (auto submit = std::dynamic_pointer_cast<const Submit>(expr)) {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(submit->op_);
      if (!gv || !modified.count(gv->name_)) return nullptr;
      auto ns = MutableCopy(submit);
      ns->args_.push_back(gm_param);
      return ns;
    }
    return nullptr;
  };
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      if (auto rw = try_rw(assign->value_)) {
        auto na = MutableCopy(assign);
        na->value_ = rw;
        new_stmts.push_back(na);
        any = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      if (auto rw = try_rw(eval->expr_)) {
        auto ne = MutableCopy(eval);
        ne->expr_ = rw;
        new_stmts.push_back(ne);
        any = true;
        continue;
      }
    } else if (auto ret = std::dynamic_pointer_cast<const ReturnStmt>(stmt)) {
      bool ret_changed = false;
      std::vector<ExprPtr> new_vals;
      new_vals.reserve(ret->value_.size());
      for (const auto& v : ret->value_) {
        if (auto rw = try_rw(v)) {
          new_vals.push_back(rw);
          ret_changed = true;
        } else {
          new_vals.push_back(v);
        }
      }
      if (ret_changed) {
        auto nr = MutableCopy(ret);
        nr->value_ = std::move(new_vals);
        new_stmts.push_back(nr);
        any = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteCallsAppendGmArg(for_stmt->body_, modified, gm_param);
      if (nb != for_stmt->body_) {
        auto nf = MutableCopy(for_stmt);
        nf->body_ = nb;
        new_stmts.push_back(nf);
        any = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      auto nt = RewriteCallsAppendGmArg(if_stmt->then_body_, modified, gm_param);
      std::optional<StmtPtr> ne;
      if (if_stmt->else_body_) ne = RewriteCallsAppendGmArg(*if_stmt->else_body_, modified, gm_param);
      if (nt != if_stmt->then_body_ || (ne && if_stmt->else_body_ && *ne != *if_stmt->else_body_)) {
        auto ni = MutableCopy(if_stmt);
        ni->then_body_ = nt;
        ni->else_body_ = ne;
        new_stmts.push_back(ni);
        any = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      auto nb = RewriteCallsAppendGmArg(while_stmt->body_, modified, gm_param);
      if (nb != while_stmt->body_) {
        auto nw = MutableCopy(while_stmt);
        nw->body_ = nb;
        new_stmts.push_back(nw);
        any = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto nb = RewriteCallsAppendGmArg(scope->body_, modified, gm_param);
      if (nb != scope->body_) {
        new_stmts.push_back(CloneScopeWithBody(scope, nb));
        any = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  if (!any) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

StmtPtr RewriteOrchCreateGm(const StmtPtr& body, const std::unordered_set<std::string>& modified,
                            int64_t nelems, const Span& span, int& counter) {
  auto gm_type =
      std::make_shared<TensorType>(std::vector<int64_t>{1, nelems}, DataType::FP8E8M0, std::nullopt, std::nullopt);
  auto stmts = FlattenBody(body);
  std::vector<StmtPtr> new_stmts;
  bool any = false;
  auto try_rw = [&](const ExprPtr& expr) -> std::pair<StmtPtr, ExprPtr> {
    if (!expr) return {};
    auto append = [&](auto&& node) -> std::pair<StmtPtr, ExprPtr> {
      auto gv = std::dynamic_pointer_cast<const GlobalVar>(node->op_);
      if (!gv || !modified.count(gv->name_)) return {};
      auto gm_var =
          std::make_shared<Var>(std::string("mx_a_scale_gm_") + std::to_string(counter++), gm_type, span);
      auto create = CreateMxScaleTensorCreate(nelems, span);
      auto create_stmt = std::make_shared<AssignStmt>(gm_var, create, span);
      auto nn = MutableCopy(node);
      nn->args_.push_back(gm_var);
      return {create_stmt, nn};
    };
    if (auto call = std::dynamic_pointer_cast<const Call>(expr)) return append(call);
    if (auto submit = std::dynamic_pointer_cast<const Submit>(expr)) return append(submit);
    return {};
  };
  for (const auto& stmt : stmts) {
    if (auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt)) {
      auto [create, rw] = try_rw(assign->value_);
      if (rw) {
        new_stmts.push_back(create);
        auto na = MutableCopy(assign);
        na->value_ = rw;
        new_stmts.push_back(na);
        any = true;
        continue;
      }
    } else if (auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt)) {
      auto [create, rw] = try_rw(eval->expr_);
      if (rw) {
        new_stmts.push_back(create);
        auto ne = MutableCopy(eval);
        ne->expr_ = rw;
        new_stmts.push_back(ne);
        any = true;
        continue;
      }
    } else if (auto ret = std::dynamic_pointer_cast<const ReturnStmt>(stmt)) {
      bool ret_changed = false;
      std::vector<ExprPtr> new_vals;
      StmtPtr create_stmt;
      new_vals.reserve(ret->value_.size());
      for (const auto& v : ret->value_) {
        auto [create, rw] = try_rw(v);
        if (rw) {
          create_stmt = create;
          new_vals.push_back(rw);
          ret_changed = true;
        } else {
          new_vals.push_back(v);
        }
      }
      if (ret_changed) {
        if (create_stmt) new_stmts.push_back(create_stmt);
        auto nr = MutableCopy(ret);
        nr->value_ = std::move(new_vals);
        new_stmts.push_back(nr);
        any = true;
        continue;
      }
    }
    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      auto nb = RewriteOrchCreateGm(for_stmt->body_, modified, nelems, span, counter);
      if (nb != for_stmt->body_) {
        auto nf = MutableCopy(for_stmt);
        nf->body_ = nb;
        new_stmts.push_back(nf);
        any = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else if (auto scope = std::dynamic_pointer_cast<const ScopeStmt>(stmt)) {
      auto nb = RewriteOrchCreateGm(scope->body_, modified, nelems, span, counter);
      if (nb != scope->body_) {
        new_stmts.push_back(CloneScopeWithBody(scope, nb));
        any = true;
      } else {
        new_stmts.push_back(stmt);
      }
    } else {
      new_stmts.push_back(stmt);
    }
  }
  if (!any) return body;
  return SeqStmts::Flatten(std::move(new_stmts), body->span_);
}

ProgramPtr TransformLegalizeMixedMxScaleViaGm(const ProgramPtr& program) {
  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size());
  for (const auto& [gvar, func] : program->functions_) {
    functions.push_back(func);
  }

  std::unordered_map<std::string, std::vector<ScalePushSite>> pushes;
  std::unordered_map<std::string, std::vector<ScalePopSite>> pops;
  int64_t nelems = 0;
  std::unordered_set<int64_t> pipe_ids;
  std::unordered_set<std::string> aiv_funcs;
  std::unordered_set<std::string> aic_funcs;

  for (const auto& func : functions) {
    auto ps = CollectScalePushes(func);
    if (!ps.empty()) {
      pushes[func->name_] = ps;
      aiv_funcs.insert(func->name_);
      for (const auto& s : ps) {
        pipe_ids.insert(s.pipe_id);
        nelems = std::max(nelems, s.nelems);
      }
    }
    auto qs = CollectScalePops(func);
    if (!qs.empty()) {
      pops[func->name_] = qs;
      aic_funcs.insert(func->name_);
      for (const auto& s : qs) {
        pipe_ids.insert(s.pipe_id);
        nelems = std::max(nelems, s.nelems);
      }
    }
  }

  if (pushes.empty() && pops.empty()) {
    return program;  // idempotent: already GM / no V2C scale
  }
  CHECK(!pushes.empty() && !pops.empty())
      << "LegalizeMixedMxScaleViaGm: found unpaired E8M0 V2C scale "
      << "(pushes=" << pushes.size() << ", pops=" << pops.size() << ")";
  CHECK(nelems > 0) << "LegalizeMixedMxScaleViaGm: empty scale payload";

  std::unordered_map<std::string, std::unordered_set<std::string>> callers, callees;
  BuildCallGraph(functions, callers, callees);

  std::unordered_set<std::string> needs_param = aiv_funcs;
  needs_param.insert(aic_funcs.begin(), aic_funcs.end());
  std::vector<std::string> worklist(needs_param.begin(), needs_param.end());
  while (!worklist.empty()) {
    auto name = worklist.back();
    worklist.pop_back();
    auto it = callers.find(name);
    if (it == callers.end()) continue;
    for (const auto& caller : it->second) {
      FunctionPtr* fp = nullptr;
      for (auto& f : functions) {
        if (f->name_ == caller) {
          fp = &f;
          break;
        }
      }
      if (!fp) continue;
      if ((*fp)->func_type_ == FunctionType::Orchestration) continue;
      if (needs_param.insert(caller).second) worklist.push_back(caller);
    }
  }

  for (auto& func : functions) {
    if (!needs_param.count(func->name_) || HasMxScaleGmParam(func)) continue;
    ParamDirection dir = ParamDirection::In;
    if (aiv_funcs.count(func->name_) || func->func_type_ == FunctionType::Group) {
      dir = ParamDirection::Out;
    }
    func = AddMxScaleGmParam(func, nelems, dir);
  }

  // Rewrite AIV / AIC bodies.
  for (auto& func : functions) {
    auto gm = FindMxScaleGmParam(func);
    if (!gm) continue;
    if (aiv_funcs.count(func->name_)) {
      AivScaleToGmMutator mut(gm, pipe_ids);
      auto nb = mut.VisitStmt(func->body_);
      auto uf = MutableCopy(func);
      uf->body_ = nb;
      func = uf;
    }
    if (aic_funcs.count(func->name_)) {
      AicScaleFromGmMutator mut(gm, pops[func->name_]);
      auto nb = mut.VisitStmt(func->body_);
      auto uf = MutableCopy(func);
      uf->body_ = nb;
      func = uf;
    }
  }

  // Propagate GM arg through non-orch callers.
  for (auto& func : functions) {
    if (!needs_param.count(func->name_)) continue;
    auto gm = FindMxScaleGmParam(func);
    if (!gm) continue;
    std::unordered_set<std::string> mod;
    auto ci = callees.find(func->name_);
    if (ci != callees.end()) {
      for (const auto& c : ci->second) {
        if (needs_param.count(c)) mod.insert(c);
      }
    }
    if (mod.empty()) continue;
    auto nb = RewriteCallsAppendGmArg(func->body_, mod, gm);
    auto uf = MutableCopy(func);
    uf->body_ = nb;
    func = uf;
  }

  // Orchestration: tensor.create per call site.
  for (auto& func : functions) {
    if (func->func_type_ != FunctionType::Orchestration || !func->body_) continue;
    int counter = 0;
    auto nb = RewriteOrchCreateGm(func->body_, needs_param, nelems, func->span_, counter);
    if (nb != func->body_) {
      auto uf = MutableCopy(func);
      uf->body_ = nb;
      func = uf;
    }
  }

  return std::make_shared<Program>(functions, program->name_, program->span_);
}

}  // namespace

Pass LegalizeMixedMxScaleViaGm() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    return TransformLegalizeMixedMxScaleViaGm(program);
  };
  return CreateProgramPass(pass_func, "LegalizeMixedMxScaleViaGm", kLegalizeMixedMxScaleViaGmProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
