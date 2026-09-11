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

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/deep_clone_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/pipeline_loop_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"

namespace pypto {
namespace ir {

namespace {

using pipeline_loop::CloneLoopVar;
using pipeline_loop::MakeConstIndex;
using pipeline_loop::MakeFreshIterArg;
using pipeline_loop::SplitBodyYield;
using transform_utils::ComputeStaticTripCount;
using transform_utils::EvalConstInt;

bool IsMxAccWithInitCond(const CallPtr& call) {
  return call && call->op_->name_ == "tile.matmul_mx_acc" && call->args_.size() == 6;
}

bool IsLoopStartPredicate(const ExprPtr& expr, const VarPtr& loop_var, const ExprPtr& start) {
  auto eq = As<Eq>(expr);
  if (!eq) return false;
  return (AreExprsEqual(eq->left_, loop_var) && AreExprsEqual(eq->right_, start)) ||
         (AreExprsEqual(eq->right_, loop_var) && AreExprsEqual(eq->left_, start));
}

class InitCondFinder : public IRVisitor {
 public:
  InitCondFinder(VarPtr loop_var, ExprPtr start) : loop_var_(std::move(loop_var)), start_(std::move(start)) {}

  [[nodiscard]] bool found() const { return found_; }

 protected:
  void VisitExpr_(const CallPtr& op) override {
    if (IsMxAccWithInitCond(op) && IsLoopStartPredicate(op->args_[5], loop_var_, start_)) found_ = true;
    IRVisitor::VisitExpr_(op);
  }

 private:
  VarPtr loop_var_;
  ExprPtr start_;
  bool found_ = false;
};

class InitCondSpecializer : public IRMutator {
 public:
  InitCondSpecializer(bool initializing, VarPtr loop_var, ExprPtr start)
      : initializing_(initializing), loop_var_(std::move(loop_var)), start_(std::move(start)) {}

 protected:
  ExprPtr VisitExpr_(const CallPtr& op) override {
    auto visited = As<Call>(IRMutator::VisitExpr_(op));
    INTERNAL_CHECK_SPAN(visited, op->span_) << "Internal error: Call mutation returned a non-Call";
    if (!IsMxAccWithInitCond(visited) || !IsLoopStartPredicate(visited->args_[5], loop_var_, start_)) {
      return visited;
    }

    std::vector<ExprPtr> args;
    OpPtr target_op;
    if (initializing_) {
      target_op = OpRegistry::GetInstance().GetOp("tile.matmul_mx");
      args.assign(visited->args_.begin() + 1, visited->args_.begin() + 5);
    } else {
      target_op = visited->op_;
      args.assign(visited->args_.begin(), visited->args_.begin() + 5);
    }
    return std::make_shared<Call>(target_op, std::move(args), visited->kwargs_, visited->attrs_,
                                  visited->GetType(), visited->span_);
  }

 private:
  bool initializing_;
  VarPtr loop_var_;
  ExprPtr start_;
};

void AttachLoopComments(const StmtPtr& body, const std::vector<std::string>& comments) {
  if (comments.empty()) return;
  StmtPtr first = body;
  while (auto seq = As<SeqStmts>(first)) {
    if (seq->stmts_.empty()) return;
    first = seq->stmts_.front();
  }
  std::vector<std::string> merged = comments;
  merged.insert(merged.end(), first->leading_comments_.begin(), first->leading_comments_.end());
  AttachLeadingComments(first, std::move(merged));
}

class PeelMatmulMxInitCondMutator : public IRMutator {
 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto visited = As<ForStmt>(IRMutator::VisitStmt_(op));
    INTERNAL_CHECK_SPAN(visited, op->span_) << "Internal error: ForStmt mutation returned a non-ForStmt";

    auto start = EvalConstInt(visited->start_);
    auto stop = EvalConstInt(visited->stop_);
    auto step = EvalConstInt(visited->step_);
    if (!start.has_value() || !stop.has_value() || !step.has_value() || *step == 0) return visited;

    const int64_t trip_count = ComputeStaticTripCount(*start, *stop, *step);
    if (trip_count == 0) return visited;

    InitCondFinder finder(visited->loop_var_, visited->start_);
    finder.VisitStmt(visited->body_);
    if (!finder.found()) return visited;

    std::unordered_map<const Var*, ExprPtr> first_substitutions;
    first_substitutions.emplace(visited->loop_var_.get(), visited->start_);
    for (const auto& iter_arg : visited->iter_args_) {
      first_substitutions.emplace(iter_arg.get(), iter_arg->initValue_);
    }
    InitCondSpecializer init_specializer(/*initializing=*/true, visited->loop_var_, visited->start_);
    auto first_template = init_specializer.VisitStmt(visited->body_);
    auto first_clone = DeepClone(first_template, first_substitutions, /*clone_def_vars=*/true);
    auto [first_stmts, first_yields] = SplitBodyYield(first_clone.cloned_body);
    AttachLoopComments(first_stmts, visited->leading_comments_);
    INTERNAL_CHECK_SPAN(first_yields.size() == visited->iter_args_.size(), visited->span_)
        << "Internal error: peeled MX loop body must yield one value per iter_arg";

    std::vector<StmtPtr> replacement{first_stmts};
    if (trip_count == 1) {
      for (size_t i = 0; i < visited->return_vars_.size(); ++i) {
        replacement.push_back(
            std::make_shared<AssignStmt>(visited->return_vars_[i], first_yields[i], visited->span_));
      }
      return SeqStmts::Flatten(std::move(replacement), visited->span_);
    }

    VarPtr remainder_var = CloneLoopVar(visited->loop_var_);
    std::vector<IterArgPtr> remainder_iter_args;
    std::unordered_map<const Var*, ExprPtr> remainder_substitutions;
    remainder_substitutions.emplace(visited->loop_var_.get(), remainder_var);
    remainder_iter_args.reserve(visited->iter_args_.size());
    for (size_t i = 0; i < visited->iter_args_.size(); ++i) {
      auto fresh = MakeFreshIterArg(visited->iter_args_[i], first_yields[i]);
      remainder_substitutions.emplace(visited->iter_args_[i].get(), fresh);
      remainder_iter_args.push_back(std::move(fresh));
    }

    InitCondSpecializer acc_specializer(/*initializing=*/false, visited->loop_var_, visited->start_);
    auto remainder_template = acc_specializer.VisitStmt(visited->body_);
    auto remainder_clone = DeepClone(remainder_template, remainder_substitutions, /*clone_def_vars=*/true);
    auto remainder_body = remainder_clone.cloned_body;
    auto remainder_start = MakeConstIndex(*start + *step, visited->span_);
    auto remainder = std::make_shared<ForStmt>(
        remainder_var, remainder_start, visited->stop_, visited->step_, std::move(remainder_iter_args),
        remainder_body, visited->return_vars_, visited->span_, visited->kind_, visited->attrs_);
    replacement.push_back(remainder);
    return SeqStmts::Flatten(std::move(replacement), visited->span_);
  }
};

FunctionPtr TransformPeelMatmulMxInitCond(const FunctionPtr& func) {
  INTERNAL_CHECK(func) << "PeelMatmulMxInitCond cannot run on null function";
  PeelMatmulMxInitCondMutator mutator;
  auto new_body = mutator.VisitStmt(func->body_);
  if (new_body.get() == func->body_.get()) return func;
  auto new_func = MutableCopy(func);
  new_func->body_ = std::move(new_body);
  return new_func;
}

}  // namespace

namespace pass {

Pass PeelMatmulMxInitCond() {
  return CreateFunctionPass(TransformPeelMatmulMxInitCond, "PeelMatmulMxInitCond",
                            kPeelMatmulMxInitCondProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
