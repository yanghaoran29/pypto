/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License file in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "pypto/ir/transforms/utils/inline_call_splice.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace inline_splice {

namespace {

// Collects Vars whose binding sites must be alpha-renamed at each splice.
//
// We deliberately omit `iter_args_` of For/While loops: the base IRMutator
// already mints fresh IterArg instances per visit (see mutator.cpp:581 / 664).
// Including them here would seed `rename_map_` with entries the base mutator
// later overwrites/erases, leading to inconsistent def-use after the splice.
class DefVarCollector : public IRVisitor {
 public:
  std::unordered_set<const Var*> defs;

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (op->var_) defs.insert(op->var_.get());
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (op->loop_var_) defs.insert(op->loop_var_.get());
    for (const auto& v : op->return_vars_) {
      if (v) defs.insert(v.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    for (const auto& v : op->return_vars_) {
      if (v) defs.insert(v.get());
    }
    IRVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& v : op->return_vars_) {
      if (v) defs.insert(v.get());
    }
    IRVisitor::VisitStmt_(op);
  }
};

class VarSubstituteMutator : public IRMutator {
 public:
  VarSubstituteMutator(std::unordered_map<const Var*, ExprPtr> param_subst,
                       std::unordered_map<const Var*, VarPtr> rename_map)
      : param_subst_(std::move(param_subst)), rename_map_(std::move(rename_map)) {}

  ExprPtr VisitExpr_(const VarPtr& op) override {
    auto pit = param_subst_.find(op.get());
    if (pit != param_subst_.end()) return pit->second;
    auto rit = rename_map_.find(op.get());
    if (rit != rename_map_.end()) return rit->second;
    return op;
  }

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto new_value = VisitExpr(op->value_);
    auto rit = rename_map_.find(op->var_.get());
    VarPtr new_var = (rit != rename_map_.end()) ? rit->second : op->var_;
    if (new_var.get() == op->var_.get() && new_value.get() == op->value_.get()) {
      return op;
    }
    return std::make_shared<const AssignStmt>(new_var, new_value, op->span_, op->leading_comments_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto fp = std::dynamic_pointer_cast<const ForStmt>(base);
    INTERNAL_CHECK(fp) << "Internal error: VisitStmt_(ForStmtPtr) must return a ForStmt";

    auto loop_var_renamed = rename_map_.find(fp->loop_var_.get());
    auto new_return_vars = RenameVarVec(fp->return_vars_);
    bool any_renamed = (loop_var_renamed != rename_map_.end()) || new_return_vars.has_value();
    if (!any_renamed) return fp;

    auto result = MutableCopy(fp);
    if (loop_var_renamed != rename_map_.end()) result->loop_var_ = loop_var_renamed->second;
    if (new_return_vars.has_value()) result->return_vars_ = std::move(*new_return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto wp = std::dynamic_pointer_cast<const WhileStmt>(base);
    INTERNAL_CHECK(wp) << "Internal error: VisitStmt_(WhileStmtPtr) must return a WhileStmt";

    auto new_return_vars = RenameVarVec(wp->return_vars_);
    if (!new_return_vars.has_value()) return wp;
    auto result = MutableCopy(wp);
    result->return_vars_ = std::move(*new_return_vars);
    return result;
  }

  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    auto base = IRMutator::VisitStmt_(op);
    auto ip = std::dynamic_pointer_cast<const IfStmt>(base);
    INTERNAL_CHECK(ip) << "Internal error: VisitStmt_(IfStmtPtr) must return an IfStmt";

    auto new_return_vars = RenameVarVec(ip->return_vars_);
    if (!new_return_vars.has_value()) return ip;
    auto result = MutableCopy(ip);
    result->return_vars_ = std::move(*new_return_vars);
    return result;
  }

 private:
  std::optional<std::vector<VarPtr>> RenameVarVec(const std::vector<VarPtr>& vars) const {
    bool any_renamed = false;
    std::vector<VarPtr> result;
    result.reserve(vars.size());
    for (const auto& v : vars) {
      auto rit = rename_map_.find(v.get());
      if (rit != rename_map_.end()) {
        result.push_back(rit->second);
        any_renamed = true;
      } else {
        result.push_back(v);
      }
    }
    if (!any_renamed) return std::nullopt;
    return result;
  }

  std::unordered_map<const Var*, ExprPtr> param_subst_;
  std::unordered_map<const Var*, VarPtr> rename_map_;
};

std::string FreshName(const std::string& orig) {
  static int counter = 0;
  return orig + "_inline" + std::to_string(counter++);
}

class NestedReturnCounter : public IRVisitor {
 public:
  int count = 0;
  void VisitStmt_(const ReturnStmtPtr& op) override {
    ++count;
    IRVisitor::VisitStmt_(op);
  }
};

}  // namespace

std::vector<StmtPtr> SpliceInlineCall(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                    const VarPtr& lhs, const Span& call_site_span) {
  CHECK(callee->params_.size() == args.size())
      << "Inline call to '" << callee->name_ << "' has " << args.size() << " argument(s) but callee expects "
      << callee->params_.size();

  std::unordered_map<const Var*, ExprPtr> param_subst;
  for (size_t i = 0; i < callee->params_.size(); ++i) {
    param_subst[callee->params_[i].get()] = args[i];
  }

  DefVarCollector def_collector;
  def_collector.VisitStmt(callee->body_);
  std::unordered_map<const Var*, VarPtr> rename_map;
  for (const Var* v : def_collector.defs) {
    if (param_subst.count(v) > 0) continue;
    auto fresh = std::make_shared<Var>(FreshName(v->name_hint_), v->GetType(), v->span_);
    rename_map[v] = fresh;
  }

  VarSubstituteMutator mutator(param_subst, rename_map);
  StmtPtr renamed_body = mutator.VisitStmt(callee->body_);

  std::vector<StmtPtr> spliced;
  std::vector<ExprPtr> return_values;
  bool has_return = false;

  auto extract_from_stmt = [&](const StmtPtr& s) {
    auto seq = std::dynamic_pointer_cast<const SeqStmts>(s);
    if (!seq) {
      auto ret = std::dynamic_pointer_cast<const ReturnStmt>(s);
      if (ret) {
        return_values = ret->value_;
        has_return = true;
      } else {
        spliced.push_back(s);
      }
      return;
    }
    for (const auto& sub : seq->stmts_) {
      auto ret = std::dynamic_pointer_cast<const ReturnStmt>(sub);
      if (ret) {
        return_values = ret->value_;
        has_return = true;
        break;
      }
      spliced.push_back(sub);
    }
  };
  extract_from_stmt(renamed_body);

  NestedReturnCounter post_extract;
  for (const auto& s : spliced) post_extract.VisitStmt(s);
  CHECK(post_extract.count == 0) << "Inline function '" << callee->name_
                                 << "' contains a non-trailing ReturnStmt; only a single trailing return is "
                                    "supported (early-return inside an If/For/While branch is rejected)";

  if (lhs) {
    CHECK(has_return) << "Inline function '" << callee->name_
                      << "' is called for its value but has no return statement";
    ExprPtr final_value;
    if (return_values.size() == 1) {
      final_value = return_values[0];
    } else {
      final_value = std::make_shared<MakeTuple>(return_values, call_site_span);
    }
    if (auto var_expr = As<Var>(final_value); var_expr && var_expr.get() == lhs.get()) {
      return spliced;
    }
    spliced.push_back(std::make_shared<const AssignStmt>(lhs, final_value, call_site_span));
  }

  return spliced;
}

}  // namespace inline_splice
}  // namespace ir
}  // namespace pypto
