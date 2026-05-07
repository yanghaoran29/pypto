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

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/inline_call_splice.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// =============================================================================
// Cycle detection in the Inline → Inline call graph
// =============================================================================

class CalledInlineCollector : public IRVisitor {
 public:
  explicit CalledInlineCollector(const std::unordered_set<std::string>& inline_names)
      : inline_names_(inline_names) {}

  void VisitExpr_(const CallPtr& op) override {
    if (op) {
      if (auto gv = As<GlobalVar>(op->op_)) {
        if (inline_names_.count(gv->name_) > 0) {
          called_.insert(gv->name_);
        }
      }
    }
    IRVisitor::VisitExpr_(op);
  }

  std::unordered_set<std::string> called_;

 private:
  const std::unordered_set<std::string>& inline_names_;
};

void DetectInlineCycles(const std::unordered_map<std::string, FunctionPtr>& inline_fns) {
  std::unordered_set<std::string> inline_names;
  for (const auto& [n, _] : inline_fns) inline_names.insert(n);

  std::unordered_map<std::string, std::unordered_set<std::string>> graph;
  for (const auto& [name, fn] : inline_fns) {
    CalledInlineCollector collector(inline_names);
    collector.VisitStmt(fn->body_);
    graph[name] = std::move(collector.called_);
  }

  enum class Color { White, Gray, Black };
  std::unordered_map<std::string, Color> color;
  for (const auto& [n, _] : inline_fns) color[n] = Color::White;
  std::vector<std::string> stack;

  std::function<void(const std::string&)> dfs = [&](const std::string& u) {
    color[u] = Color::Gray;
    stack.push_back(u);
    for (const auto& v : graph[u]) {
      if (color[v] == Color::Gray) {
        std::string cycle;
        bool started = false;
        for (const auto& s : stack) {
          if (s == v) started = true;
          if (started) cycle += s + " -> ";
        }
        cycle += v;
        throw pypto::ValueError("Cycle detected in FunctionType::Inline call graph: " + cycle);
      }
      if (color[v] == Color::White) dfs(v);
    }
    stack.pop_back();
    color[u] = Color::Black;
  };

  for (const auto& [n, _] : inline_fns) {
    if (color.at(n) == Color::White) dfs(n);
  }
}

// =============================================================================
// InlineCallsMutator — walks a function body and replaces top-level inline-call
// statements with the spliced inline body.
// =============================================================================

class InlineCallsMutator : public IRMutator {
 public:
  explicit InlineCallsMutator(const std::unordered_map<std::string, FunctionPtr>& inline_fns)
      : inline_fns_(inline_fns) {}

  bool Changed() const { return changed_; }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool any_changed = false;
    for (const auto& stmt : op->stmts_) {
      auto handled = HandleTopLevelInlineCall(stmt);
      if (handled.has_value()) {
        for (auto& s : *handled) new_stmts.push_back(std::move(s));
        any_changed = true;
        changed_ = true;
        continue;
      }
      auto recursed = VisitStmt(stmt);
      if (recursed.get() != stmt.get()) any_changed = true;
      new_stmts.push_back(recursed);
    }
    if (!any_changed) return op;
    return SeqStmts::Flatten(std::move(new_stmts), op->span_);
  }

  // Bare AssignStmt body — e.g. `if c: x = inline_f(...)` where the IfStmt's
  // then_body is a single AssignStmt, not a SeqStmts. InlineFunctions runs
  // before NormalizeStmtStructure, so non-SeqStmts bodies are possible. Wrap
  // the splice in a SeqStmts so the parent body remains a single Stmt;
  // SeqStmts::Flatten collapses any redundant nesting later.
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto handled = HandleTopLevelInlineCall(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto handled = HandleTopLevelInlineCall(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

 private:
  // Recognise `LHS = inline_call(args...)` or `EvalStmt(inline_call(args...))`
  // and return the spliced sequence; otherwise return std::nullopt.
  std::optional<std::vector<StmtPtr>> HandleTopLevelInlineCall(const StmtPtr& stmt) {
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (!call) return std::nullopt;
    auto callee = LookupInlineCallee(call);
    if (!callee) return std::nullopt;

    if (auto assign = As<AssignStmt>(stmt)) {
      return inline_splice::SpliceInlineCall(callee, call->args_, assign->var_, assign->span_);
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      return inline_splice::SpliceInlineCall(callee, call->args_, /*lhs=*/nullptr, eval->span_);
    }
    return std::nullopt;
  }

  FunctionPtr LookupInlineCallee(const CallPtr& call) const {
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return nullptr;
    auto it = inline_fns_.find(gv->name_);
    return (it == inline_fns_.end()) ? nullptr : it->second;
  }

  const std::unordered_map<std::string, FunctionPtr>& inline_fns_;
  bool changed_ = false;
};

}  // namespace

namespace pass {

/**
 * @brief Pass that eliminates FunctionType::Inline functions by splicing their
 *        bodies at every call site.
 *
 * Runs as the first pipeline pass. Subsequent passes never observe Inline
 * functions or Calls to them.
 *
 * Algorithm:
 *  1. Collect all FunctionType::Inline functions in the program.
 *  2. Detect cycles in the Inline → Inline call graph (raise on cycle).
 *  3. Iterate all non-Inline AND Inline functions, splicing top-level
 *     `LHS = inline_call(...)` or `EvalStmt(inline_call(...))` statements
 *     with the inlined body (alpha-rename + param substitution).
 *  4. Repeat (3) to fixpoint so that Inline-calls-Inline is fully expanded.
 *  5. Drop all Inline functions from the program.
 *
 * Edge cases:
 *  - Multi-return inline: emits `LHS = MakeTuple([rets...])`. Subsequent
 *    Simplify can fold `TupleGetItemExpr(MakeTuple(...), i)` if needed.
 *  - Nested Call to inline (e.g. inside a binary expression) is left alone in
 *    v1; the verifier flags any surviving Calls to Inline functions.
 *  - Inline function with no callers is silently dropped in step (5) — that
 *    naturally covers the "Inline function as program entry" case too: with
 *    no Call sites it just disappears in the cleanup phase.
 *  - Inline body containing a non-trailing ReturnStmt is rejected at splice
 *    time with a CHECK (only a single trailing return is supported).
 */
Pass InlineFunctions() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    // Collect inline functions
    std::unordered_map<std::string, FunctionPtr> inline_fns;
    for (const auto& [gvar, fn] : program->functions_) {
      if (fn->func_type_ == FunctionType::Inline) {
        CHECK(inline_fns.count(fn->name_) == 0)
            << "Duplicate FunctionType::Inline function name '" << fn->name_ << "' in program";
        inline_fns[fn->name_] = fn;
      }
    }

    // Fast path: nothing to do
    if (inline_fns.empty()) return program;

    // Cycle detection
    DetectInlineCycles(inline_fns);

    // Iterate to fixpoint. Each iteration mutates every function (incl. Inline
    // ones, so that Inline-calls-Inline expands too). The loop terminates after
    // at most (inline_fns.size() + 1) iterations because each iteration either
    // makes progress or hits the fixpoint.
    std::unordered_map<std::string, FunctionPtr> current;
    for (const auto& [gvar, fn] : program->functions_) {
      current[fn->name_] = fn;
    }

    const size_t max_iters = inline_fns.size() + 1;
    for (size_t iter = 0; iter < max_iters; ++iter) {
      bool any_changed = false;

      // Refresh inline_fns view to point at the *latest* bodies — important
      // because a previous iteration may have inlined Inline-calls-Inline.
      std::unordered_map<std::string, FunctionPtr> latest_inline;
      for (const auto& [name, fn] : inline_fns) {
        latest_inline[name] = current[name];
      }

      for (auto& [name, fn] : current) {
        InlineCallsMutator mutator(latest_inline);
        auto new_body = mutator.VisitStmt(fn->body_);
        if (mutator.Changed()) {
          auto updated = MutableCopy(fn);
          updated->body_ = new_body;
          fn = updated;
          any_changed = true;
        }
      }

      if (!any_changed) break;

      INTERNAL_CHECK(iter + 1 < max_iters) << "InlineFunctions did not reach a fixpoint within " << max_iters
                                           << " iterations; this indicates a bug or an undetected cycle";
    }

    // Drop inline functions and rebuild the program
    std::vector<FunctionPtr> kept_functions;
    for (const auto& [gvar, fn] : program->functions_) {
      auto it = current.find(fn->name_);
      INTERNAL_CHECK(it != current.end()) << "Internal error: function '" << fn->name_ << "' missing";
      const auto& latest = it->second;
      if (latest->func_type_ == FunctionType::Inline) continue;
      kept_functions.push_back(latest);
    }

    return std::make_shared<Program>(kept_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "InlineFunctions", kInlineFunctionsProperties);
}

}  // namespace pass

}  // namespace ir
}  // namespace pypto
