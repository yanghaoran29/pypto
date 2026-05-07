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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/inline_call_splice.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/scope_outline_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

/// True when \p callee_fn is an outlined InCore name nested under \p parent_fn
/// (e.g. ``qwen3_decode_incore_13_incore_0`` under ``qwen3_decode_incore_13``).
bool IsNestedOutlinedIncoreCallee(const std::string& parent_fn, const std::string& callee_fn) {
  const std::string prefix = parent_fn + "_incore_";
  return callee_fn.size() > prefix.size() && callee_fn.compare(0, prefix.size(), prefix) == 0;
}

/**
 * Like InlineFunctions' mutator, but splices only ``Call(GlobalVar)`` to
 * ``FunctionType::InCore`` callees whose names are nested under the caller
 * (``parent_incore_N_incore_M``), then records those callees for removal.
 */
class NestedIncoreCallInlineMutator : public IRMutator {
 public:
  NestedIncoreCallInlineMutator(const std::unordered_map<std::string, FunctionPtr>& funcs, const std::string& caller_fn,
                               std::unordered_set<std::string>* inlined_names)
      : funcs_(funcs), caller_fn_(caller_fn), inlined_names_(inlined_names) {}

  [[nodiscard]] bool Changed() const { return changed_; }

  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    std::vector<StmtPtr> new_stmts;
    bool any_changed = false;
    for (const auto& stmt : op->stmts_) {
      auto handled = HandleTopLevel(stmt);
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

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto handled = HandleTopLevel(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    auto handled = HandleTopLevel(op);
    if (!handled.has_value()) return IRMutator::VisitStmt_(op);
    changed_ = true;
    return SeqStmts::Flatten(std::move(*handled), op->span_);
  }

 private:
  std::optional<std::vector<StmtPtr>> HandleTopLevel(const StmtPtr& stmt) {
    auto call = transform_utils::GetCallFromStmt(stmt);
    if (!call) return std::nullopt;
    auto callee = LookupNestedIncoreCallee(call);
    if (!callee) return std::nullopt;
    inlined_names_->insert(callee->name_);
    if (auto assign = As<AssignStmt>(stmt)) {
      return inline_splice::SpliceInlineCall(callee, call->args_, assign->var_, assign->span_);
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      return inline_splice::SpliceInlineCall(callee, call->args_, /*lhs=*/nullptr, eval->span_);
    }
    return std::nullopt;
  }

  FunctionPtr LookupNestedIncoreCallee(const CallPtr& call) const {
    auto gv = As<GlobalVar>(call->op_);
    if (!gv) return nullptr;
    auto it = funcs_.find(gv->name_);
    if (it == funcs_.end()) return nullptr;
    const auto& fn = it->second;
    if (fn->func_type_ != FunctionType::InCore) return nullptr;
    if (!IsNestedOutlinedIncoreCallee(caller_fn_, gv->name_)) return nullptr;
    return fn;
  }

  const std::unordered_map<std::string, FunctionPtr>& funcs_;
  std::string caller_fn_;
  std::unordered_set<std::string>* inlined_names_;
  bool changed_ = false;
};

void InlineNestedIncoreCalleesIntoParents(std::vector<FunctionPtr>& functions) {
  for (;;) {
    std::unordered_map<std::string, FunctionPtr> by_name;
    for (const auto& f : functions) {
      by_name[f->name_] = f;
    }
    std::unordered_set<std::string> inlined_this_round;
    bool any_body_changed = false;
    for (size_t i = 0; i < functions.size(); ++i) {
      auto f = functions[i];
      if (f->func_type_ != FunctionType::InCore) continue;
      NestedIncoreCallInlineMutator mut(by_name, f->name_, &inlined_this_round);
      StmtPtr nb = mut.VisitStmt(f->body_);
      if (mut.Changed()) {
        auto u = MutableCopy(f);
        u->body_ = std::move(nb);
        functions[i] = u;
        by_name[u->name_] = u;
        any_body_changed = true;
      }
    }
    if (!inlined_this_round.empty()) {
      std::vector<FunctionPtr> kept;
      kept.reserve(functions.size());
      for (const auto& f : functions) {
        if (inlined_this_round.count(f->name_) > 0) continue;
        kept.push_back(f);
      }
      functions = std::move(kept);
    }
    if (!any_body_changed) break;
  }
}

}  // namespace

namespace pass {

/**
 * @brief Pass to outline InCore scopes into separate functions
 *
 * This pass transforms ScopeStmt(InCore) nodes into separate Function(InCore) definitions
 * and replaces the scope with a Call to the outlined function.
 *
 * Requirements:
 * - Input IR must be in SSA form (run ConvertToSSA first)
 * - Processes Opaque and Orchestration functions. Orchestration functions can
 *   carry InCore scopes when the parser desugars high-level constructs
 *   (e.g. ``for i in pl.spmd(...)``) into SpmdScopeStmt(InCoreScopeStmt(...)).
 *
 * Transformation:
 * 1. For each ScopeStmt(InCore) in an Opaque/Orchestration function:
 *    - Analyze body to determine external variable references (inputs)
 *    - Analyze subsequent statements to determine which definitions are outputs
 *    - Extract body into new Function(InCore) with appropriate params/returns
 *    - Replace scope with Call to the outlined function + output assignments
 *    - EvalStmt(store) calls on output tensors are converted to AssignStmt
 * 2. Recursively transforms nested InCore bodies; scopes nested under an InCore
 *    kernel being outlined are spliced inline (no nested InCore callee) so PTO
 *    codegen never sees Call(GlobalVar) to another InCore kernel.
 * 3. Add outlined functions to the program, then inline any remaining nested
 *    InCore callees (``parent_incore_N_incore_M``) into their parent kernels and
 *    drop the leaf functions — covers paths where nested scopes became Calls.
 * 4. Promote Opaque parents to Orchestration when at least one InCore scope is
 *    outlined. Orchestration parents stay Orchestration.
 */
Pass OutlineIncoreScopes() {
  auto pass_func = [](const ProgramPtr& program) -> ProgramPtr {
    std::vector<FunctionPtr> new_functions;
    std::vector<FunctionPtr> all_outlined_functions;

    for (const auto& [gvar, func] : program->functions_) {
      // Process Opaque and Orchestration functions; other function types
      // (InCore/Group/Spmd) are already outlined or not expected to carry
      // InCore scopes.
      if (func->func_type_ != FunctionType::Opaque && func->func_type_ != FunctionType::Orchestration) {
        new_functions.push_back(func);
        continue;
      }

      // Build symbol table for this function
      outline_utils::VarCollector type_collector;
      for (const auto& var : func->params_) {
        type_collector.var_types[var.get()] = var->GetType();
        type_collector.var_objects[var.get()] = var;
        type_collector.known_names.insert(var->name_hint_);
      }
      type_collector.VisitStmt(func->body_);

      // Outline InCore scopes in this function
      outline_utils::ScopeOutliner outliner(func->name_, type_collector.var_types, type_collector.var_objects,
                                            type_collector.known_names, ScopeKind::InCore,
                                            FunctionType::InCore, "_incore_");
      auto new_body = outliner.VisitStmt(func->body_);

      // Create new function with transformed body.
      // If any InCore scopes were outlined, promote Opaque -> Orchestration.
      const auto& outlined = outliner.GetOutlinedFunctions();
      FunctionType new_func_type = outlined.empty() ? func->func_type_ : FunctionType::Orchestration;
      auto new_func = MutableCopy(func);
      new_func->body_ = new_body;
      new_func->func_type_ = new_func_type;
      if (new_func_type == FunctionType::Orchestration) {
        new_func->level_ = FunctionTypeToLevel(new_func_type);
        new_func->role_ = Role::Orchestrator;
      }
      new_functions.push_back(new_func);

      // Collect outlined functions (prepend before parent so inner functions come first)
      all_outlined_functions.insert(all_outlined_functions.end(), outlined.begin(), outlined.end());
    }

    // Add all outlined functions before the originals
    all_outlined_functions.insert(all_outlined_functions.end(), new_functions.begin(), new_functions.end());

    // PTO codegen does not lower ``Call(GlobalVar)`` from one InCore kernel into
    // another. If nested outlining still produced ``parent_incore_N_incore_M``
    // callees, splice them into the parent body and drop the leaf functions.
    InlineNestedIncoreCalleesIntoParents(all_outlined_functions);

    // Create new program with all functions
    return std::make_shared<Program>(all_outlined_functions, program->name_, program->span_);
  };

  return CreateProgramPass(pass_func, "OutlineIncoreScopes", kOutlineIncoreScopesProperties);
}

}  // namespace pass

// ============================================================================
// SplitIncoreOrch property verifier
// ============================================================================

namespace {

/**
 * @brief Checks no InCore ScopeStmts remain in Opaque or Orchestration functions.
 */
using SplitIncoreOrchVerifier = outline_utils::ScopeKindAbsenceVerifier<ScopeKind::InCore>;

static bool IsComputeTensorOp(const std::string& op_name) {
  return transform_utils::IsComputeTensorOp(op_name);
}

/// Checks Orchestration functions for compute tensor ops that should be in InCore.
class OrchComputeTensorOpVerifier : public IRVisitor {
 public:
  explicit OrchComputeTensorOpVerifier(std::vector<Diagnostic>& diagnostics) : diagnostics_(diagnostics) {}

  void VisitExpr_(const CallPtr& op) override {
    if (op && op->op_ && IsComputeTensorOp(op->op_->name_)) {
      diagnostics_.emplace_back(DiagnosticSeverity::Warning, "SplitIncoreOrch", 0,
                                "Compute tensor op '" + op->op_->name_ +
                                    "' found in Orchestration function (should be inside InCore)",
                                op->span_);
    }
    IRVisitor::VisitExpr_(op);
  }

 private:
  std::vector<Diagnostic>& diagnostics_;
};

}  // namespace

class SplitIncoreOrchPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "SplitIncoreOrch"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [gv, func] : program->functions_) {
      if (!func || !func->body_) continue;
      // Check Opaque and Orchestration functions — InCore functions are expected to have InCore content
      if (func->func_type_ == FunctionType::InCore) continue;
      SplitIncoreOrchVerifier verifier(
          diagnostics, "SplitIncoreOrch",
          "InCore ScopeStmt found in non-InCore function (should have been outlined)");
      verifier.VisitStmt(func->body_);
      // Also check Orchestration functions for leaked compute tensor ops
      if (func->func_type_ == FunctionType::Orchestration) {
        OrchComputeTensorOpVerifier compute_verifier(diagnostics);
        compute_verifier.VisitStmt(func->body_);
      }
    }
  }
};

PropertyVerifierPtr CreateSplitIncoreOrchPropertyVerifier() {
  return std::make_shared<SplitIncoreOrchPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
