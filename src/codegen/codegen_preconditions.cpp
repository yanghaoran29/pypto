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

#include "pypto/codegen/codegen_preconditions.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/ir_property.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/return_lineage_utils.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace codegen {

using namespace pypto::ir;  // NOLINT(build/namespaces)

namespace {

class DistributedTensorUseCollector : public IRVisitor {
 public:
  bool uses_distributed_tensor = false;
  bool has_comm_domain_scope = false;
  bool has_windowing_ops = false;

 protected:
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override {
    has_comm_domain_scope = true;
    uses_distributed_tensor = true;
    IRVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const VarPtr& var) override {
    if (As<DistributedTensorType>(var->GetType())) {
      uses_distributed_tensor = true;
    }
    IRVisitor::VisitExpr_(var);
  }

  void VisitExpr_(const CallPtr& call) override {
    const std::string& op_name = call->op_->name_;
    if (op_name.rfind("pld.tensor.", 0) == 0 || op_name.rfind("tensor.window", 0) == 0 ||
        op_name.rfind("tensor.alloc_window_buffer", 0) == 0 ||
        op_name.rfind("tensor.window_buffer", 0) == 0 || As<DistributedTensorType>(call->GetType())) {
      uses_distributed_tensor = true;
      has_windowing_ops = true;
    }
    IRVisitor::VisitExpr_(call);
  }
};

using MxParamMap = std::unordered_map<std::string, std::unordered_set<size_t>>;
using ReturnParamMap = std::unordered_map<std::string, std::vector<std::optional<size_t>>>;

class MxLoadParamCollector : public IRVisitor {
 public:
  MxLoadParamCollector(const FunctionPtr& func, const ReturnParamMap& return_params)
      : return_params_(return_params) {
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (AsTensorTypeLike(func->params_[i]->GetType())) origins_[func->params_[i].get()] = i;
    }
  }

  std::unordered_set<size_t> param_indices;

 protected:
  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (AsTensorTypeLike(assign->var_->GetType())) {
      if (auto source = AsVarLike(assign->value_)) {
        auto it = origins_.find(source.get());
        if (it != origins_.end()) origins_[assign->var_.get()] = it->second;
      } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
        auto tuple = AsVarLike(tuple_get->tuple_);
        auto it = tuple ? tuple_origins_.find(tuple.get()) : tuple_origins_.end();
        if (it != tuple_origins_.end() && tuple_get->index_ >= 0 &&
            static_cast<size_t>(tuple_get->index_) < it->second.size()) {
          const auto& origin = it->second[static_cast<size_t>(tuple_get->index_)];
          if (origin.has_value()) origins_[assign->var_.get()] = *origin;
        }
      } else if (auto call = As<Call>(assign->value_)) {
        const bool preserves_source_buffer = IsOp(call, "tensor.view") || IsOp(call, "tensor.reshape") ||
                                             IsOp(call, "tensor.reinterpret_view") ||
                                             IsOp(call, "tensor.transpose") || IsOp(call, "tensor.slice");
        if (preserves_source_buffer && !call->args_.empty()) {
          auto source = AsVarLike(call->args_[0]);
          auto it = source ? origins_.find(source.get()) : origins_.end();
          if (it != origins_.end()) origins_[assign->var_.get()] = it->second;
        } else if (auto callee = As<GlobalVar>(call->op_)) {
          auto returned = return_params_.find(callee->name_);
          if (returned != return_params_.end() && !returned->second.empty() &&
              returned->second[0].has_value() && *returned->second[0] < call->args_.size()) {
            auto source = AsVarLike(call->args_[*returned->second[0]]);
            auto it = source ? origins_.find(source.get()) : origins_.end();
            if (it != origins_.end()) origins_[assign->var_.get()] = it->second;
          }
        }
      }
    } else if (auto tuple_type = As<TupleType>(assign->var_->GetType())) {
      std::vector<std::optional<size_t>> tuple_origins(tuple_type->types_.size());
      if (auto make_tuple = As<MakeTuple>(assign->value_)) {
        const size_t count = std::min(tuple_origins.size(), make_tuple->elements_.size());
        for (size_t i = 0; i < count; ++i) {
          auto element = AsVarLike(make_tuple->elements_[i]);
          auto it = element ? origins_.find(element.get()) : origins_.end();
          if (it != origins_.end()) tuple_origins[i] = it->second;
        }
      } else if (auto call = As<Call>(assign->value_)) {
        if (auto callee = As<GlobalVar>(call->op_)) {
          auto returned = return_params_.find(callee->name_);
          if (returned != return_params_.end()) {
            const size_t count = std::min(tuple_origins.size(), returned->second.size());
            for (size_t i = 0; i < count; ++i) {
              const auto param_idx = returned->second[i];
              if (!param_idx.has_value() || *param_idx >= call->args_.size()) continue;
              auto source = AsVarLike(call->args_[*param_idx]);
              auto it = source ? origins_.find(source.get()) : origins_.end();
              if (it != origins_.end()) tuple_origins[i] = it->second;
            }
          }
        }
      }
      tuple_origins_[assign->var_.get()] = std::move(tuple_origins);
    }
    IRVisitor::VisitStmt_(assign);
  }

  void VisitExpr_(const CallPtr& call) override {
    if (IsOp(call, "tile.load") && !call->args_.empty()) {
      if (auto tensor_type = AsTensorTypeLike(call->args_[0]->GetType());
          tensor_type && tensor_type->tensor_view_.has_value() &&
          IsMxTensorLayout(tensor_type->tensor_view_->layout)) {
        auto source = AsVarLike(call->args_[0]);
        if (source) {
          auto it = origins_.find(source.get());
          if (it != origins_.end()) param_indices.insert(it->second);
        }
      }
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  const ReturnParamMap& return_params_;
  std::unordered_map<const Var*, size_t> origins_;
  std::unordered_map<const Var*, std::vector<std::optional<size_t>>> tuple_origins_;
};

class MxSliceCallVerifier : public IRVisitor {
 public:
  MxSliceCallVerifier(const MxParamMap& mx_params, const ReturnParamMap& return_params)
      : mx_params_(mx_params), return_params_(return_params) {}

 protected:
  void VisitStmt_(const AssignStmtPtr& assign) override {
    bool slice_derived = false;
    if (auto call = As<Call>(assign->value_)) {
      slice_derived = IsOp(call, "tensor.slice");
      if (!slice_derived && AsTensorTypeLike(assign->var_->GetType())) {
        slice_derived = ReturnedValueIsSliceDerived(call->op_, call->args_, 0);
      }
      VerifyCall(call->op_, call->args_, call->span_);
    } else if (auto submit = As<Submit>(assign->value_)) {
      VerifyCall(submit->op_, submit->args_, submit->span_);
      if (AsTensorTypeLike(assign->var_->GetType())) {
        slice_derived = ReturnedValueIsSliceDerived(submit->op_, submit->args_, 0);
      }
    } else if (auto var = AsVarLike(assign->value_)) {
      slice_derived = slice_derived_vars_.count(var.get()) > 0;
    } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
      auto tuple = AsVarLike(tuple_get->tuple_);
      auto it = tuple ? tuple_slice_derived_.find(tuple.get()) : tuple_slice_derived_.end();
      slice_derived = it != tuple_slice_derived_.end() && tuple_get->index_ >= 0 &&
                      static_cast<size_t>(tuple_get->index_) < it->second.size() &&
                      it->second[static_cast<size_t>(tuple_get->index_)];
    } else if (auto make_tuple = As<MakeTuple>(assign->value_)) {
      std::vector<bool> tuple_derived(make_tuple->elements_.size(), false);
      for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
        const auto& element = make_tuple->elements_[i];
        auto var = AsVarLike(element);
        if (var && slice_derived_vars_.count(var.get()) > 0) {
          tuple_derived[i] = true;
        }
      }
      tuple_slice_derived_[assign->var_.get()] = std::move(tuple_derived);
    }
    if (As<TupleType>(assign->var_->GetType())) {
      const auto* args = static_cast<const std::vector<ExprPtr>*>(nullptr);
      OpPtr op;
      if (auto call = As<Call>(assign->value_)) {
        args = &call->args_;
        op = call->op_;
      } else if (auto submit = As<Submit>(assign->value_)) {
        args = &submit->args_;
        op = submit->op_;
      }
      if (args) {
        auto tuple_type = As<TupleType>(assign->var_->GetType());
        std::vector<bool> tuple_derived(tuple_type->types_.size(), false);
        for (size_t i = 0; i < tuple_derived.size(); ++i) {
          tuple_derived[i] = ReturnedValueIsSliceDerived(op, *args, i);
        }
        tuple_slice_derived_[assign->var_.get()] = std::move(tuple_derived);
      }
    }
    if (slice_derived) slice_derived_vars_.insert(assign->var_.get());
    IRVisitor::VisitStmt_(assign);
  }

  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_)) {
      VerifyCall(call->op_, call->args_, call->span_);
    } else if (auto submit = As<Submit>(eval->expr_)) {
      VerifyCall(submit->op_, submit->args_, submit->span_);
    }
    IRVisitor::VisitStmt_(eval);
  }

 private:
  bool ReturnedValueIsSliceDerived(const OpPtr& op, const std::vector<ExprPtr>& args,
                                   size_t return_idx) const {
    auto callee = As<GlobalVar>(op);
    if (!callee) return false;
    auto returned = return_params_.find(callee->name_);
    if (returned == return_params_.end() || return_idx >= returned->second.size()) return false;
    const auto param_idx = returned->second[return_idx];
    if (!param_idx.has_value() || *param_idx >= args.size()) return false;
    auto arg = AsVarLike(args[*param_idx]);
    return arg && slice_derived_vars_.count(arg.get()) > 0;
  }

  void VerifyCall(const OpPtr& op, const std::vector<ExprPtr>& args, const Span& span) {
    auto callee = As<GlobalVar>(op);
    if (!callee) return;
    auto it = mx_params_.find(callee->name_);
    if (it == mx_params_.end()) return;
    for (size_t param_idx : it->second) {
      if (param_idx >= args.size()) continue;
      auto arg = AsVarLike(args[param_idx]);
      CHECK_SPAN(!arg || slice_derived_vars_.count(arg.get()) == 0, span)
          << "MX tile.load parameter " << param_idx << " of function '" << callee->name_
          << "' must be passed a packed top-level tensor, not a tensor.slice-derived view";
      auto tensor_type = AsTensorTypeLike(args[param_idx]->GetType());
      if (tensor_type && tensor_type->tensor_view_.has_value()) {
        const auto& view = *tensor_type->tensor_view_;
        // MX scale GM packs use row-major logical strides (same family as ND);
        // the layout tag selects the hardware load path. Allow ND as well for
        // SSA-forwarded values whose return annotation dropped the MX layout.
        const bool layout_ok = view.layout == TensorLayout::ND || IsMxTensorLayout(view.layout);
        const auto packed = tensor_view_semantics::BuildLogicalStridesFromLayout(tensor_type->shape_,
                                                                                 view.layout);
        CHECK_SPAN(layout_ok &&
                       (view.stride.empty() ||
                        tile_view_semantics::ShapeExprListsEquivalent(view.stride, packed)),
                   span)
            << "MX tile.load parameter " << param_idx << " of function '" << callee->name_
            << "' must be passed a packed MX/ND tensor, not a strided or incompatible TensorView";
      }
    }
  }

  const MxParamMap& mx_params_;
  const ReturnParamMap& return_params_;
  std::unordered_set<const Var*> slice_derived_vars_;
  std::unordered_map<const Var*, std::vector<bool>> tuple_slice_derived_;
};

}  // namespace

void VerifyOrchestrationCodegenPreconditions(const ProgramPtr& program, const FunctionPtr& func) {
  INTERNAL_CHECK(program != nullptr)
      << "Internal error: GenerateOrchestration preconditions — program must not be null";
  INTERNAL_CHECK(func != nullptr)
      << "Internal error: GenerateOrchestration preconditions — function must not be null";

  // Codegen assumes hierarchy references resolved, explicit RuntimeScopeStmt
  // materialization, and a stamped iter_arg carry plan on every ForStmt.
  //
  // ReturnParamsExplicit is listed because codegen *reads the IR directly* for
  // the return->param map: it takes each callee's returned param off the
  // ReturnStmt by pointer identity (NormalizeReturnOrder establishes that form)
  // instead of tracing SSA lineage. Without it, an SSA-aliased return would
  // silently alias a result to the wrong buffer rather than raise.
  //
  // TODO(call-directions-precondition): CallDirectionsResolved belongs here
  // too — codegen equally trusts `callee->param_directions_`, which
  // DeriveCallDirections materializes on Group/Spmd wrappers. It is not listed
  // yet because two existing tests feed orchestration codegen IR that
  // deliberately violates it (an `Input` arg direction on an `Out` param, and a
  // convert_to_ssa-only program), so wiring it needs those tests reworked.
  pass::VerifyProperties(
      IRPropertySet{IRProperty::SplitIncoreOrch, IRProperty::OrchestrationReferencesResolved,
                    IRProperty::RuntimeScopesMaterialized, IRProperty::IterArgCarryClassified,
                    IRProperty::ReturnParamsExplicit},
      program, "GenerateOrchestration preconditions");

  ReturnParamMap return_params;
  for (const auto& [global, candidate] : program->functions_) {
    return_params.emplace(global->name_, return_lineage::ExplicitReturnedParamIndices(candidate));
  }

  MxParamMap mx_params;
  for (const auto& [global, candidate] : program->functions_) {
    MxLoadParamCollector collector(candidate, return_params);
    collector.VisitFunction(candidate);
    if (!collector.param_indices.empty()) {
      mx_params.emplace(global->name_, std::move(collector.param_indices));
    }
  }
  if (!mx_params.empty()) {
    MxSliceCallVerifier verifier(mx_params, return_params);
    verifier.VisitFunction(func);
  }
}

void VerifyDistributedCodegenPreconditions(const ProgramPtr& program) {
  INTERNAL_CHECK(program != nullptr)
      << "Internal error: DistributedCodegen preconditions — program must not be null";

  DistributedTensorUseCollector collector;
  collector.VisitProgram(program);
  if (!collector.uses_distributed_tensor) {
    return;
  }

  INTERNAL_CHECK(!collector.has_windowing_ops || collector.has_comm_domain_scope)
      << "Internal error: DistributedCodegen preconditions — MaterializeCommDomainScopes must run before "
         "DistributedCodegen when window-buffer/distributed-tensor ops are present. "
         "The pass pipeline is incomplete.";

  // Comm-domain materialization is required when DistributedTensor values are
  // present in host orchestration paths.
  pass::VerifyProperties(IRPropertySet{IRProperty::CommDomainScopesMaterialized}, program,
                         "DistributedCodegen preconditions");
}

}  // namespace codegen
}  // namespace pypto
