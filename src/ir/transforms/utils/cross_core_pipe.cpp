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

#include "pypto/ir/transforms/utils/cross_core_pipe.h"

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "pypto/ir/transforms/utils/core_side_ops.h"
#include "pypto/ir/transforms/utils/loop_state_repair.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace cross_core_pipe {

namespace {

const auto& FlattenBody = transform_utils::FlattenToStmts;
using loop_repair::MakeBody;

struct TransferEndpoint {
  CallPtr call;
  TypePtr type;
  VarPtr pop_var;
};

struct DirectionTransferEndpoints {
  std::vector<TransferEndpoint> pushes;
  std::vector<TransferEndpoint> pops;
};

struct CrossCoreTransferEndpoints {
  DirectionTransferEndpoints c2v;
  DirectionTransferEndpoints v2c;
  std::unordered_map<const Var*, VarPtr> alias_roots;
};

struct PipeIdAssignments {
  std::unordered_map<const Call*, int> call_ids;
  std::unordered_map<const Var*, int> popped_var_ids;

  [[nodiscard]] bool Empty() const { return call_ids.empty(); }
};

struct DirectionPipePlan {
  PipeDirectionMetadata metadata;
  PipeIdAssignments assignments;
  std::optional<int> single_pipe_id;
};

CallPtr CallWithPipeId(const CallPtr& call, int pipe_id) {
  if (call->HasKwarg("id")) {
    CHECK_SPAN(call->GetKwarg<int>("id", -1) == pipe_id, call->span_)
        << "Automatic cross-core pipe setup assigned frontend pipe id " << pipe_id
        << " to a paired transfer operation, but the operation already carries id="
        << call->GetKwarg<int>("id", -1)
        << "; remove the explicit id or provide a complete manual pipe setup";
    return call;
  }
  auto kwargs = call->kwargs_;
  kwargs.emplace_back("id", std::any(pipe_id));
  return std::make_shared<Call>(call->op_, call->args_, std::move(kwargs), call->attrs_, call->GetType(),
                                call->span_);
}

bool IsMxScaleTransferType(const TypePtr& type) {
  auto tile_type = As<TileType>(type);
  if (!tile_type || tile_type->dtype_ != DataType::FP8E8M0) return false;
  const TileView view = tile_view_semantics::GetEffectiveTileView(*tile_type);
  return view.fractal == tile_view_semantics::kMXScaleFractal && view.blayout == view.slayout &&
         (view.blayout == TileLayout::row_major || view.blayout == TileLayout::col_major);
}

VarPtr ResolveAliasRoot(const VarPtr& var, const std::unordered_map<const Var*, VarPtr>& alias_roots) {
  VarPtr current = var;
  std::unordered_set<const Var*> seen;
  while (current && seen.insert(current.get()).second) {
    auto it = alias_roots.find(current.get());
    if (it == alias_roots.end() || !it->second) break;
    current = it->second;
  }
  return current;
}

VarPtr ResolveAliasedExpr(const ExprPtr& expr, const std::unordered_map<const Var*, VarPtr>& alias_roots) {
  ExprPtr source = expr;
  while (source) {
    if (auto var = AsVarLike(source)) return ResolveAliasRoot(var, alias_roots);
    auto call = As<Call>(source);
    if (!call || !call->op_ || call->args_.empty() ||
        !op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
      return nullptr;
    }
    source = call->args_[0];
  }
  return nullptr;
}

void RecordAlias(const VarPtr& alias, const ExprPtr& source, CrossCoreTransferEndpoints& endpoints) {
  auto root = ResolveAliasedExpr(source, endpoints.alias_roots);
  if (alias && root && alias.get() != root.get()) endpoints.alias_roots[alias.get()] = root;
}

void CollectTransferEndpoints(const std::vector<StmtPtr>& stmts, CrossCoreTransferEndpoints& endpoints) {
  for (const auto& stmt : stmts) {
    auto assign = As<AssignStmt>(stmt);
    auto eval = As<EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = As<Call>(assign->value_);
    } else if (eval) {
      call = As<Call>(eval->expr_);
    }

    if (call) {
      if (IsOp(call, "tile.tpush_to_aiv") && call->args_.size() == 1) {
        endpoints.c2v.pushes.push_back({call, call->args_[0]->GetType(), nullptr});
      } else if (IsOp(call, "tile.tpop_from_aic") && assign) {
        endpoints.c2v.pops.push_back({call, assign->var_->GetType(), assign->var_});
      } else if (IsOp(call, "tile.tpush_to_aic") && call->args_.size() == 1) {
        endpoints.v2c.pushes.push_back({call, call->args_[0]->GetType(), nullptr});
      } else if (IsOp(call, "tile.tpop_from_aiv") && assign) {
        endpoints.v2c.pops.push_back({call, assign->var_->GetType(), assign->var_});
      }
    }

    if (assign) RecordAlias(assign->var_, assign->value_, endpoints);

    if (auto for_stmt = As<ForStmt>(stmt)) {
      for (const auto& iter_arg : for_stmt->iter_args_) {
        RecordAlias(iter_arg, iter_arg ? iter_arg->initValue_ : nullptr, endpoints);
      }
      CollectTransferEndpoints(FlattenBody(for_stmt->body_), endpoints);
    } else if (auto if_stmt = As<IfStmt>(stmt)) {
      CollectTransferEndpoints(FlattenBody(if_stmt->then_body_), endpoints);
      if (if_stmt->else_body_.has_value()) {
        CollectTransferEndpoints(FlattenBody(*if_stmt->else_body_), endpoints);
      }
    } else if (auto while_stmt = As<WhileStmt>(stmt)) {
      for (const auto& iter_arg : while_stmt->iter_args_) {
        RecordAlias(iter_arg, iter_arg ? iter_arg->initValue_ : nullptr, endpoints);
      }
      CollectTransferEndpoints(FlattenBody(while_stmt->body_), endpoints);
    }
  }
}

void AssignPipeIdToEndpoints(const DirectionTransferEndpoints& endpoints, int pipe_id,
                             PipeIdAssignments& assignments) {
  for (const auto& push : endpoints.pushes) assignments.call_ids[push.call.get()] = pipe_id;
  for (const auto& pop : endpoints.pops) {
    assignments.call_ids[pop.call.get()] = pipe_id;
    assignments.popped_var_ids[pop.pop_var.get()] = pipe_id;
  }
}

DirectionPipePlan BuildDirectionPipePlan(const DirectionTransferEndpoints& endpoints,
                                         const std::string& direction_name, const Span& span) {
  DirectionPipePlan plan;
  if (endpoints.pushes.empty() && endpoints.pops.empty()) return plan;

  CHECK_SPAN(endpoints.pushes.size() == endpoints.pops.size(), span)
      << "Automatic " << direction_name
      << " pipe setup requires every tpush to have one paired tpop, but got " << endpoints.pushes.size()
      << " tpush operation(s) and " << endpoints.pops.size()
      << " tpop operation(s); provide a complete manual pipe setup for an unpaired transfer graph";

  std::vector<bool> pair_is_scale;
  pair_is_scale.reserve(endpoints.pushes.size());
  for (size_t i = 0; i < endpoints.pushes.size(); ++i) {
    const auto& push = endpoints.pushes[i];
    const auto& pop = endpoints.pops[i];
    auto push_size = TryGetTileSlotSizeBytes(push.type);
    auto pop_size = TryGetTileSlotSizeBytes(pop.type);
    CHECK_SPAN(push_size.has_value() && pop_size.has_value(), push.call->span_)
        << "Automatic " << direction_name << " pipe setup requires static tile sizes for paired transfer "
        << i;

    const bool push_is_scale = IsMxScaleTransferType(push.type);
    const bool pop_is_scale = IsMxScaleTransferType(pop.type);
    CHECK_SPAN(push_is_scale == pop_is_scale, pop.call->span_)
        << "Automatic " << direction_name << " pipe setup paired transfer " << i
        << " as different payload kinds on its tpush and tpop endpoints; MX scale transfers must pair with "
           "MX scale transfers in the same lexical order";

    pair_is_scale.push_back(push_is_scale);
    RecordObservedSlotSize(plan.metadata, std::max(push_size.value(), pop_size.value()));
  }

  if (plan.metadata.observed_slot_sizes.size() <= 1) {
    // A homogeneous direction uses one frontend pipe. Honor a common explicit
    // endpoint id and stamp it onto any paired endpoint/tfree that omitted it.
    // Multiple explicit ids cannot share the single automatically allocated
    // buffer, even when their slot sizes happen to match.
    for (size_t i = 0; i < endpoints.pushes.size(); ++i) {
      for (const auto& endpoint : {endpoints.pushes[i].call, endpoints.pops[i].call}) {
        if (!endpoint->HasKwarg("id")) continue;
        const int explicit_id = endpoint->GetKwarg<int>("id", -1);
        CHECK_SPAN(!plan.single_pipe_id.has_value() || plan.single_pipe_id.value() == explicit_id,
                   endpoint->span_)
            << "Automatic " << direction_name
            << " pipe setup found multiple explicit frontend pipe ids for one homogeneous pipe ("
            << plan.single_pipe_id.value_or(explicit_id) << " and " << explicit_id
            << "); use one common id or provide a complete manual pipe setup";
        plan.single_pipe_id = explicit_id;
      }
    }
    if (plan.single_pipe_id.has_value()) {
      AssignPipeIdToEndpoints(endpoints, plan.single_pipe_id.value(), plan.assignments);
    }
    return plan;
  }

  CHECK_SPAN(endpoints.pushes.size() == 2, span)
      << "Automatic " << direction_name
      << " setup currently supports heterogeneous slot sizes only for exactly two paired transfers "
         "(data first, MX scale second), but got "
      << endpoints.pushes.size() << " pairs; use a complete manual pipe setup for other transfer graphs";
  CHECK_SPAN(!pair_is_scale[0] && pair_is_scale[1], endpoints.pushes[1].call->span_)
      << "Automatic " << direction_name
      << " setup currently requires heterogeneous paired transfers to be ordered as ordinary data first "
         "and FP8E8M0 MX scale second (fractal=32)";

  for (size_t i = 0; i < endpoints.pushes.size(); ++i) {
    const int pipe_id = static_cast<int>(i);
    plan.assignments.call_ids.emplace(endpoints.pushes[i].call.get(), pipe_id);
    plan.assignments.call_ids.emplace(endpoints.pops[i].call.get(), pipe_id);
    plan.assignments.popped_var_ids.emplace(endpoints.pops[i].pop_var.get(), pipe_id);
  }
  return plan;
}

std::vector<StmtPtr> StampPipeIdsOnTransfers(const std::vector<StmtPtr>& stmts,
                                             const PipeIdAssignments& v2c_assignments,
                                             const PipeIdAssignments& c2v_assignments,
                                             const std::unordered_map<const Var*, VarPtr>& alias_roots);

StmtPtr StampPipeIdsOnStmt(const StmtPtr& stmt, const PipeIdAssignments& v2c_assignments,
                           const PipeIdAssignments& c2v_assignments,
                           const std::unordered_map<const Var*, VarPtr>& alias_roots) {
  if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
    auto new_for = MutableCopy(for_stmt);
    new_for->body_ = MakeBody(
        StampPipeIdsOnTransfers(FlattenBody(for_stmt->body_), v2c_assignments, c2v_assignments, alias_roots),
        for_stmt->span_);
    return new_for;
  }
  if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
    auto new_if = MutableCopy(if_stmt);
    new_if->then_body_ = MakeBody(StampPipeIdsOnTransfers(FlattenBody(if_stmt->then_body_), v2c_assignments,
                                                          c2v_assignments, alias_roots),
                                  if_stmt->span_);
    if (if_stmt->else_body_.has_value()) {
      new_if->else_body_ = MakeBody(StampPipeIdsOnTransfers(FlattenBody(*if_stmt->else_body_),
                                                            v2c_assignments, c2v_assignments, alias_roots),
                                    if_stmt->span_);
    }
    return new_if;
  }
  if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
    auto new_while = MutableCopy(while_stmt);
    new_while->body_ = MakeBody(StampPipeIdsOnTransfers(FlattenBody(while_stmt->body_), v2c_assignments,
                                                        c2v_assignments, alias_roots),
                                while_stmt->span_);
    return new_while;
  }

  auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
  auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
  CallPtr call;
  if (assign) {
    call = std::dynamic_pointer_cast<const Call>(assign->value_);
  } else if (eval) {
    call = std::dynamic_pointer_cast<const Call>(eval->expr_);
  }
  if (!call) return stmt;

  VarPtr canonical_tfree_var;

  auto lookup_call_id = [&](const PipeIdAssignments& assignments) -> std::optional<int> {
    if (assignments.Empty()) return std::nullopt;
    auto it = assignments.call_ids.find(call.get());
    INTERNAL_CHECK_SPAN(it != assignments.call_ids.end(), call->span_)
        << "Internal error: paired cross-core transfer has no assigned pipe id";
    return it->second;
  };
  auto lookup_tfree_id = [&](const PipeIdAssignments& assignments) -> std::optional<int> {
    if (assignments.Empty()) return std::nullopt;
    auto freed_var = AsVarLike(call->args_[0]);
    INTERNAL_CHECK_SPAN(freed_var, call->span_)
        << "Internal error: automatic multi-pipe tfree operand must be a Var or IterArg";
    auto it = assignments.popped_var_ids.find(freed_var.get());
    if (it == assignments.popped_var_ids.end()) {
      auto canonical = ResolveAliasRoot(freed_var, alias_roots);
      it = assignments.popped_var_ids.find(canonical.get());
      if (it != assignments.popped_var_ids.end()) canonical_tfree_var = canonical;
    }
    INTERNAL_CHECK_SPAN(it != assignments.popped_var_ids.end(), call->span_)
        << "Internal error: automatic multi-pipe tfree does not reference a paired tpop result";
    return it->second;
  };

  std::optional<int> pipe_id;
  if (op_predicates::IsTPush(call) && call->args_.size() == 1) {
    auto op = std::dynamic_pointer_cast<const Op>(call->op_);
    if (op && IsOp(op, "tile.tpush_to_aic")) {
      pipe_id = lookup_call_id(v2c_assignments);
    } else if (op && IsOp(op, "tile.tpush_to_aiv")) {
      pipe_id = lookup_call_id(c2v_assignments);
    }
  } else if (op_predicates::IsTPop(call) && assign) {
    auto op = std::dynamic_pointer_cast<const Op>(call->op_);
    if (op && IsOp(op, "tile.tpop_from_aiv")) {
      pipe_id = lookup_call_id(v2c_assignments);
    } else if (op && IsOp(op, "tile.tpop_from_aic")) {
      pipe_id = lookup_call_id(c2v_assignments);
    }
  } else if (op_predicates::IsTFree(call) && call->args_.size() == 1) {
    auto op = std::dynamic_pointer_cast<const Op>(call->op_);
    if (op && IsOp(op, "system.tfree_to_aiv")) {
      pipe_id = lookup_tfree_id(v2c_assignments);
    } else if (op && IsOp(op, "system.tfree_to_aic")) {
      pipe_id = lookup_tfree_id(c2v_assignments);
    }
  }

  if (!pipe_id.has_value()) return stmt;
  if (canonical_tfree_var && canonical_tfree_var.get() != AsVarLike(call->args_[0]).get()) {
    auto canonical_call = MutableCopy(call);
    canonical_call->args_[0] = canonical_tfree_var;
    call = canonical_call;
  }
  auto stamped = CallWithPipeId(call, pipe_id.value());
  if (stamped == call) return stmt;
  if (assign) {
    return std::make_shared<AssignStmt>(assign->var_, stamped, assign->span_);
  }
  return std::make_shared<EvalStmt>(stamped, eval->span_);
}

std::vector<StmtPtr> StampPipeIdsOnTransfers(const std::vector<StmtPtr>& stmts,
                                             const PipeIdAssignments& v2c_assignments,
                                             const PipeIdAssignments& c2v_assignments,
                                             const std::unordered_map<const Var*, VarPtr>& alias_roots) {
  if (v2c_assignments.Empty() && c2v_assignments.Empty()) return stmts;
  std::vector<StmtPtr> result;
  result.reserve(stmts.size());
  for (const auto& stmt : stmts) {
    result.push_back(StampPipeIdsOnStmt(stmt, v2c_assignments, c2v_assignments, alias_roots));
  }
  return result;
}

}  // namespace

std::optional<int64_t> TryGetConstIntValue(const ExprPtr& expr) {
  auto const_int = std::dynamic_pointer_cast<const ConstInt>(expr);
  if (!const_int || const_int->value_ < 0) return std::nullopt;
  return const_int->value_;
}

std::optional<int64_t> TryGetTileSlotSizeBytes(const TypePtr& type) {
  auto tile_type = std::dynamic_pointer_cast<const TileType>(type);
  if (!tile_type) return std::nullopt;

  int64_t element_count = 1;
  for (const auto& dim : tile_type->shape_) {
    auto dim_value = TryGetConstIntValue(dim);
    if (!dim_value.has_value()) return std::nullopt;
    INTERNAL_CHECK(*dim_value == 0 || element_count <= std::numeric_limits<int64_t>::max() / *dim_value)
        << "Tile element count overflow while inferring cross-core slot size";
    element_count *= *dim_value;
  }

  const int64_t bit_width = static_cast<int64_t>(tile_type->dtype_.GetBit());
  INTERNAL_CHECK(bit_width > 0) << "Unsupported dtype for cross-core slot size inference: "
                                << tile_type->dtype_.ToString();
  INTERNAL_CHECK(element_count <= (std::numeric_limits<int64_t>::max() - 7) / bit_width)
      << "Tile byte size overflow while inferring cross-core slot size";
  return (element_count * bit_width + 7) / 8;
}

void RecordObservedSlotSize(PipeDirectionMetadata& metadata, int64_t slot_size) {
  metadata.has_ops = true;
  if (std::find(metadata.observed_slot_sizes.begin(), metadata.observed_slot_sizes.end(), slot_size) ==
      metadata.observed_slot_sizes.end()) {
    metadata.observed_slot_sizes.push_back(slot_size);
  }
  if (!metadata.slot_size_bytes.has_value()) {
    metadata.slot_size_bytes = slot_size;
    return;
  }
  if (metadata.slot_size_bytes.value() != slot_size) {
    metadata.has_inconsistent_slot_size = true;
    metadata.slot_size_bytes = std::max(metadata.slot_size_bytes.value(), slot_size);
  }
}

void RecordTileSlotSize(PipeDirectionMetadata& metadata, const TypePtr& type) {
  metadata.has_ops = true;
  auto slot_size = TryGetTileSlotSizeBytes(type);
  if (slot_size.has_value()) {
    RecordObservedSlotSize(metadata, slot_size.value());
  }
}

void MergeDirectionMetadata(PipeDirectionMetadata& dst, const PipeDirectionMetadata& src) {
  dst.has_ops = dst.has_ops || src.has_ops;
  dst.has_inconsistent_slot_size = dst.has_inconsistent_slot_size || src.has_inconsistent_slot_size;
  for (int64_t slot_size : src.observed_slot_sizes) {
    RecordObservedSlotSize(dst, slot_size);
  }
}

CrossCorePipeMetadata MergeCrossCorePipeMetadata(const CrossCorePipeMetadata& lhs,
                                                 const CrossCorePipeMetadata& rhs) {
  CrossCorePipeMetadata merged;
  MergeDirectionMetadata(merged.c2v, lhs.c2v);
  MergeDirectionMetadata(merged.c2v, rhs.c2v);
  MergeDirectionMetadata(merged.v2c, lhs.v2c);
  MergeDirectionMetadata(merged.v2c, rhs.v2c);
  merged.has_reserve_buffer = lhs.has_reserve_buffer || rhs.has_reserve_buffer;
  merged.has_import_peer_buffer = lhs.has_import_peer_buffer || rhs.has_import_peer_buffer;
  merged.has_aic_initialize_pipe = lhs.has_aic_initialize_pipe || rhs.has_aic_initialize_pipe;
  merged.has_aiv_initialize_pipe = lhs.has_aiv_initialize_pipe || rhs.has_aiv_initialize_pipe;
  return merged;
}

int BuildDirMask(const CrossCorePipeMetadata& metadata) {
  int dir_mask = 0;
  if (metadata.c2v.has_ops) dir_mask |= core_affinity::kDirMaskC2V;
  if (metadata.v2c.has_ops) dir_mask |= core_affinity::kDirMaskV2C;
  return dir_mask;
}

int GetSlotNumForDirMask(int dir_mask) {
  return dir_mask == (core_affinity::kDirMaskC2V | core_affinity::kDirMaskV2C) ? 4 : 8;
}

std::optional<int64_t> GetCommonSlotSizeBytes(const CrossCorePipeMetadata& metadata) {
  std::optional<int64_t> common_slot_size;
  for (const auto* direction : {&metadata.c2v, &metadata.v2c}) {
    if (!direction->has_ops) continue;
    if (!direction->slot_size_bytes.has_value()) {
      return std::nullopt;
    }
    if (!common_slot_size.has_value()) {
      common_slot_size = direction->slot_size_bytes;
      continue;
    }
    common_slot_size = std::max(common_slot_size.value(), direction->slot_size_bytes.value());
  }
  return common_slot_size;
}

std::string BuildPipeBufferName(const std::string& func_name, core_affinity::PipeDirection direction,
                                int pipe_id) {
  std::string name = func_name + ((direction == core_affinity::PipeDirection::C2V) ? "_c2v_slot_buffer"
                                                                                   : "_v2c_slot_buffer");
  if (pipe_id != 0) {
    name += "_id" + std::to_string(pipe_id);
  }
  return name;
}

CallPtr CreateSystemOpCall(const std::string& op_name,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return CreateSystemOpCall(op_name, {}, kwargs, span);
}

CallPtr CreateSystemOpCall(const std::string& op_name, const std::vector<ExprPtr>& args,
                           const std::vector<std::pair<std::string, std::any>>& kwargs, const Span& span) {
  return OpRegistry::GetInstance().Create(op_name, args, kwargs, span);
}

CallPtr CreateReserveBuffer(const std::string& buffer_name, int64_t size_bytes, const Span& span) {
  INTERNAL_CHECK_SPAN(size_bytes >= 0 && size_bytes <= std::numeric_limits<int>::max(), span)
      << "Cross-core reserve_buffer size out of range: " << size_bytes;
  return CreateSystemOpCall("system.reserve_buffer",
                            {{"name", std::any(buffer_name)},
                             {"size", std::any(static_cast<int>(size_bytes))},
                             {"base", std::any(kAutoBufferBase)}},
                            span);
}

CallPtr CreateImportPeerBuffer(const std::string& buffer_name, const std::string& peer_func,
                               const Span& span) {
  return CreateSystemOpCall("system.import_peer_buffer",
                            {{"name", std::any(buffer_name)}, {"peer_func", std::any(peer_func)}}, span);
}

CallPtr CreateInitializePipe(core_affinity::CoreSide side, int dir_mask, int slot_size_bytes,
                             const ExprPtr& c2v_consumer_buf, const ExprPtr& v2c_consumer_buf,
                             std::optional<int> slot_num, const Span& span, std::optional<int> pipe_id) {
  INTERNAL_CHECK_SPAN(slot_size_bytes >= 0 && slot_size_bytes <= std::numeric_limits<int>::max(), span)
      << "Cross-core slot_size out of range: " << slot_size_bytes;
  std::vector<std::pair<std::string, std::any>> kwargs = {{"dir_mask", std::any(dir_mask)},
                                                          {"slot_size", std::any(slot_size_bytes)}};
  if (slot_num.has_value()) {
    INTERNAL_CHECK_SPAN(slot_num.value() > 0, span)
        << "Cross-core slot_num override must be positive: " << slot_num.value();
    kwargs.emplace_back("slot_num", std::any(slot_num.value()));
  }
  if (pipe_id.has_value()) {
    kwargs.emplace_back("id", std::any(pipe_id.value()));
  }
  const std::string op_name = core_side_ops::InitializePipeOp(side);
  return CreateSystemOpCall(op_name, {c2v_consumer_buf, v2c_consumer_buf}, kwargs, span);
}

void CollectCrossCorePipeMetadata(const std::vector<StmtPtr>& stmts, CrossCorePipeMetadata& metadata) {
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (eval) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    if (op) {
      if (IsOp(op, "system.reserve_buffer")) {
        metadata.has_reserve_buffer = true;
      } else if (IsOp(op, "system.import_peer_buffer")) {
        metadata.has_import_peer_buffer = true;
      } else if (IsOp(op, "system.aic_initialize_pipe")) {
        metadata.has_aic_initialize_pipe = true;
      } else if (IsOp(op, "system.aiv_initialize_pipe")) {
        metadata.has_aiv_initialize_pipe = true;
      } else if (IsOp(op, "tile.tpush_to_aiv") && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.c2v, call->args_[0]->GetType());
      } else if (IsOp(op, "tile.tpush_to_aic") && call->args_.size() == 1) {
        RecordTileSlotSize(metadata.v2c, call->args_[0]->GetType());
      } else if (IsOp(op, "tile.tpop_from_aiv") && assign) {
        RecordTileSlotSize(metadata.v2c, assign->var_->GetType());
      } else if (IsOp(op, "tile.tpop_from_aic") && assign) {
        RecordTileSlotSize(metadata.c2v, assign->var_->GetType());
      }
    }

    if (auto for_stmt = std::dynamic_pointer_cast<const ForStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(for_stmt->body_), metadata);
    } else if (auto if_stmt = std::dynamic_pointer_cast<const IfStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(if_stmt->then_body_), metadata);
      const auto& else_body = if_stmt->else_body_;
      if (else_body) {
        CollectCrossCorePipeMetadata(FlattenBody(*else_body), metadata);
      }
    } else if (auto while_stmt = std::dynamic_pointer_cast<const WhileStmt>(stmt)) {
      CollectCrossCorePipeMetadata(FlattenBody(while_stmt->body_), metadata);
    }
  }
}

CrossCorePipeMetadata CollectDominatingPipeSetupMetadata(const std::vector<StmtPtr>& stmts) {
  CrossCorePipeMetadata metadata;
  for (const auto& stmt : stmts) {
    auto assign = std::dynamic_pointer_cast<const AssignStmt>(stmt);
    auto eval = std::dynamic_pointer_cast<const EvalStmt>(stmt);
    CallPtr call;
    if (assign) {
      call = std::dynamic_pointer_cast<const Call>(assign->value_);
    } else if (eval) {
      call = std::dynamic_pointer_cast<const Call>(eval->expr_);
    }
    auto op = call ? std::dynamic_pointer_cast<const Op>(call->op_) : nullptr;
    CrossCorePipeMetadata stmt_metadata;
    CollectCrossCorePipeMetadata({stmt}, stmt_metadata);
    if (stmt_metadata.HasCrossCoreOps()) {
      break;
    }
    if (op) {
      if (IsOp(op, "system.reserve_buffer")) {
        metadata.has_reserve_buffer = true;
      } else if (IsOp(op, "system.import_peer_buffer")) {
        metadata.has_import_peer_buffer = true;
      } else if (IsOp(op, "system.aic_initialize_pipe")) {
        metadata.has_aic_initialize_pipe = true;
      } else if (IsOp(op, "system.aiv_initialize_pipe")) {
        metadata.has_aiv_initialize_pipe = true;
      }
    }
  }
  return metadata;
}

AutomaticPipeSetup BuildAutomaticPipeSetup(const std::string& func_name, const std::string& aic_name,
                                           const std::string& aiv_name, const std::vector<StmtPtr>& aic_stmts,
                                           const std::vector<StmtPtr>& aiv_stmts,
                                           std::optional<int> slot_num_override, const Span& span) {
  CrossCorePipeMetadata aic_metadata;
  CollectCrossCorePipeMetadata(aic_stmts, aic_metadata);
  CrossCorePipeMetadata aiv_metadata;
  CollectCrossCorePipeMetadata(aiv_stmts, aiv_metadata);
  CrossCorePipeMetadata combined = MergeCrossCorePipeMetadata(aic_metadata, aiv_metadata);

  if (!combined.HasCrossCoreOps() || aic_metadata.HasAnySetup() || aiv_metadata.HasAnySetup()) {
    return {};
  }

  if (slot_num_override.has_value()) {
    INTERNAL_CHECK_SPAN(slot_num_override.value() > 0, span)
        << "Cross-core slot_num override must be positive: " << slot_num_override.value();
  }

  auto zero_i32 = [&]() { return std::make_shared<ConstInt>(0, DataType::INT32, span); };
  auto var_as_expr = [](const std::shared_ptr<Var>& v) -> ExprPtr {
    return std::static_pointer_cast<const Expr>(v);
  };

  CrossCoreTransferEndpoints endpoints;
  CollectTransferEndpoints(aic_stmts, endpoints);
  CollectTransferEndpoints(aiv_stmts, endpoints);
  auto c2v_plan = BuildDirectionPipePlan(endpoints.c2v, "C2V", span);
  auto v2c_plan = BuildDirectionPipePlan(endpoints.v2c, "V2C", span);
  CrossCorePipeMetadata paired_metadata;
  paired_metadata.c2v = c2v_plan.metadata;
  paired_metadata.v2c = v2c_plan.metadata;

  AutomaticPipeSetup setup;
  const bool needs_multi_id = paired_metadata.c2v.observed_slot_sizes.size() > 1 ||
                              paired_metadata.v2c.observed_slot_sizes.size() > 1;

  if (needs_multi_id) {
    // One unidirectional pipe per paired payload, with distinct frontend ids.
    // Stamp matching ids onto tpush/tpop/tfree so PTOAS routes both endpoints
    // of a transfer onto the same pipe even when a split changes its tile size.
    CHECK_SPAN(!paired_metadata.c2v.has_ops || !paired_metadata.c2v.observed_slot_sizes.empty(), span)
        << "Automatic multi-id C2V pipe setup requires concrete tile slot sizes";
    CHECK_SPAN(!paired_metadata.v2c.has_ops || !paired_metadata.v2c.observed_slot_sizes.empty(), span)
        << "Automatic multi-id V2C pipe setup requires concrete tile slot sizes";

    setup.aic_body =
        StampPipeIdsOnTransfers(aic_stmts, v2c_plan.assignments, c2v_plan.assignments, endpoints.alias_roots);
    setup.aiv_body =
        StampPipeIdsOnTransfers(aiv_stmts, v2c_plan.assignments, c2v_plan.assignments, endpoints.alias_roots);

    auto emit_unidirectional_pipes = [&](core_affinity::PipeDirection direction,
                                         const DirectionPipePlan& direction_plan, int dir_mask) {
      const auto& direction_meta = direction_plan.metadata;
      if (!direction_meta.has_ops) return;
      const int effective_slot_num = slot_num_override.value_or(GetSlotNumForDirMask(dir_mask));
      for (size_t i = 0; i < direction_meta.observed_slot_sizes.size(); ++i) {
        const int pipe_id = static_cast<int>(i);
        const int64_t slot_size_i64 = direction_meta.observed_slot_sizes[i];
        INTERNAL_CHECK_SPAN(slot_size_i64 >= 0 && slot_size_i64 <= std::numeric_limits<int>::max(), span)
            << "Cross-core slot_size out of range: " << slot_size_i64;
        const int slot_size_bytes = static_cast<int>(slot_size_i64);
        const int64_t buffer_size = slot_size_i64 * effective_slot_num;
        const std::optional<int> explicit_id = direction_meta.observed_slot_sizes.size() > 1
                                                   ? std::optional<int>(pipe_id)
                                                   : direction_plan.single_pipe_id;
        const auto buffer_name = BuildPipeBufferName(func_name, direction, explicit_id.value_or(pipe_id));

        if (direction == core_affinity::PipeDirection::V2C) {
          auto reserve = CreateReserveBuffer(buffer_name, buffer_size, span);
          auto aic_reserve_var = std::make_shared<Var>(buffer_name, reserve->GetType(), span);
          setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_reserve_var, reserve, span));
          auto import = CreateImportPeerBuffer(buffer_name, aic_name, span);
          auto aiv_import_var = std::make_shared<Var>(buffer_name + "_import", import->GetType(), span);
          setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_import_var, import, span));
          setup.aic_stmts.push_back(std::make_shared<EvalStmt>(
              CreateInitializePipe(core_affinity::CoreSide::AIC, dir_mask, slot_size_bytes, zero_i32(),
                                   var_as_expr(aic_reserve_var), slot_num_override, span, explicit_id),
              span));
          setup.aiv_stmts.push_back(std::make_shared<EvalStmt>(
              CreateInitializePipe(core_affinity::CoreSide::AIV, dir_mask, slot_size_bytes, zero_i32(),
                                   var_as_expr(aiv_import_var), slot_num_override, span, explicit_id),
              span));
        } else {
          auto reserve = CreateReserveBuffer(buffer_name, buffer_size, span);
          auto aiv_reserve_var = std::make_shared<Var>(buffer_name, reserve->GetType(), span);
          setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_reserve_var, reserve, span));
          auto import = CreateImportPeerBuffer(buffer_name, aiv_name, span);
          auto aic_import_var = std::make_shared<Var>(buffer_name + "_import", import->GetType(), span);
          setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_import_var, import, span));
          setup.aic_stmts.push_back(std::make_shared<EvalStmt>(
              CreateInitializePipe(core_affinity::CoreSide::AIC, dir_mask, slot_size_bytes,
                                   var_as_expr(aic_import_var), zero_i32(), slot_num_override, span,
                                   explicit_id),
              span));
          setup.aiv_stmts.push_back(std::make_shared<EvalStmt>(
              CreateInitializePipe(core_affinity::CoreSide::AIV, dir_mask, slot_size_bytes,
                                   var_as_expr(aiv_reserve_var), zero_i32(), slot_num_override, span,
                                   explicit_id),
              span));
        }
      }
    };

    emit_unidirectional_pipes(core_affinity::PipeDirection::V2C, v2c_plan, core_affinity::kDirMaskV2C);
    emit_unidirectional_pipes(core_affinity::PipeDirection::C2V, c2v_plan, core_affinity::kDirMaskC2V);
    return setup;
  }

  // Single-size-per-direction path: one pipe (uni or bidirectional) with the
  // common max slot size. A common explicit endpoint id is inherited by the
  // initializer and by all endpoints sharing a bidirectional pipe.
  std::optional<int> explicit_pipe_id = c2v_plan.single_pipe_id;
  if (v2c_plan.single_pipe_id.has_value()) {
    CHECK_SPAN(!explicit_pipe_id.has_value() || explicit_pipe_id.value() == v2c_plan.single_pipe_id.value(),
               span)
        << "Automatic bidirectional pipe setup found different explicit frontend pipe ids for C2V and V2C ("
        << explicit_pipe_id.value_or(v2c_plan.single_pipe_id.value()) << " and "
        << v2c_plan.single_pipe_id.value() << "); use one common id or provide a complete manual pipe setup";
    explicit_pipe_id = v2c_plan.single_pipe_id;
  }
  if (explicit_pipe_id.has_value()) {
    AssignPipeIdToEndpoints(endpoints.c2v, explicit_pipe_id.value(), c2v_plan.assignments);
    AssignPipeIdToEndpoints(endpoints.v2c, explicit_pipe_id.value(), v2c_plan.assignments);
    setup.aic_body =
        StampPipeIdsOnTransfers(aic_stmts, v2c_plan.assignments, c2v_plan.assignments, endpoints.alias_roots);
    setup.aiv_body =
        StampPipeIdsOnTransfers(aiv_stmts, v2c_plan.assignments, c2v_plan.assignments, endpoints.alias_roots);
  }
  const int dir_mask = BuildDirMask(paired_metadata);
  auto common_slot_size = GetCommonSlotSizeBytes(paired_metadata);
  if (dir_mask == 0 || !common_slot_size.has_value()) {
    return {};
  }

  const int effective_slot_num = slot_num_override.value_or(GetSlotNumForDirMask(dir_mask));
  const int64_t slot_size_i64 = common_slot_size.value();
  INTERNAL_CHECK_SPAN(slot_size_i64 >= 0 && slot_size_i64 <= std::numeric_limits<int>::max(), span)
      << "Cross-core slot_size out of range: " << slot_size_i64;
  const int slot_size_bytes = static_cast<int>(slot_size_i64);
  const int64_t buffer_size = slot_size_i64 * effective_slot_num;

  std::shared_ptr<Var> aic_v2c_reserve_var;
  std::shared_ptr<Var> aic_c2v_import_var;
  std::shared_ptr<Var> aiv_c2v_reserve_var;
  std::shared_ptr<Var> aiv_v2c_import_var;

  if (dir_mask & core_affinity::kDirMaskV2C) {
    const auto v2c_name =
        BuildPipeBufferName(func_name, core_affinity::PipeDirection::V2C, explicit_pipe_id.value_or(0));
    auto v2c_reserve = CreateReserveBuffer(v2c_name, buffer_size, span);
    aic_v2c_reserve_var = std::make_shared<Var>(v2c_name, v2c_reserve->GetType(), span);
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_v2c_reserve_var, v2c_reserve, span));
    auto v2c_import = CreateImportPeerBuffer(v2c_name, aic_name, span);
    aiv_v2c_import_var = std::make_shared<Var>(v2c_name + "_import", v2c_import->GetType(), span);
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_v2c_import_var, v2c_import, span));
  }

  if (dir_mask & core_affinity::kDirMaskC2V) {
    const auto c2v_name =
        BuildPipeBufferName(func_name, core_affinity::PipeDirection::C2V, explicit_pipe_id.value_or(0));
    auto c2v_reserve = CreateReserveBuffer(c2v_name, buffer_size, span);
    aiv_c2v_reserve_var = std::make_shared<Var>(c2v_name, c2v_reserve->GetType(), span);
    setup.aiv_stmts.push_back(std::make_shared<AssignStmt>(aiv_c2v_reserve_var, c2v_reserve, span));
    auto c2v_import = CreateImportPeerBuffer(c2v_name, aiv_name, span);
    aic_c2v_import_var = std::make_shared<Var>(c2v_name + "_import", c2v_import->GetType(), span);
    setup.aic_stmts.push_back(std::make_shared<AssignStmt>(aic_c2v_import_var, c2v_import, span));
  }

  const ExprPtr aic_c2v_arg = aic_c2v_import_var ? var_as_expr(aic_c2v_import_var) : ExprPtr(zero_i32());
  const ExprPtr aic_v2c_arg = aic_v2c_reserve_var ? var_as_expr(aic_v2c_reserve_var) : ExprPtr(zero_i32());
  const ExprPtr aiv_c2v_arg = aiv_c2v_reserve_var ? var_as_expr(aiv_c2v_reserve_var) : ExprPtr(zero_i32());
  const ExprPtr aiv_v2c_arg = aiv_v2c_import_var ? var_as_expr(aiv_v2c_import_var) : ExprPtr(zero_i32());

  setup.aic_stmts.push_back(std::make_shared<EvalStmt>(
      CreateInitializePipe(core_affinity::CoreSide::AIC, dir_mask, slot_size_bytes, aic_c2v_arg, aic_v2c_arg,
                           slot_num_override, span, explicit_pipe_id),
      span));
  setup.aiv_stmts.push_back(std::make_shared<EvalStmt>(
      CreateInitializePipe(core_affinity::CoreSide::AIV, dir_mask, slot_size_bytes, aiv_c2v_arg, aiv_v2c_arg,
                           slot_num_override, span, explicit_pipe_id),
      span));

  return setup;
}

std::vector<StmtPtr> PrependPipeSetup(const std::vector<StmtPtr>& prologue,
                                      const std::vector<StmtPtr>& body) {
  if (prologue.empty()) return body;
  std::vector<StmtPtr> result;
  result.reserve(prologue.size() + body.size());
  result.insert(result.end(), prologue.begin(), prologue.end());
  result.insert(result.end(), body.begin(), body.end());
  return result;
}

std::string FormatObservedSlotSizes(const std::vector<int64_t>& slot_sizes) {
  std::string result;
  for (size_t i = 0; i < slot_sizes.size(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(slot_sizes[i]);
  }
  return result;
}

}  // namespace cross_core_pipe
}  // namespace ir
}  // namespace pypto
