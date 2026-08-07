# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExpandMxPackedQuant."""

from collections import Counter

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _run_default_through(program: ir.Program, last_pass: str) -> ir.Program:
    manager = PassManager(OptimizationStrategy.Default)
    stop = manager.pass_names.index(last_pass)
    pipeline = passes.PassPipeline()
    for pass_obj in manager.passes[: stop + 1]:
        pipeline.add_pass(pass_obj)
    return pipeline.run(program)


def _chunk_lifetime_stats(program: ir.Program) -> tuple[list[set[int]], int]:
    groups: list[set[int]] = []
    current_bases: set[int] = set()
    barrier_count = 0

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):
            name = stmt.var.name_hint
            tile_type = stmt.var.type
            if "__chunk_keep_tmp_" in name and isinstance(tile_type, ir.TileType):
                if tile_type.memref is not None:
                    current_bases.add(tile_type.memref.base_.unique_id)
            super().visit_assign_stmt(stmt)

        def visit_call(self, call):
            nonlocal barrier_count, current_bases
            if call.op.name == "system.bar_all":
                barrier_count += 1
                groups.append(current_bases)
                current_bases = set()
            super().visit_call(call)

    _Collector().visit_program(program)
    return groups, barrier_count


def _static_shape(expr: ir.Expr) -> tuple[int, ...]:
    tile_type = expr.type
    assert isinstance(tile_type, ir.TileType)
    shape: list[int] = []
    for dim in tile_type.shape:
        assert isinstance(dim, ir.ConstInt)
        shape.append(dim.value)
    return tuple(shape)


def _static_tuple(expr: ir.Expr) -> tuple[int, ...]:
    assert isinstance(expr, ir.MakeTuple)
    values: list[int] = []
    for element in expr.elements:
        assert isinstance(element, ir.ConstInt)
        values.append(element.value)
    return tuple(values)


def _expanded_packing_ops(
    program: ir.Program,
) -> tuple[
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    list[tuple[int, ...]],
    Counter[tuple[int, ...]],
    Counter[tuple[int, ...]],
    list[tuple[int, ...]],
]:
    source_load_shapes: list[tuple[int, ...]] = []
    result_load_shapes: list[tuple[int, ...]] = []
    quant_input_shapes: list[tuple[int, ...]] = []
    data_store_offsets: Counter[tuple[int, ...]] = Counter()
    scale_store_offsets: Counter[tuple[int, ...]] = Counter()
    transpose_output_shapes: list[tuple[int, ...]] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, call):
            if call.op.name == "tile.load" and call.type.dtype == ir.DataType.FP32:
                source_load_shapes.append(_static_shape(call))
            elif call.op.name == "tile.load" and call.type.dtype in {
                ir.DataType.FP8E4M3FN,
                ir.DataType.FP8E8M0,
            }:
                result_load_shapes.append(_static_shape(call))
            elif call.op.name == "tile.tquant_mx":
                quant_input_shapes.append(_static_shape(call.args[0]))
            elif call.op.name == "tile.store":
                value_type = call.args[0].type
                if isinstance(value_type, ir.TileType):
                    offsets = _static_tuple(call.args[1])
                    if value_type.dtype == ir.DataType.FP8E4M3FN and _static_shape(call.args[0]) == (16, 64):
                        data_store_offsets[offsets] += 1
                    elif value_type.dtype == ir.DataType.FP8E8M0 and _static_shape(call.args[0]) == (1, 32):
                        scale_store_offsets[offsets] += 1
            elif call.op.name == "tile.transpose":
                transpose_output_shapes.append(_static_shape(call))
            super().visit_call(call)

    _Collector().visit_program(program)
    return (
        source_load_shapes,
        result_load_shapes,
        quant_input_shapes,
        data_store_offsets,
        scale_store_offsets,
        transpose_output_shapes,
    )


class TestExpandMxPackedQuant:
    """Packed MX expansion bounds async lifetimes with explicit pipe drains."""

    def test_memory_reuse_is_bounded_by_pipe_all_drains(self):
        """A/B packed quant may reuse buffers only after a real pipe drain.

        PTO load/store/vector pipes execute asynchronously. The store-fused expansion
        keeps each chunk's load, quant result, and scale alive through ``bar_all``.
        MemoryReuse can then merge A into B after A's drain without merging buffers
        that are still in flight inside either chunk.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[32, 128], pl.FP32],
                b_nk: pl.Tensor[[64, 128], pl.FP32],
                a_q_out: pl.Out[pl.Tensor[[32, 128], pl.FP8E4M3FN]],
                a_s_out: pl.Out[pl.Tensor[[1, 128], pl.FP8E8M0]],
                b_q_out: pl.Out[pl.Tensor[[128, 64], pl.FP8E4M3FN]],
                b_s_out: pl.Out[pl.Tensor[[1, 256], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[32, 128], pl.FP8E4M3FN],
                pl.Tensor[[1, 128], pl.FP8E8M0],
                pl.Tensor[[128, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 256], pl.FP8E8M0],
            ]:
                a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [32, 128]), layout=pl.MX_A_ZZ)
                b_q, b_s = pl.quant_mx(pl.load(b_nk, [0, 0], [64, 128]), layout=pl.MX_B_NN)
                a_q_out = pl.store(a_q, [0, 0], a_q_out)
                a_s_out = pl.store(a_s, [0, 0], a_s_out)
                b_q_out = pl.store(b_q, [0, 0], b_q_out)
                b_s_out = pl.store(b_s, [0, 0], b_s_out)
                return a_q_out, a_s_out, b_q_out, b_s_out

        expanded = _run_default_through(Before, "ExpandMxPackedQuant")
        (
            source_load_shapes,
            result_load_shapes,
            quant_shapes,
            data_offsets,
            scale_offsets,
            transpose_shapes,
        ) = _expanded_packing_ops(expanded)

        # A is split into four [32, 32] source boxes and B into eight. Packed A
        # stores four [16, 64] boxes directly; packed B is assembled in Vec and
        # transposed once to its public [K, N] result layout.
        assert source_load_shapes == [(16, 64)] * 12
        assert result_load_shapes == []
        assert quant_shapes == [(32, 32)] * 12
        assert data_offsets == Counter({(0, 0): 1, (0, 64): 1, (16, 0): 1, (16, 64): 1})
        assert scale_offsets == Counter(
            {
                (0, 0): 2,
                (0, 32): 2,
                (0, 64): 2,
                (0, 96): 2,
                (0, 128): 1,
                (0, 160): 1,
                (0, 192): 1,
                (0, 224): 1,
            }
        )
        assert transpose_shapes == [(128, 64)]

        _, expanded_barriers = _chunk_lifetime_stats(expanded)
        # One drain for A's four-box chunk, one for B's eight-box chunk, and a
        # final drain for B's full-tile transpose/store.
        assert expanded_barriers == 3

        after = _run_default_through(Before, "MemoryReuse")
        chunk_groups, barrier_count = _chunk_lifetime_stats(after)

        # Each live chunk has distinct load/data/scale buffers per box. Across
        # A's drain, B is allowed to reuse at least some of A's physical bases.
        assert [len(group) for group in chunk_groups] == [4 * 3, 8 * 3, 0]
        assert chunk_groups[0] & chunk_groups[1]
        assert barrier_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
