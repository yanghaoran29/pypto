# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExpandMxPackedQuant."""

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


def _keepalive_bases(program: ir.Program) -> tuple[set[int], set[int]]:
    a_bases: set[int] = set()
    b_bases: set[int] = set()

    class _Collector(ir.IRVisitor):
        def visit_assign_stmt(self, stmt):
            name = stmt.var.name_hint
            tile_type = stmt.var.type
            if "__keep_tmp_" in name and isinstance(tile_type, ir.TileType):
                assert tile_type.memref is not None
                base_id = tile_type.memref.base_.unique_id
                if name.startswith(("a_q_out", "a_s_out")):
                    a_bases.add(base_id)
                elif name.startswith(("b_q_out", "b_s_out")):
                    b_bases.add(base_id)
            super().visit_assign_stmt(stmt)

    _Collector().visit_program(program)
    return a_bases, b_bases


class TestExpandMxPackedQuant:
    """Packed MX expansion preserves the hardware lifetime boundary between sites."""

    def test_memory_reuse_keeps_consecutive_a_b_quant_buffers_disjoint(self):
        """A/B packed quant in one InCore must not share per-box physical buffers.

        PTO load/store/vector pipes execute asynchronously. The store-fused expansion
        therefore keeps every per-box load, quant result, and scale alive until the
        original store position. Without those aliases MemoryReuse observes disjoint
        SSA lifetimes and merges A into B, corrupting the layout-ab device result.
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
                b_nk_q_out: pl.Out[pl.Tensor[[64, 128], pl.FP8E4M3FN]],
                b_q_out: pl.Out[pl.Tensor[[128, 64], pl.FP8E4M3FN]],
                b_s_out: pl.Out[pl.Tensor[[1, 256], pl.FP8E8M0]],
            ) -> tuple[
                pl.Tensor[[32, 128], pl.FP8E4M3FN],
                pl.Tensor[[1, 128], pl.FP8E8M0],
                pl.Tensor[[64, 128], pl.FP8E4M3FN],
                pl.Tensor[[128, 64], pl.FP8E4M3FN],
                pl.Tensor[[1, 256], pl.FP8E8M0],
            ]:
                a_q, a_s = pl.quant_mx(pl.load(a, [0, 0], [32, 128]), layout=pl.MX_A_ZZ)
                b_q, b_s = pl.quant_mx(pl.load(b_nk, [0, 0], [64, 128]), layout=pl.MX_B_NN)
                a_q_out = pl.store(a_q, [0, 0], a_q_out)
                a_s_out = pl.store(a_s, [0, 0], a_s_out)
                b_q_out = pl.store(b_q, [0, 0], b_q_out)
                b_s_out = pl.store(b_s, [0, 0], b_s_out)
                return a_q_out, a_s_out, b_nk_q_out, b_q_out, b_s_out

        after = _run_default_through(Before, "MemoryReuse")
        a_bases, b_bases = _keepalive_bases(after)

        # A has 4 boxes and B has 8. Each box keeps load/data/scale (3 buffers),
        # plus the final full data and scale reloads for the packed result.
        assert len(a_bases) == 4 * 3 + 2
        assert len(b_bases) == 8 * 3 + 2
        assert a_bases.isdisjoint(b_bases)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
