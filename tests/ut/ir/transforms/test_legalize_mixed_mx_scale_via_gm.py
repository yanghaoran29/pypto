# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for LegalizeMixedMxScaleViaGm."""

import pypto.language as pl
from pypto import ir, passes


def _ops(program: ir.Program, name: str) -> list[ir.Call]:
    found: list[ir.Call] = []

    class _V(ir.IRVisitor):
        def visit_call(self, call):
            if call.op.name == name:
                found.append(call)
            super().visit_call(call)

    _V().visit_program(program)
    return found


def _run_pass(program: ir.Program) -> ir.Program:
    return passes.legalize_mixed_mx_scale_via_gm()(program)


def _e8m0_tpush_count(program: ir.Program) -> int:
    n = 0
    for call in _ops(program, "tile.tpush_to_aic"):
        tt = call.args[0].type
        if isinstance(tt, ir.TileType) and tt.dtype == ir.DataType.FP8E8M0:
            n += 1
    return n


@pl.program
class _MxScaleV2CProgram:
    """Minimal AIV/AIC pair that still pushes E8M0 scale over V2C."""

    @pl.function(type=pl.FunctionType.AIV)
    def vector_quantize(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        data_peer = pl.import_peer_buffer(name="v2c_data", peer_func="cube_matmul")
        scale_peer = pl.import_peer_buffer(name="v2c_scale", peer_func="cube_matmul")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), data_peer, dir_mask=2, slot_size=8192, id=0)
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), scale_peer, dir_mask=2, slot_size=256, id=1)
        quant, scale = pl.quant_mx(pl.load(a, [0, 0], [32, 256]), layout=pl.MX_A_ZZ)
        quant_nz = pl.move(
            quant,
            target_memory=pl.Mem.Vec,
            blayout=pl.TileLayout.col_major,
            slayout=pl.TileLayout.row_major,
        )
        pl.tpush_to_aic(quant_nz, split=0, id=0)
        pl.tpush_to_aic(scale, split=0, id=1)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_matmul(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        data_slot = pl.reserve_buffer(name="v2c_data", size=8192, base=pl.AUTO)
        scale_slot = pl.reserve_buffer(name="v2c_scale", size=256, base=pl.AUTO)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), data_slot, dir_mask=2, slot_size=8192, id=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), scale_slot, dir_mask=2, slot_size=256, id=1)
        data_mat: pl.Tile[
            [32, 256],
            pl.FP8E4M3FN,
            pl.Mem.Mat,
            pl.TileView(
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
                fractal=512,
            ),
        ] = pl.tpop_from_aiv(split=0, id=0)
        scale_mat: pl.Tile[
            [32, 8],
            pl.FP8E8M0,
            pl.Mem.Mat,
            pl.TileView(
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.row_major,
                fractal=32,
            ),
        ] = pl.tpop_from_aiv(split=0, id=1)
        pl.tfree_to_aiv(data_mat, id=0)
        pl.tfree_to_aiv(scale_mat, id=1)
        zeros = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
        return pl.store(zeros, [0, 0], out)

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        self.vector_quantize(a, out)
        return self.cube_matmul(a, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        return self.group_func(a, out)


@pl.program
class _AlreadyGmProgram:
    """Scale already on GM; pass must be a no-op for E8M0 V2C rewrite."""

    @pl.function(type=pl.FunctionType.AIV)
    def vector_quantize(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        a_s_gm: pl.Out[pl.Tensor[[1, 256], pl.FP8E8M0]],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ):
        data_peer = pl.import_peer_buffer(name="v2c_data", peer_func="cube_matmul")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), data_peer, dir_mask=2, slot_size=8192, id=0)
        quant, scale = pl.quant_mx(pl.load(a, [0, 0], [32, 256]), layout=pl.MX_A_ZZ)
        quant_nz = pl.move(
            quant,
            target_memory=pl.Mem.Vec,
            blayout=pl.TileLayout.col_major,
            slayout=pl.TileLayout.row_major,
        )
        a_s_gm = pl.store(scale, [0, 0], a_s_gm)
        pl.tpush_to_aic(quant_nz, split=0, id=0)

    @pl.function(type=pl.FunctionType.AIC)
    def cube_matmul(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        a_s_gm: pl.Tensor[[1, 256], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        data_slot = pl.reserve_buffer(name="v2c_data", size=8192, base=pl.AUTO)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), data_slot, dir_mask=2, slot_size=8192, id=0)
        a_s_mx = pl.tensor.view(a_s_gm, [32, 8], layout=pl.MX_A_ZZ)
        data_mat: pl.Tile[
            [32, 256],
            pl.FP8E4M3FN,
            pl.Mem.Mat,
            pl.TileView(
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
                fractal=512,
            ),
        ] = pl.tpop_from_aiv(split=0, id=0)
        scale_mat = pl.load(a_s_mx, [0, 0], [32, 8], target_memory=pl.Mem.Mat)
        pl.tfree_to_aiv(data_mat, id=0)
        # Keep the load live for the structural check.
        _kept = scale_mat
        zeros = pl.tile.full([32, 32], dtype=pl.FP32, value=0.0)
        return pl.store(zeros, [0, 0], out)

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        a_s_gm: pl.Out[pl.Tensor[[1, 256], pl.FP8E8M0]],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        self.vector_quantize(a, a_s_gm, out)
        return self.cube_matmul(a, a_s_gm, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[32, 256], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        a_s_gm = pl.create_tensor([1, 256], dtype=pl.FP8E8M0)
        return self.group_func(a, a_s_gm, out)


class TestLegalizeMixedMxScaleViaGm:
    def test_rewrites_e8m0_v2c_to_gm_store_load(self):
        prog = _MxScaleV2CProgram
        assert _e8m0_tpush_count(prog) >= 1

        out = _run_pass(prog)
        assert _e8m0_tpush_count(out) == 0
        assert _ops(out, "tile.store"), "expected tile.store of packed scale to GM"
        assert _ops(out, "tensor.view"), "expected MX_A_ZZ view of GM scale"
        assert any(c.kwargs.get("target_memory") == ir.MemorySpace.Mat for c in _ops(out, "tile.load"))
        # Data V2C still present.
        assert any(
            isinstance(c.args[0].type, ir.TileType) and c.args[0].type.dtype == ir.DataType.FP8E4M3FN
            for c in _ops(out, "tile.tpush_to_aic")
        )

    def test_idempotent_when_no_e8m0_v2c(self):
        prog = _AlreadyGmProgram
        assert _e8m0_tpush_count(prog) == 0
        out = _run_pass(prog)
        assert _e8m0_tpush_count(out) == 0
        assert len(_ops(out, "tile.tpush_to_aic")) == len(_ops(prog, "tile.tpush_to_aic"))
