# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 quant_mx-to-matmul_mx transfer examples.

1. mix: quant_mx and matmul_mx in one InCore, split into Vec/Cube by the compiler.
2. v2c: hand-written AIV/AIC with paired data (id=0) and scale (id=1) pipes.
3. gm: AIV stores to GM and AIC loads from GM without tpush_to_aic.
"""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

M, K, N = 64, 256, 64
GROUPS = K // 32
DATA_SLOT_SIZE = M * K
SCALE_SLOT_SIZE = M * GROUPS


def _input_tensor_specs() -> list[TensorSpec]:
    a = torch.zeros((M, K), dtype=torch.float32)
    for row in range(M):
        box = row // 16
        row_in_box = row % 16
        for group in range(GROUPS):
            code = 100 + (box * 13 + row_in_box * 7 + group * 3) % 64
            a[row, group * 32 : (group + 1) * 32] = float(2.0 ** (code - 127))
    b = torch.zeros((K, N), dtype=torch.float32)
    for group in range(GROUPS):
        b[group * 32 : (group + 1) * 32, group] = 1.0
    unit_scale = torch.full((1, GROUPS * N), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return [
        TensorSpec("a", [M, K], DataType.FP32, init_value=a),
        TensorSpec("b", [K, N], DataType.FP8E4M3FN, init_value=b.to(torch.float8_e4m3fn)),
        TensorSpec("b_scale", [1, GROUPS * N], DataType.FP8E8M0, init_value=unit_scale),
    ]


def _compute_expected(tensors, params=None):
    tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"].to(torch.float32))


# ======================================================================
# Example 1: InCore mix (compiler-driven ExpandMixedKernel)
# ======================================================================


@pl.program
class MixProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        quant, scale = pl.quant_mx(pl.load(a, [0, 0], [M, K]), layout=pl.MX_A_ZZ)
        lhs = pl.move(
            pl.move(
                quant,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            ),
            target_memory=pl.Mem.Left,
        )
        lhs_scale = pl.move(
            pl.move(
                scale,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.row_major,
            ),
            target_memory=pl.Mem.LeftScale,
        )
        b_scale_mx = pl.tensor.view(b_scale, [GROUPS, N], layout=pl.MX_B_NN)
        rhs = pl.move(pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right)
        rhs_scale = pl.move(
            pl.load(b_scale_mx, [0, 0], [GROUPS, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        return self.kernel(a, b, b_scale, out)


class MixCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_mix_64x256x64"

    def get_program(self) -> Any:
        return MixProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [*_input_tensor_specs(), TensorSpec("out", [M, N], DataType.FP32, is_output=True)]

    compute_expected = staticmethod(_compute_expected)


# ======================================================================
# Example 2: AIV quant + AIC matmul with paired tpush/tpop pipes
# ======================================================================


@pl.program
class V2CProgram:
    @pl.function(type=pl.FunctionType.AIV)
    def vector_quantize(self, a: pl.Tensor[[M, K], pl.FP32]):
        data_peer = pl.import_peer_buffer(name="v2c_mx_data", peer_func="cube_matmul")
        scale_peer = pl.import_peer_buffer(name="v2c_mx_scale", peer_func="cube_matmul")
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), data_peer, dir_mask=2, slot_size=DATA_SLOT_SIZE, id=0)
        pl.aiv_initialize_pipe(pl.const(0, pl.INT32), scale_peer, dir_mask=2, slot_size=SCALE_SLOT_SIZE, id=1)
        quant, scale = pl.quant_mx(pl.load(a, [0, 0], [M, K]), layout=pl.MX_A_ZZ)
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
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        data_slot = pl.reserve_buffer(name="v2c_mx_data", size=DATA_SLOT_SIZE, base=pl.AUTO)
        scale_slot = pl.reserve_buffer(name="v2c_mx_scale", size=SCALE_SLOT_SIZE, base=pl.AUTO)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), data_slot, dir_mask=2, slot_size=DATA_SLOT_SIZE, id=0)
        pl.aic_initialize_pipe(pl.const(0, pl.INT32), scale_slot, dir_mask=2, slot_size=SCALE_SLOT_SIZE, id=1)
        data_mat: pl.Tile[
            [M, K],
            pl.FP8E4M3FN,
            pl.Mem.Mat,
            pl.TileView(blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512),
        ] = pl.tpop_from_aiv(split=0, id=0)
        scale_mat: pl.Tile[
            [M, GROUPS],
            pl.FP8E8M0,
            pl.Mem.Mat,
            pl.TileView(blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.row_major, fractal=32),
        ] = pl.tpop_from_aiv(split=0, id=1)
        lhs = pl.move(data_mat, target_memory=pl.Mem.Left)
        lhs_scale = pl.move(scale_mat, target_memory=pl.Mem.LeftScale)
        pl.tfree_to_aiv(data_mat, id=0)
        pl.tfree_to_aiv(scale_mat, id=1)
        b_scale_mx = pl.tensor.view(b_scale, [GROUPS, N], layout=pl.MX_B_NN)
        rhs = pl.move(pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right)
        rhs_scale = pl.move(
            pl.load(b_scale_mx, [0, 0], [GROUPS, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

    @pl.function(type=pl.FunctionType.Group)
    def group_func(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        self.vector_quantize(a)
        return self.cube_matmul(b, b_scale, out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        return self.group_func(a, b, b_scale, out)


class V2CCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_v2c_64x256x64"

    def get_program(self) -> Any:
        return V2CProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [*_input_tensor_specs(), TensorSpec("out", [M, N], DataType.FP32, is_output=True)]

    compute_expected = staticmethod(_compute_expected)


# ======================================================================
# Example 3: AIV writes GM and AIC reads GM without tpush_to_aic
# ======================================================================


@pl.program
class GmProgram:
    @pl.function(type=pl.FunctionType.AIV)
    def vector_quantize(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        a_quant: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
        a_scale: pl.Out[pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]],
    ) -> tuple[pl.Tensor[[M, K], pl.FP8E4M3FN], pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]]:
        quant, scale = pl.quant_mx(pl.load(a, [0, 0], [M, K]), layout=pl.MX_A_ZZ)
        a_quant = pl.store(quant, [0, 0], a_quant)
        a_scale = pl.store(pl.reshape(scale, [1, M * GROUPS]), [0, 0], a_scale)
        return a_quant, a_scale

    @pl.function(type=pl.FunctionType.AIC)
    def cube_matmul(
        self,
        a_quant: pl.Tensor[[M, K], pl.FP8E4M3FN],
        a_scale: pl.Tensor[[1, M * GROUPS], pl.FP8E8M0],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        a_scale_mx = pl.tensor.view(a_scale, [M, GROUPS], layout=pl.MX_A_ZZ)
        lhs = pl.move(
            pl.load(a_quant, [0, 0], [M, K], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        lhs_scale = pl.move(
            pl.load(a_scale_mx, [0, 0], [M, GROUPS], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.LeftScale,
        )
        b_scale_mx = pl.tensor.view(b_scale, [GROUPS, N], layout=pl.MX_B_NN)
        rhs = pl.move(pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat), target_memory=pl.Mem.Right)
        rhs_scale = pl.move(
            pl.load(b_scale_mx, [0, 0], [GROUPS, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        return pl.store(pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale), [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP8E4M3FN],
        b_scale: pl.Tensor[[1, GROUPS * N], pl.FP8E8M0],
        a_quant: pl.Out[pl.Tensor[[M, K], pl.FP8E4M3FN]],
        a_scale: pl.Out[pl.Tensor[[1, M * GROUPS], pl.FP8E8M0]],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        a_quant, a_scale = self.vector_quantize(a, a_quant, a_scale)
        return self.cube_matmul(a_quant, a_scale, b, b_scale, out)


class GmCase(PTOTestCase):
    __test__ = False

    def __init__(self, *, platform=None):
        super().__init__(RunConfig(rtol=1e-3, atol=1e-3), platform=platform)

    def get_name(self) -> str:
        return "quantized_matmul_mx_gm_64x256x64"

    def get_program(self) -> Any:
        return GmProgram

    def define_tensors(self) -> list[TensorSpec]:
        return [
            *_input_tensor_specs(),
            TensorSpec(
                "a_quant",
                [M, K],
                DataType.FP8E4M3FN,
                init_value=torch.zeros((M, K), dtype=torch.float8_e4m3fn),
            ),
            TensorSpec(
                "a_scale",
                [1, M * GROUPS],
                DataType.FP8E8M0,
                init_value=torch.zeros((1, M * GROUPS), dtype=torch.uint8).view(torch.float8_e8m0fnu),
            ),
            TensorSpec("out", [M, N], DataType.FP32, is_output=True),
        ]

    compute_expected = staticmethod(_compute_expected)


# ======================================================================
# One test class with three example entry points
# ======================================================================


@pytest.mark.platforms("a5")
class TestQuantizedMatmulMx:
    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_mix(self, test_runner, platform):
        result = test_runner.run(MixCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_v2c(self, test_runner, platform):
        result = test_runner.run(V2CCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"

    @pytest.mark.parametrize("platform", [pytest.param("a5", id="a5")])
    def test_gm(self, test_runner, platform):
        result = test_runner.run(GmCase(platform=platform))
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
