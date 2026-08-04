# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime tests for MX matmul.

The codegen unit tests verify the emitted ``tmatmul_mx`` and
``tget_scale_addr`` instructions. This test additionally executes the base and
accumulating forms on real Ascend950 hardware and compares their FP32 outputs
with torch. A second case covers the full FP32 -> MXFP8 quantization -> MX
matmul pipeline.
"""

import pypto.language as pl
import pytest
import torch

M, K, N = 16, 64, 32
MX_GROUP_SIZE = 32
SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2


def _pack_a_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A scales into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def _matmul_mx_golden(
    a: torch.Tensor,
    a_scale_codes: torch.Tensor,
    b: torch.Tensor,
    b_scale_codes: torch.Tensor,
) -> torch.Tensor:
    """Compute MXFP8 matmul with logical per-32-element E8M0 scales."""
    k_group = torch.arange(K) // MX_GROUP_SIZE
    a_scale = torch.pow(2.0, a_scale_codes.to(torch.float64) - 127)
    b_scale = torch.pow(2.0, b_scale_codes.to(torch.float64) - 127)
    a_scaled = a.to(torch.float64) * a_scale[:, k_group]
    b_scaled = b.to(torch.float64) * b_scale[k_group, :]
    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


def _exact_quantizable_matrix(rows: int, cols: int, *, transpose_pattern: bool = False) -> torch.Tensor:
    """Create FP32 blocks that quant_mx represents exactly with power-of-two scales."""
    pattern = (torch.arange(MX_GROUP_SIZE, dtype=torch.float32) % 9) - 4
    pattern[0] = 448
    pattern[1] = -448
    groups_per_row = cols // MX_GROUP_SIZE
    exponents = (torch.arange(rows * groups_per_row) % 4 - 1).reshape(rows, groups_per_row)
    if transpose_pattern:
        exponents = exponents.flip(0)
    scales = torch.pow(2.0, exponents).to(torch.float32)
    return (pattern.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(rows, cols).contiguous()


@pl.jit
def matmul_mx_onboard(
    a: pl.Tensor[[M, K], pl.FP8E4M3FN],
    a_scale: pl.Tensor[[M, K // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    b: pl.Tensor[[K, N], pl.FP8E4M3FN],
    b_scale: pl.Tensor[[K // 32, N], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    out_acc: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Run base and accumulating MX matmul with shared GM operands."""
    with pl.at(level=pl.Level.CORE_GROUP):
        lhs = pl.move(
            pl.load(a, [0, 0], [M, K], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Left,
        )
        lhs_scale = pl.move(
            pl.load(a_scale, [0, 0], [M, K // 32], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.LeftScale,
        )
        rhs = pl.move(
            pl.load(b, [0, 0], [K, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.Right,
        )
        rhs_scale = pl.move(
            pl.load(b_scale, [0, 0], [K // 32, N], target_memory=pl.Mem.Mat),
            target_memory=pl.Mem.RightScale,
        )
        base = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(base, [0, 0], out)
        accumulated = pl.matmul_mx_acc(base, lhs, lhs_scale, rhs, rhs_scale)
        out_acc = pl.store(accumulated, [0, 0], out_acc)
    return out, out_acc


@pl.jit
def quantized_matmul_mx_onboard(
    a: pl.Tensor[[M, K], pl.FP32],
    b_transposed: pl.Tensor[[N, K], pl.FP32],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
):
    """Quantize both FP32 operands on-chip before executing MX matmul."""
    with pl.at(level=pl.Level.CORE_GROUP):
        a_src = pl.load(a, [0, 0], [M, K])
        a_quant, a_scale = pl.quant_mx(a_src)
        a_mat = pl.move(a_quant, target_memory=pl.Mem.Mat)
        lhs = pl.move(a_mat, target_memory=pl.Mem.Left)
        a_scale_2d = pl.tile.reshape(a_scale, [M, K // MX_GROUP_SIZE])
        a_scale_mat = pl.move(
            a_scale_2d,
            target_memory=pl.Mem.Mat,
            blayout=pl.TileLayout.row_major,
            slayout=pl.TileLayout.row_major,
        )
        lhs_scale = pl.move(a_scale_mat, target_memory=pl.Mem.LeftScale)

        # Quantize B as [N, K] so every block still spans K, then transpose the
        # quantized values and scale groups into matmul's [K, N] / [K/32, N].
        b_src = pl.load(b_transposed, [0, 0], [N, K])
        b_quant, b_scale = pl.quant_mx(b_src)
        b_quant_t = pl.tile.transpose_view(b_quant)
        b_mat = pl.move(b_quant_t, target_memory=pl.Mem.Mat)
        rhs = pl.move(b_mat, target_memory=pl.Mem.Right)
        b_scale_2d = pl.tile.reshape(b_scale, [N, K // MX_GROUP_SIZE])
        b_scale_t = pl.tile.transpose_view(b_scale_2d)
        b_scale_mat = pl.move(
            b_scale_t,
            target_memory=pl.Mem.Mat,
            blayout=pl.TileLayout.col_major,
            slayout=pl.TileLayout.col_major,
        )
        rhs_scale = pl.move(b_scale_mat, target_memory=pl.Mem.RightScale)

        result = pl.matmul_mx(lhs, lhs_scale, rhs, rhs_scale)
        out = pl.store(result, [0, 0], out)
    return out


@pytest.mark.platforms("a5")
class TestMatmulMxOnBoard:
    """Numerical execution coverage for the Ascend950-only MX matmul path."""

    def test_matmul_mx_onboard(self, test_config):
        matmul_mx_onboard._cache.clear()

        generator = torch.Generator().manual_seed(19)
        a = torch.randint(-2, 3, (M, K), generator=generator).to(torch.float8_e4m3fn)
        b = torch.randint(-2, 3, (K, N), generator=generator).to(torch.float8_e4m3fn)

        # E8M0 codes [126, 130) represent reproducible random scales in
        # {0.5, 1, 2, 4}. Keep logical scales for the golden and pass packed
        # physical buffers to the kernel.
        a_scale_codes = torch.randint(126, 130, (M, K // MX_GROUP_SIZE), generator=generator).to(torch.uint8)
        b_scale_codes = torch.randint(126, 130, (K // MX_GROUP_SIZE, N), generator=generator).to(torch.uint8)
        assert torch.unique(a_scale_codes).numel() > 1
        assert torch.unique(b_scale_codes).numel() > 1
        a_scale = _pack_a_scale(a_scale_codes).view(torch.float8_e8m0fnu)
        b_scale = _pack_b_scale(b_scale_codes).view(torch.float8_e8m0fnu)

        base = _matmul_mx_golden(a, a_scale_codes, b, b_scale_codes)
        out = torch.zeros_like(base)
        out_acc = torch.zeros_like(base)

        if test_config.codegen_only:
            matmul_mx_onboard.compile(a, a_scale, b, b_scale, out, out_acc, config=test_config)
            return

        matmul_mx_onboard(a, a_scale, b, b_scale, out, out_acc, config=test_config)

        torch.testing.assert_close(out, base, rtol=0, atol=0)
        torch.testing.assert_close(out_acc, 2 * base, rtol=0, atol=0)

    def test_quantized_matmul_mx_onboard(self, test_config):
        quantized_matmul_mx_onboard._cache.clear()
        a = _exact_quantizable_matrix(M, K)
        b_transposed = _exact_quantizable_matrix(N, K, transpose_pattern=True)
        expected = torch.matmul(a, b_transposed.T)
        out = torch.empty_like(expected)

        if test_config.codegen_only:
            quantized_matmul_mx_onboard.compile(a, b_transposed, out, config=test_config)
            return

        quantized_matmul_mx_onboard(a, b_transposed, out, config=test_config)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
