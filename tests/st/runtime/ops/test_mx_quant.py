# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime tests for MX quantization and affine dequantization."""

import pypto.language as pl
import pytest
import torch

ROWS, COLS = 16, 64
MX_GROUP_SIZE = 32


def _quant_inputs_and_golden() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact MXFP8 blocks with known E8M0 scales and byte outputs."""
    fp8_values = torch.tensor(
        [
            -448,
            -256,
            -128,
            -64,
            -32,
            -16,
            -8,
            -4,
            -2,
            -1,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.015625,
            0,
            0.015625,
            0.0625,
            0.125,
            0.25,
            0.5,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            448,
            0,
        ],
        dtype=torch.float32,
    )
    group_exponents = (torch.arange(ROWS * COLS // MX_GROUP_SIZE) % 4 - 1).reshape(ROWS, 2)
    scales = torch.pow(2.0, group_exponents).to(torch.float32)
    src = (fp8_values.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(ROWS, COLS)
    quantized = fp8_values.to(torch.float8_e4m3fn).view(torch.int8).repeat(ROWS, 2)
    scale_codes = (group_exponents + 127).to(torch.uint8).reshape(1, -1)
    return src.contiguous(), quantized.contiguous(), scale_codes.contiguous()


@pl.jit
def quant_mx_onboard(
    src: pl.Tensor[[ROWS, COLS], pl.FP32],
    out_quant: pl.Out[pl.Tensor[[ROWS, COLS], pl.INT8]],
    out_scale: pl.Out[pl.Tensor[[1, ROWS * COLS // MX_GROUP_SIZE], pl.UINT8]],
):
    """Quantize FP32 blocks into raw E4M3 bytes and E8M0 scale codes."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src_tile = pl.load(src, [0, 0], [ROWS, COLS])
        quantized, scale = pl.quant_mx(src_tile)
        # Inspect the exact encodings produced by the semantic FP8 results.
        quantized_raw = pl.reinterpret_view(quantized, pl.INT8)
        scale_raw = pl.reinterpret_view(scale, pl.UINT8)
        out_quant = pl.store(quantized_raw, [0, 0], out_quant)
        out_scale = pl.store(scale_raw, [0, 0], out_scale)
    return out_quant, out_scale


@pl.jit
def tdequant_onboard(
    src: pl.Tensor[[ROWS, COLS], pl.INT8],
    scale: pl.Tensor[[ROWS, 1], pl.FP32],
    offset: pl.Tensor[[ROWS, 1], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Apply per-row affine dequantization to an INT8 tile."""
    with pl.at(level=pl.Level.CORE_GROUP):
        src_tile = pl.load(src, [0, 0], [ROWS, COLS])
        scale_tile = pl.load(scale, [0, 0], [ROWS, 1])
        offset_tile = pl.load(offset, [0, 0], [ROWS, 1])
        result = pl.tdequant(src_tile, scale_tile, offset_tile)
        out = pl.store(result, [0, 0], out)
    return out


@pytest.mark.platforms("a5")
class TestMxQuantOnBoard:
    """Numerical execution coverage for the Ascend950-only quantization ops."""

    def test_quant_mx_onboard(self, test_config):
        quant_mx_onboard._cache.clear()
        src, expected_quant, expected_scale = _quant_inputs_and_golden()
        out_quant = torch.empty_like(expected_quant)
        out_scale = torch.empty_like(expected_scale)

        if test_config.codegen_only:
            quant_mx_onboard.compile(src, out_quant, out_scale, config=test_config)
            return

        quant_mx_onboard(src, out_quant, out_scale, config=test_config)
        torch.testing.assert_close(out_quant, expected_quant, rtol=0, atol=0)
        torch.testing.assert_close(out_scale, expected_scale, rtol=0, atol=0)

    def test_tdequant_onboard(self, test_config):
        tdequant_onboard._cache.clear()
        src = (torch.arange(ROWS * COLS).reshape(ROWS, COLS) % 31 - 15).to(torch.int8)
        scale = torch.pow(2.0, (torch.arange(ROWS) % 4 - 2).to(torch.float32)).reshape(ROWS, 1)
        offset = (torch.arange(ROWS) % 5 - 2).to(torch.float32).reshape(ROWS, 1)
        expected = (src.to(torch.float32) - offset) * scale
        out = torch.empty_like(expected)

        if test_config.codegen_only:
            tdequant_onboard.compile(src, scale, offset, out, config=test_config)
            return

        tdequant_onboard(src, scale, offset, out, config=test_config)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
