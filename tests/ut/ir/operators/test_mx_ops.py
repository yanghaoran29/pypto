# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX scale load path (TensorLayout.MX_*) and LeftScale/RightScale spaces."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import DataType


def _mx_tensor(name: str, shape: list[int], layout: ir.TensorLayout) -> ir.Var:
    span = ir.Span.unknown()
    return ir.Var(
        name,
        ir.TensorType(shape, DataType.FP8E8M0, tensor_view=ir.TensorView([], layout)),
        span,
    )


class TestMxLoad:
    def test_mx_tensor_layout_sets_fractal(self):
        span = ir.Span.unknown()
        tensor = _mx_tensor("t", [16, 2], ir.TensorLayout.MX_A_ZZ)
        call = ir.op.tile.load(
            tensor,
            [0, 0],
            [16, 2],
            target_memory=ir.MemorySpace.Mat,
            span=span,
        )
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP8E8M0
        assert call.type.tile_view is not None
        assert call.type.tile_view.fractal == 32

    def test_rejects_vec_target_with_mx_tensor_layout(self):
        # Explicit Vec must not silently rewrite to Mat for MX sources.
        span = ir.Span.unknown()
        tensor = _mx_tensor("t", [16, 2], ir.TensorLayout.MX_A_ZZ)
        with pytest.raises(Exception, match="Mat|Vec"):
            ir.op.tile.load(
                tensor,
                [0, 0],
                [16, 2],
                target_memory=ir.MemorySpace.Vec,
                span=span,
            )

    def test_mx_source_without_target_memory_stamps_mat_kwarg(self):
        """IR create_op_call of MX-layout tensor must stamp target_memory=Mat.

        DeduceTileLoadType already defaults the TileType to Mat+fractal=32; the
        Call kwargs must also carry target_memory so InferTileMemorySpace does
        not rebuild via GetImplicitTileView (ordinary Mat NZ drops fractal).
        """
        span = ir.Span.unknown()
        tensor = _mx_tensor("t", [16, 2], ir.TensorLayout.MX_A_ZZ)
        offsets = ir.MakeTuple(
            [ir.ConstInt(0, DataType.INDEX, span), ir.ConstInt(0, DataType.INDEX, span)], span
        )
        shapes = ir.MakeTuple(
            [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(2, DataType.INDEX, span)], span
        )
        call = ir.create_op_call(
            "tile.load",
            [tensor, offsets, shapes, shapes],
            {},
            span,
        )
        assert any(k == "target_memory" and v == ir.MemorySpace.Mat for k, v in call.kwargs.items())
        assert isinstance(call.type, ir.TileType)
        assert call.type.memory_space == ir.MemorySpace.Mat
        assert call.type.tile_view is not None
        assert call.type.tile_view.fractal == 32


class TestDtypeAndMemorySpace:
    def test_fp8e8m0_exists(self):
        assert DataType.FP8E8M0.get_bit() == 8
        assert DataType.FP8E8M0.to_string() == "fp8e8m0"
        assert pl.FP8E8M0 == DataType.FP8E8M0

    def test_left_right_scale_spaces(self):
        assert ir.MemorySpace.LeftScale == pl.Mem.LeftScale
        assert ir.MemorySpace.RightScale == pl.Mem.RightScale
