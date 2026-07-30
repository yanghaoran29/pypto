# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for MX LeftScale/RightScale memspace and MX-layout load/move IR."""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.ir.op import tensor_ops as tensor
from pypto.ir.op import tile_ops as tile


@pytest.fixture(autouse=True)
def _ascend950():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        yield
    finally:
        backend.reset_for_testing()


def _const_shape(rows: int, cols: int):
    span = ir.Span.unknown()
    return [ir.ConstInt(rows, DataType.INDEX, span), ir.ConstInt(cols, DataType.INDEX, span)]


def _mx_tensor_var(name: str, rows: int, cols: int, layout: ir.TensorLayout):
    return ir.Var(
        name,
        ir.TensorType(
            _const_shape(rows, cols),
            DataType.FP8E8M0,
            tensor_view=ir.TensorView([], layout),
        ),
        ir.Span.unknown(),
    )


def _tile_var(shape, dtype, *, memory=None, view=None, name="t"):
    tt = ir.TileType(shape, dtype, None, view, memory)
    return ir.Var(name, tt, ir.Span.unknown())


class TestMxScaleMemSpaces:
    def test_leftscale_move_layout_row_major_fractal32(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.row_major,
                slayout=ir.TileLayout.row_major,
                fractal=32,
            ),
        )
        call = tile.move(src, target_memory=ir.MemorySpace.LeftScale)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.LeftScale
        assert out.dtype == DataType.FP8E8M0  # ui8 promotes for loc=scaling
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major
        assert view.fractal == 32

    def test_rightscale_move_layout_col_major_fractal32(self):
        src = _tile_var(
            _const_shape(8, 16),
            DataType.UINT8,
            memory=ir.MemorySpace.Mat,
            view=ir.TileView(
                blayout=ir.TileLayout.col_major,
                slayout=ir.TileLayout.col_major,
                fractal=32,
            ),
        )
        call = tile.move(src, target_memory=ir.MemorySpace.RightScale)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.RightScale
        assert out.dtype == DataType.FP8E8M0
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_mx_tensor_layout_load_defaults_to_mat(self):
        tensor = _mx_tensor_var("s", 16, 8, ir.TensorLayout.MX_A_ZZ)
        call = tile.load(tensor, [0, 0], [16, 8])
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.Mat
        view = out.get_effective_tile_view()
        assert view.fractal == 32
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major

    def test_mx_b_layout_load_col_major(self):
        tensor = _mx_tensor_var("w_s", 8, 16, ir.TensorLayout.MX_B_NN)
        call = tile.load(tensor, [0, 0], [8, 16], target_memory=ir.MemorySpace.Mat)
        out = call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.Mat
        view = out.get_effective_tile_view()
        assert view.blayout == ir.TileLayout.col_major
        assert view.slayout == ir.TileLayout.col_major
        assert view.fractal == 32

    def test_mx_tensor_layout_rejects_vec_target(self):
        tensor = _mx_tensor_var("s", 16, 8, ir.TensorLayout.MX_A_ZZ)
        with pytest.raises(Exception, match="Mat|Vec"):
            tile.load(tensor, [0, 0], [16, 8], target_memory=ir.MemorySpace.Vec)

    def test_mx_tensor_layout_rejects_non_2d_load_window(self):
        tensor = _mx_tensor_var("s", 16, 8, ir.TensorLayout.MX_A_ZZ)
        with pytest.raises(Exception, match="2D load window"):
            tile.load(tensor, [0], [16])

    def test_scale_move_rejects_non_scale_dtype(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.FP16,
            memory=ir.MemorySpace.Mat,
        )
        with pytest.raises(Exception, match="UINT8|FP8E8M0"):
            tile.move(src, target_memory=ir.MemorySpace.LeftScale)

    def test_scale_move_rejects_non_mat_input(self):
        src = _tile_var(
            _const_shape(16, 8),
            DataType.FP8E8M0,
            memory=ir.MemorySpace.Vec,
        )
        with pytest.raises(Exception, match="Mat memory"):
            tile.move(src, target_memory=ir.MemorySpace.LeftScale)

    def test_scale_move_rejects_non_2d_input(self):
        span = ir.Span.unknown()
        src = _tile_var(
            [ir.ConstInt(16, DataType.INDEX, span)],
            DataType.FP8E8M0,
            memory=ir.MemorySpace.Mat,
        )
        with pytest.raises(Exception, match="2D tile"):
            tile.move(src, target_memory=ir.MemorySpace.RightScale)

    def test_scale_move_accepts_unresolved_mat_source(self):
        """tile.move into LeftScale must not hard-CHECK at construction when the
        source memory_space is still unresolved (nullopt). Mat-ness may only be
        inferred later by InferTileMemorySpace; the post-Infer verifier is the
        authority. Validates the relaxation of DeduceTileMoveType's Mat CHECK.
        """
        span = ir.Span.unknown()
        # Plain ND source; load built without target_memory -> memory_space None.
        nd_tensor = ir.Var("a", ir.TensorType(_const_shape(16, 8), DataType.FP8E8M0), span)
        zero = ir.ConstInt(0, DataType.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple(_const_shape(16, 8), span)
        load_call = ir.create_op_call("tile.load", [nd_tensor, offsets, shapes, shapes], {}, span)
        assert isinstance(load_call.type, ir.TileType)
        assert load_call.type.memory_space is None
        # Must not raise at construction even though source is not yet Mat.
        move_call = tile.move(load_call, target_memory=ir.MemorySpace.LeftScale)
        out = move_call.type
        assert isinstance(out, ir.TileType)
        assert out.memory_space == ir.MemorySpace.LeftScale

    def test_scale_move_rejects_vec_source_after_infer(self):
        """A load that Infer resolves to Vec, moved into LeftScale, must be rejected
        by the TileMemoryInferred verifier (not at construction). The source is
        unresolved (nullopt) at construction so DeduceTileMoveType's relaxed CHECK
        passes; after Infer it becomes Vec and the verifier must flag it.
        """
        span = ir.Span.unknown()
        nd_tensor = ir.Var("a", ir.TensorType(_const_shape(16, 8), DataType.FP8E8M0), span)
        out = ir.Var("out", ir.TensorType(_const_shape(16, 8), DataType.FP8E8M0), span)
        zero = ir.ConstInt(0, DataType.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple(_const_shape(16, 8), span)
        # No target_memory -> memory_space None at construction; Infer defaults to Vec.
        load_call = ir.create_op_call("tile.load", [nd_tensor, offsets, shapes, shapes], {}, span)
        tas = ir.Var("tas", load_call.type, span)
        move_call = tile.move(tas, target_memory=ir.MemorySpace.LeftScale)
        las = ir.Var("las", move_call.type, span)
        # Sink the load via a Vec store so the function is well-formed and the
        # load resolves to Vec (no Mat demand anywhere).
        store_call = tile.store(tas, [0, 0], out)
        result = ir.Var("result", out.type, span)
        body = ir.SeqStmts(
            [
                ir.AssignStmt(tas, load_call, span),
                ir.AssignStmt(las, move_call, span),
                ir.AssignStmt(result, store_call, span),
                ir.ReturnStmt([result], span),
            ],
            span,
        )
        func = ir.Function(
            "mx_move_vec_src",
            [(nd_tensor, ir.ParamDirection.In), (out, ir.ParamDirection.Out)],
            [out.type],
            body,
            span,
            ir.FunctionType.InCore,
        )
        prog = ir.Program([func], "mx_move_vec_src", span)
        # Infer resolves the load to Vec, then rebuilds the tile.move with that
        # Vec source; DeduceTileMoveType's (relaxed) CHECK re-fires on the
        # now-resolved Vec source and rejects it. The post-Infer verifier is the
        # authority for any path that does not rebuild the move.
        with pytest.raises(Exception, match="requires the input tile to be in Mat"):
            passes.infer_tile_memory_space()(prog)

    def test_scale_move_verifier_rejects_vec_iter_arg(self):
        span = ir.Span.unknown()
        shape = _const_shape(16, 8)
        vec_type = ir.TileType(shape, DataType.FP8E8M0, None, None, ir.MemorySpace.Vec)
        init = ir.Var("init", vec_type, span)
        carried = ir.IterArg("carried", vec_type, init, span)
        scale_type = ir.TileType(
            shape,
            DataType.FP8E8M0,
            None,
            ir.TileView(
                blayout=ir.TileLayout.row_major,
                slayout=ir.TileLayout.row_major,
                fractal=32,
            ),
            ir.MemorySpace.LeftScale,
        )
        move = ir.Call(
            ir.get_op("tile.move"),
            [carried],
            {"target_memory": ir.MemorySpace.LeftScale},
            scale_type,
            span,
        )
        result = ir.Var("result", scale_type, span)
        function = ir.Function(
            "mx_move_vec_iter_arg",
            [init],
            [],
            ir.SeqStmts([ir.AssignStmt(result, move, span), ir.ReturnStmt([], span)], span),
            span,
            ir.FunctionType.InCore,
        )
        program = ir.Program([function], "mx_move_vec_iter_arg", span)
        properties = passes.IRPropertySet()
        properties.insert(passes.IRProperty.TileMemoryInferred)

        diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)

        assert any("requires input in Mat memory" in diagnostic.message for diagnostic in diagnostics)

    @pytest.mark.parametrize(
        ("space", "blayout", "slayout"),
        [
            (ir.MemorySpace.LeftScale, ir.TileLayout.col_major, ir.TileLayout.row_major),
            (ir.MemorySpace.RightScale, ir.TileLayout.col_major, ir.TileLayout.row_major),
        ],
    )
    def test_scale_move_rejects_layout_override(self, space, blayout, slayout):
        src = _tile_var(_const_shape(16, 8), DataType.FP8E8M0, memory=ir.MemorySpace.Mat)
        with pytest.raises(Exception, match="hardware-fixed layout"):
            tile.move(src, target_memory=space, blayout=blayout, slayout=slayout)

    @pytest.mark.parametrize("space", [ir.MemorySpace.LeftScale, ir.MemorySpace.RightScale])
    def test_tile_create_rejects_scale_memory(self, space):
        with pytest.raises(Exception, match="does not support target_memory=LeftScale/RightScale"):
            tile.create([16, 8], DataType.FP8E8M0, target_memory=space)

    def test_mx_load_rejects_strided_tensor_view(self):
        shape = _const_shape(8, 8)
        tensor = ir.Var(
            "s",
            ir.TensorType(
                shape,
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([16, 1], ir.TensorLayout.MX_A_ZZ),
            ),
            ir.Span.unknown(),
        )
        with pytest.raises(Exception, match="packed 2D"):
            tile.load(tensor, [0, 0], [8, 8], target_memory=ir.MemorySpace.Mat)

    def test_mx_load_rejects_tensor_slice_with_nonzero_offset(self):
        source = _mx_tensor_var("source", 16, 8, ir.TensorLayout.MX_A_ZZ)
        sliced = tensor.slice(source, [8, 8], [1, 0])
        with pytest.raises(Exception, match="tensor.slice|start_offset|base offset"):
            tile.load(sliced, [0, 0], [8, 8], target_memory=ir.MemorySpace.Mat)

    def test_mx_load_rejects_tensor_slice_with_zero_offset(self):
        """Any start_offset presence rejects MX load, including ConstInt(0)."""
        source = _mx_tensor_var("source", 16, 8, ir.TensorLayout.MX_A_ZZ)
        sliced = tensor.slice(source, [16, 8], [0, 0])
        assert sliced.type.tensor_view is not None
        assert sliced.type.tensor_view.start_offset is not None
        sliced_var = ir.Var("sliced0", sliced.type, ir.Span.unknown())
        with pytest.raises(Exception, match="start_offset|sliced tensor"):
            tile.load(sliced_var, [0, 0], [16, 8], target_memory=ir.MemorySpace.Mat)

    def test_tensor_view_start_offset_affects_structural_equal(self):
        """CSE/equal must not collapse sliced vs unsliced MX tensors."""
        span = ir.Span.unknown()
        shape = [ir.ConstInt(16, DataType.INDEX, span), ir.ConstInt(8, DataType.INDEX, span)]
        base = ir.TensorType(
            shape,
            DataType.FP8E8M0,
            tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
        )
        sliced = ir.TensorType(
            shape,
            DataType.FP8E8M0,
            tensor_view=ir.TensorView(
                [],
                ir.TensorLayout.MX_A_ZZ,
                start_offset=ir.ConstInt(0, DataType.INDEX, span),
            ),
        )
        assert not ir.structural_equal(base, sliced)
        assert ir.structural_hash(base) != ir.structural_hash(sliced)

    def test_mx_load_rejects_ssa_bound_tensor_slice(self):
        """SSA-bound Var from tensor.slice must fail MX load (not only direct Call)."""
        source = _mx_tensor_var("source", 16, 8, ir.TensorLayout.MX_A_ZZ)
        sliced_call = tensor.slice(source, [8, 8], [1, 0])
        assert sliced_call.type.tensor_view is not None
        assert sliced_call.type.tensor_view.start_offset is not None
        sliced_var = ir.Var("sliced", sliced_call.type, ir.Span.unknown())
        with pytest.raises(Exception, match="tensor.slice|start_offset|base offset"):
            tile.load(sliced_var, [0, 0], [8, 8], target_memory=ir.MemorySpace.Mat)

    def test_mx_load_rejects_ssa_bound_tensor_slice_dsl(self):
        """Natural DSL: sliced = pl.slice(...); pl.load(sliced, ...) must reject."""

        with pytest.raises(Exception, match="tensor.slice|start_offset|base offset"):

            @pl.program
            class Input:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(
                    self,
                    source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                ) -> pl.Tensor[[16, 8], pl.FP8E8M0]:
                    sliced = pl.slice(source, [8, 8], [1, 0])
                    _scale = pl.load(sliced, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                    return source

    def test_mx_load_rejects_slice_derived_orchestration_argument(self):
        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def forward(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                return source

            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                source_tuple = (source,)
                source_alias = source_tuple[0]
                _scale = pl.load(source_alias, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                return source

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                sliced = pl.slice(source, [8, 8], [1, 0])
                forwarded, _ = pl.submit(self.forward, sliced)
                result, _ = pl.submit(self.kernel, forwarded)
                return result

        program = passes.convert_to_ssa()(Input)
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        with pytest.raises(Exception, match="packed top-level tensor.*tensor.slice-derived"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_rejects_slice_forwarded_by_plain_tuple_call(self):
        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def forward(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> tuple[
                pl.Tensor[[8, 8], pl.FP8E8M0],
                pl.Tensor[[8, 8], pl.FP8E8M0],
            ]:
                return source, source

            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                _scale = pl.load(source, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                return source

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                sliced = pl.slice(source, [8, 8], [1, 0])
                forwarded, forwarded_copy = self.forward(sliced)
                result, task_id = pl.submit(self.kernel, forwarded)
                return result

        program = passes.convert_to_ssa()(Input)
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        with pytest.raises(Exception, match="packed top-level tensor.*tensor.slice-derived"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_rejects_slice_forwarded_through_tensor_view(self):
        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                _scale = pl.load(source, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                return source

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                source: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                sliced = pl.slice(source, [8, 8], [1, 0])
                viewed = pl.tensor.view(sliced, [8, 8])
                result, _ = pl.submit(self.kernel, viewed)
                return result

        program = passes.convert_to_ssa()(Input)
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        with pytest.raises(Exception, match="packed top-level tensor.*tensor.slice-derived"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_tracks_the_correct_plain_tuple_result(self):
        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def pair(
                self,
                first: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                second: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> tuple[
                pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ]:
                return first, second

            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                first: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                second: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                _, scale_source = self.pair(first, second)
                _scale = pl.load(scale_source, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                return first

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                first: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
                second: pl.Tensor[[16, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                sliced_second = pl.slice(second, [8, 8], [1, 0])
                result, _ = pl.submit(self.kernel, first, sliced_second)
                return result

        program = passes.convert_to_ssa()(Input)
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        with pytest.raises(Exception, match="parameter 1.*tensor.slice-derived"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_accepts_top_level_mx_orchestration_argument(self):
        """Packed top-level MX_A_ZZ args must pass the orchestration MX precondition."""

        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                _scale = pl.load(source, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                return source

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                result, _ = pl.submit(self.kernel, source)
                return result

        program = passes.convert_to_ssa()(Input)
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        # MX precondition must accept the packed MX top-level tensor. This
        # deliberately minimal convert_to_ssa-only fixture then reaches the
        # unrelated orchestration requirement that plain calls target AIC/AIV.
        with pytest.raises(Exception, match="InferFunctionCoreType expects AIC or AIV"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_accepts_return_that_does_not_alias_the_sliced_argument(self):
        @pl.program
        class Input:
            @pl.function(type=pl.FunctionType.InCore)
            def select_packed(
                self,
                ignored_slice: pl.Tensor[[8, 8], pl.FP8E8M0],
                packed: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                return packed

            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                source: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                _scale = pl.load(source, [0, 0], [8, 8], target_memory=pl.Mem.Mat)
                return source

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestration(
                self,
                source: pl.Tensor[[16, 8], pl.FP8E8M0],
                packed: pl.Tensor[[8, 8], pl.FP8E8M0, pl.MX_A_ZZ],
            ) -> pl.Tensor[[8, 8], pl.FP8E8M0]:
                sliced = pl.slice(source, [8, 8], [1, 0])
                selected = self.select_packed(sliced, packed)
                result, _ = pl.submit(self.kernel, selected)
                return result

        program = passes.convert_to_ssa()(Input)
        orchestration = next(
            function
            for function in program.functions.values()
            if function.func_type == pl.FunctionType.Orchestration
        )
        # The MX precondition must accept ``selected`` because the callee returns
        # its packed second parameter, not the sliced first parameter. This
        # deliberately minimal convert_to_ssa-only fixture then reaches the
        # unrelated orchestration requirement that plain calls target AIC/AIV.
        with pytest.raises(Exception, match="InferFunctionCoreType expects AIC or AIV"):
            codegen.generate_orchestration(program, orchestration)

    def test_mx_load_accepts_materialized_packed_mx_tensor_view(self):
        shape = _const_shape(8, 8)
        tensor = ir.Var(
            "s",
            ir.TensorType(
                shape,
                DataType.FP8E8M0,
                tensor_view=ir.TensorView([8, 1], ir.TensorLayout.MX_A_ZZ),
            ),
            ir.Span.unknown(),
        )
        call = tile.load(tensor, [0, 0], [8, 8], target_memory=ir.MemorySpace.Mat)
        assert isinstance(call.type, ir.TileType)
        assert call.type.tile_view.fractal == 32

    def test_mx_load_pto_codegen_uses_packed_source_view(self):
        span = ir.Span.unknown()
        tensor_type = ir.TensorType(
            _const_shape(16, 2),
            DataType.FP8E8M0,
            tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
        )
        builder = ir.IRBuilder()
        with builder.function("mx_load_codegen", type=ir.FunctionType.InCore) as function:
            source = function.param("source", tensor_type)
            scale = builder.let(
                "scale",
                tile.load(source, [0, 0], [16, 2], target_memory=ir.MemorySpace.Mat),
            )
            function.return_type(scale.type)
            builder.return_stmt(scale)

        program = ir.Program([function.get_result()], "mx_load_codegen", span)
        pto = codegen.PTOCodegen().generate(program)
        assert "pto.make_tensor_view %arg0" in pto
        assert "strides = [%c2_index, %c1_index] {layout = #pto.layout<mx_a_zz>}" in pto
        assert "pto.tload" in pto

    def test_infer_preserves_mx_fractal_when_target_memory_kwarg_absent(self):
        """Infer must keep fractal=32 when MX-layout load lacks target_memory kwarg.

        Simulates pre-stamp IR: type already Mat+fractal=32, kwargs empty of
        target_memory. Consumer move→LeftScale demands Mat; Infer must not
        rebuild via NZ Mat view.
        """
        span = ir.Span.unknown()
        shape = _const_shape(16, 2)
        tensor = ir.Var(
            "a_s",
            ir.TensorType(shape, DataType.FP8E8M0, tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ)),
            span,
        )
        out = ir.Var("out", ir.TensorType(shape, DataType.FP8E8M0), span)

        zero = ir.ConstInt(0, DataType.INDEX, span)
        offsets = ir.MakeTuple([zero, zero], span)
        shapes = ir.MakeTuple(shape, span)

        # Canonical create stamps Mat; strip it to exercise Infer's MX-source path.
        stamped = ir.create_op_call(
            "tile.load",
            [tensor, offsets, shapes, shapes],
            {},
            span,
        )
        stripped_kwargs = {k: v for k, v in stamped.kwargs.items() if k != "target_memory"}
        assert "target_memory" not in stripped_kwargs
        load_call = ir.Call(stamped.op, list(stamped.args), stripped_kwargs, stamped.type, span)

        tas = ir.Var("tas", load_call.type, span)
        move_call = tile.move(tas, target_memory=ir.MemorySpace.LeftScale)
        las = ir.Var("las", move_call.type, span)

        # Separate Vec load/store needs a non-MX source; use a plain ND twin.
        nd_tensor = ir.Var("a_nd", ir.TensorType(shape, DataType.FP8E8M0), span)
        vec_load = tile.load(nd_tensor, [0, 0], [16, 2], target_memory=ir.MemorySpace.Vec)
        vec_tile = ir.Var("vec_tile", vec_load.type, span)
        store_call = tile.store(vec_tile, [0, 0], out)
        result = ir.Var("result", out.type, span)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(tas, load_call, span),
                ir.AssignStmt(las, move_call, span),
                ir.AssignStmt(vec_tile, vec_load, span),
                ir.AssignStmt(result, store_call, span),
                ir.ReturnStmt([result], span),
            ],
            span,
        )
        func = ir.Function(
            "mx_infer_preserve",
            [
                (tensor, ir.ParamDirection.In),
                (nd_tensor, ir.ParamDirection.In),
                (out, ir.ParamDirection.Out),
            ],
            [out.type],
            body,
            span,
            ir.FunctionType.InCore,
        )
        prog = ir.Program([func], "mx_infer_preserve", span)
        after = passes.infer_tile_memory_space()(prog)

        load_calls = []

        def _collect(stmt):
            if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
                if stmt.value.op.name == "tile.load":
                    load_calls.append(stmt.value)
            if isinstance(stmt, ir.SeqStmts):
                for s in stmt.stmts:
                    _collect(s)

        for fn in after.functions.values():
            _collect(fn.body)

        def _is_mx_load(call):
            src = call.args[0]
            tv = getattr(src.type, "tensor_view", None)
            return tv is not None and tv.layout == ir.TensorLayout.MX_A_ZZ

        mx_loads = [c for c in load_calls if _is_mx_load(c)]
        assert len(mx_loads) == 1
        mx_load = mx_loads[0]
        assert mx_load.kwargs.get("target_memory") == ir.MemorySpace.Mat
        view = mx_load.type.get_effective_tile_view()
        assert view.fractal == 32
        assert view.blayout == ir.TileLayout.row_major
        assert view.slayout == ir.TileLayout.row_major
