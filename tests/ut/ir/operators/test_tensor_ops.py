# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comprehensive tests for tensor operations.

Tests cover:
- Memory operations (create, view, assemble)
- Matrix multiplication (matmul)
- Reduction operations (row_max, row_sum)
- Unary operations (exp, cast)
- Binary operations (maximum)
- Python helper functions
"""

import pytest
from pypto import ir
from pypto.pypto_core import DataType


def test_tensor_create():
    """Test tensor.create operation."""
    # Create a 2D tensor [4, 8] with FP32
    call = ir.op.tensor.create([4, 8], DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.create"

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_view():
    """Test tensor.view operation."""
    span = ir.Span.unknown()

    # Create a tensor variable [16, 32]
    dim16 = ir.ConstInt(16, DataType.INT32, span)
    dim32 = ir.ConstInt(32, DataType.INT32, span)
    tensor_type = ir.TensorType([dim16, dim32], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Create a view [8, 16]
    call = ir.op.tensor.view(tensor_var, [8, 16], [0, 0])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.view"

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16


def test_tensor_matmul():
    """Test tensor.matmul operation."""
    span = ir.Span.unknown()

    # Create two tensors [4, 8] and [8, 16]
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim16 = ir.ConstInt(16, DataType.INT32, span)

    lhs_type = ir.TensorType([dim4, dim8], DataType.FP32)
    rhs_type = ir.TensorType([dim8, dim16], DataType.FP32)

    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    # Perform matmul
    call = ir.op.tensor.matmul(lhs, rhs, out_dtype=DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.matmul"

    # Check result type - should be [4, 16]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_matmul_with_transpose():
    """Test tensor.matmul with transpose flags."""
    span = ir.Span.unknown()

    # Create tensors
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)

    lhs_type = ir.TensorType([dim8, dim4], DataType.FP16)  # [8, 4]
    rhs_type = ir.TensorType([dim8, dim4], DataType.FP16)  # [8, 4]

    lhs = ir.Var("lhs", lhs_type, span)
    rhs = ir.Var("rhs", rhs_type, span)

    # Transpose lhs: [8, 4]^T x [8, 4] -> [4, 4]
    call = ir.op.tensor.matmul(lhs, rhs, out_dtype=DataType.FP16, a_trans=True, b_trans=False)

    assert isinstance(call, ir.Call)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_row_max():
    """Test tensor.row_max reduction."""
    span = ir.Span.unknown()

    # Create a tensor [64, 128]
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Row max reduction (reduce last axis)
    call = ir.op.tensor.row_max(tensor_var, axis=-1, keep_dim=1)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_max"

    # Check result type - should be [64, 1]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_row_sum():
    """Test tensor.row_sum reduction."""
    span = ir.Span.unknown()

    # Create a tensor [64, 128]
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Row sum reduction (reduce last axis)
    call = ir.op.tensor.row_sum(tensor_var, axis=-1, keep_dim=1)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.row_sum"

    # Check result type - should be [64, 1]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_exp():
    """Test tensor.exp operation."""
    span = ir.Span.unknown()

    # Create a tensor [64, 128]
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Apply exp
    call = ir.op.tensor.exp(tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.exp"

    # Check result type - should preserve shape and dtype
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_cast():
    """Test tensor.cast operation."""
    span = ir.Span.unknown()

    # Create a FP16 tensor
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Cast to FP32
    call = ir.op.tensor.cast(tensor_var, DataType.FP32)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.cast"

    # Check result type - should preserve shape but change dtype
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_assemble():
    """Test tensor.assemble operation."""
    span = ir.Span.unknown()

    # Create target and source tensors
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    target_type = ir.TensorType([dim64, dim128], DataType.FP32)
    source_type = ir.TensorType([dim64, dim128], DataType.FP32)

    target = ir.Var("target", target_type, span)
    source = ir.Var("source", source_type, span)

    # Assemble at offset [0, 0]
    call = ir.op.tensor.assemble(target, source, [0, 0])

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.assemble"

    # Check result type - should be target type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_maximum():
    """Test tensor.maximum operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    type_a = ir.TensorType([dim64, dim1], DataType.FP32)
    type_b = ir.TensorType([dim64, dim1], DataType.FP32)

    var_a = ir.Var("a", type_a, span)
    var_b = ir.Var("b", type_b, span)

    # Element-wise maximum
    call = ir.op.tensor.maximum(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.maximum"

    # Check result type
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)


def test_tensor_mul():
    """Test tensor.mul operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim64 = ir.ConstInt(64, DataType.INT32, span)
    dim128 = ir.ConstInt(128, DataType.INT32, span)
    tensor_type = ir.TensorType([dim64, dim128], DataType.FP16)
    tensor_var = ir.Var("t", tensor_type, span)

    # Create a second tensor for multiplication (broadcasting: scalar tensor)
    scalar_tensor_type = ir.TensorType([], DataType.FP32)  # 0-D tensor (scalar)
    scalar_tensor_var = ir.Var("s", scalar_tensor_type, span)
    call = ir.op.tensor.mul(tensor_var, scalar_tensor_var)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.mul"


def test_tensor_add():
    """Test tensor.add operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Add
    call = ir.op.tensor.add(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.add"


def test_tensor_sub():
    """Test tensor.sub operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Subtract
    call = ir.op.tensor.sub(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.sub"


def test_tensor_div():
    """Test tensor.div operation."""
    span = ir.Span.unknown()

    # Create two tensors
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    tensor_type = ir.TensorType([dim8], DataType.FP32)
    var_a = ir.Var("a", tensor_type, span)
    var_b = ir.Var("b", tensor_type, span)

    # Divide
    call = ir.op.tensor.div(var_a, var_b)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.div"


def test_tensor_where():
    """Test tensor.where operation."""
    span = ir.Span.unknown()

    # Create condition, x, and y tensors [8]
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    condition_type = ir.TensorType([dim8], DataType.INT32)  # INT32 as boolean (0/1)
    x_type = ir.TensorType([dim8], DataType.FP32)
    y_type = ir.TensorType([dim8], DataType.FP32)

    condition = ir.Var("condition", condition_type, span)
    x = ir.Var("x", x_type, span)
    y = ir.Var("y", y_type, span)

    # Apply where
    call = ir.op.tensor.where(condition, x, y)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.where_tt"

    # Check result type - should be same shape as inputs, dtype promoted from x and y
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 1


def test_tensor_where_broadcasting():
    """Test tensor.where with broadcasting."""
    span = ir.Span.unknown()

    # Create tensors with different shapes that can broadcast
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    dim1 = ir.ConstInt(1, DataType.INT32, span)

    # condition: [4, 1], x: [4, 8], y: [8]
    condition_type = ir.TensorType([dim4, dim1], DataType.INT32)
    x_type = ir.TensorType([dim4, dim8], DataType.FP16)
    y_type = ir.TensorType([dim8], DataType.FP16)

    condition = ir.Var("cond", condition_type, span)
    x = ir.Var("x", x_type, span)
    y = ir.Var("y", y_type, span)

    # Apply where with broadcasting
    call = ir.op.tensor.where(condition, x, y)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.where_tt"

    # Result should broadcast to [4, 8]
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP16
    assert len(result_type.shape) == 2


def test_tensor_where_type_promotion():
    """Test tensor.where with type promotion between x and y."""
    span = ir.Span.unknown()

    # Create tensors with different dtypes
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    condition_type = ir.TensorType([dim8], DataType.INT32)
    x_type = ir.TensorType([dim8], DataType.FP16)  # FP16
    y_type = ir.TensorType([dim8], DataType.FP32)  # FP32

    condition = ir.Var("cond", condition_type, span)
    x = ir.Var("x", x_type, span)
    y = ir.Var("y", y_type, span)

    # Apply where - should promote to FP32
    call = ir.op.tensor.where(condition, x, y)

    assert isinstance(call, ir.Call)
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    # FP32 should win over FP16
    assert result_type.dtype == DataType.FP32


def test_tensor_where_ts():
    """Test tensor.where with tensor x and scalar y."""
    span = ir.Span.unknown()

    # Create condition and x tensor [8], y scalar
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    condition_type = ir.TensorType([dim8], DataType.INT32)
    x_type = ir.TensorType([dim8], DataType.FP32)

    condition = ir.Var("condition", condition_type, span)
    x = ir.Var("x", x_type, span)
    y_scalar = 0.5  # scalar value

    # Apply where with scalar y
    call = ir.op.tensor.where(condition, x, y_scalar)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.where_ts"

    # Check result type - should be same shape as x, dtype promoted
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 1


def test_tensor_where_st():
    """Test tensor.where with scalar x and tensor y."""
    span = ir.Span.unknown()

    # Create condition and y tensor [4, 8], x scalar
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    condition_type = ir.TensorType([dim4, dim8], DataType.INT32)
    y_type = ir.TensorType([dim4, dim8], DataType.FP16)

    condition = ir.Var("condition", condition_type, span)
    x_scalar = 1.0  # scalar value
    y = ir.Var("y", y_type, span)

    # Apply where with scalar x
    call = ir.op.tensor.where(condition, x_scalar, y)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.where_st"

    # Check result type - should be same shape as y
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_where_ss():
    """Test tensor.where with both x and y as scalars."""
    span = ir.Span.unknown()

    # Create condition tensor [4, 8], both x and y scalars
    dim4 = ir.ConstInt(4, DataType.INT32, span)
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    condition_type = ir.TensorType([dim4, dim8], DataType.INT32)

    condition = ir.Var("condition", condition_type, span)
    x_scalar = 1.0  # scalar value
    y_scalar = 0.0  # scalar value

    # Apply where with both scalars
    call = ir.op.tensor.where(condition, x_scalar, y_scalar)

    assert isinstance(call, ir.Call)
    assert call.op.name == "tensor.where_ss"

    # Check result type - should be same shape as condition
    result_type = call.type
    assert isinstance(result_type, ir.TensorType)
    assert result_type.dtype == DataType.FP32
    assert len(result_type.shape) == 2


def test_tensor_where_all_four_combinations():
    """Test all four parameter combinations of tensor.where."""
    span = ir.Span.unknown()

    # Setup common condition
    dim8 = ir.ConstInt(8, DataType.INT32, span)
    condition_type = ir.TensorType([dim8], DataType.INT32)
    condition = ir.Var("cond", condition_type, span)

    # Setup tensors
    x_type = ir.TensorType([dim8], DataType.FP32)
    y_type = ir.TensorType([dim8], DataType.FP32)
    x_tensor = ir.Var("x", x_type, span)
    y_tensor = ir.Var("y", y_type, span)

    # Test 1: Both tensors
    call1 = ir.op.tensor.where(condition, x_tensor, y_tensor)
    assert call1.op.name == "tensor.where_tt"

    # Test 2: Tensor x, scalar y
    call2 = ir.op.tensor.where(condition, x_tensor, 0.5)
    assert call2.op.name == "tensor.where_ts"

    # Test 3: Scalar x, tensor y
    call3 = ir.op.tensor.where(condition, 1.0, y_tensor)
    assert call3.op.name == "tensor.where_st"

    # Test 4: Both scalars
    call4 = ir.op.tensor.where(condition, 1.0, 0.0)
    assert call4.op.name == "tensor.where_ss"

    # All should return TensorType with same shape as condition
    for call in [call1, call2, call3, call4]:
        result_type = call.type
        assert isinstance(result_type, ir.TensorType)
        assert len(result_type.shape) == 1


def test_const_float():
    """Test ConstFloat expression creation and usage."""
    span = ir.Span.unknown()

    # Create a ConstFloat with FP32
    const_float = ir.ConstFloat(3.14, DataType.FP32, span)
    assert isinstance(const_float, ir.ConstFloat)
    assert const_float.value == 3.14
    assert const_float.dtype == DataType.FP32

    # Create a ConstFloat with FP16
    const_float_fp16 = ir.ConstFloat(2.718, DataType.FP16, span)
    assert isinstance(const_float_fp16, ir.ConstFloat)
    assert const_float_fp16.value == 2.718
    assert const_float_fp16.dtype == DataType.FP16

    # Test with negative value
    const_float_neg = ir.ConstFloat(-1.5, DataType.FP32, span)
    assert const_float_neg.value == -1.5

    # Test with zero
    const_float_zero = ir.ConstFloat(0.0, DataType.FP32, span)
    assert const_float_zero.value == 0.0


def test_operator_registration():
    """Test that all new operators are registered."""
    # Check that our new operators are registered
    assert ir.is_op_registered("tensor.create")
    assert ir.is_op_registered("tensor.view")
    assert ir.is_op_registered("tensor.matmul")
    assert ir.is_op_registered("tensor.row_max")
    assert ir.is_op_registered("tensor.row_sum")
    assert ir.is_op_registered("tensor.exp")
    assert ir.is_op_registered("tensor.cast")
    assert ir.is_op_registered("tensor.assemble")
    assert ir.is_op_registered("tensor.maximum")
    assert ir.is_op_registered("tensor.where_tt")
    assert ir.is_op_registered("tensor.where_ts")
    assert ir.is_op_registered("tensor.where_st")
    assert ir.is_op_registered("tensor.where_ss")


def test_get_new_ops():
    """Test getting new operator instances."""
    matmul_op = ir.get_op("tensor.matmul")
    assert matmul_op.name == "tensor.matmul"

    exp_op = ir.get_op("tensor.exp")
    assert exp_op.name == "tensor.exp"

    cast_op = ir.get_op("tensor.cast")
    assert cast_op.name == "tensor.cast"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
