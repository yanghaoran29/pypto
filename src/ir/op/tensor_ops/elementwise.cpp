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

/**
 * @file elementwise.cpp
 * @brief Element-wise tensor operations (Add, Sub, Mul, Div)
 *
 * This file implements element-wise tensor operations that support
 * N-dimensional tensors with NumPy-style broadcasting.
 */

#include <memory>
#include <string>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {

TypePtr DeduceTensorOpElementwiseBinaryType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Try TensorType first
  auto tensor_type1 = As<TensorType>(args[0]->GetType());
  auto tensor_type2 = As<TensorType>(args[1]->GetType());

  CHECK(tensor_type1) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(tensor_type2) << "The operator " << op_name
                      << " requires second argument to be a TensorType, but got "
                      << args[1]->GetType()->TypeName();

  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, tensor_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  auto broadcast_result = BroadcastShapes(tensor_type1->shape_, tensor_type2->shape_);
  CHECK(broadcast_result.success) << "The operator " << op_name << " requires compatible shapes, but got "
                                  << tensor_type1->shape_ << " and " << tensor_type2->shape_;

  return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
}

TypePtr DeduceTensorOpElementwiseScalarType(const std::vector<ExprPtr>& args,
                                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                                            const std::string& op_name) {
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  auto tensor_type1 = As<TensorType>(args[0]->GetType());
  auto scalar_type2 = As<ScalarType>(args[1]->GetType());

  CHECK(tensor_type1) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                      << args[0]->GetType()->TypeName();
  CHECK(scalar_type2) << "The operator " << op_name
                      << " requires second argument to be a ScalarType, but got "
                      << args[1]->GetType()->TypeName();

  // TensorType + ScalarType - result is TensorType with same shape as first argument
  auto result_dtype = PromoteDataTypes(tensor_type1->dtype_, scalar_type2->dtype_);
  CHECK(result_dtype) << "The operator " << op_name << " requires compatible data types, but got "
                      << args[0]->GetType()->TypeName() << " and " << args[1]->GetType()->TypeName();

  return std::make_shared<TensorType>(tensor_type1->shape_, *result_dtype);
}

// ============================================================================
// Registration Function for Tensor Element-wise Operations
// ============================================================================

REGISTER_OP("tensor.add")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.add");
    });

REGISTER_OP("tensor.add_scalar")
    .set_op_category("TensorOp")
    .set_description("Element-wise addition of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.add_scalar");
    });

REGISTER_OP("tensor.sub")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.sub");
    });

REGISTER_OP("tensor.sub_scalar")
    .set_op_category("TensorOp")
    .set_description("Element-wise subtraction of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.sub_scalar");
    });

REGISTER_OP("tensor.mul")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.mul");
    });

REGISTER_OP("tensor.mul_scalar")
    .set_op_category("TensorOp")
    .set_description("Element-wise multiplication of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.mul_scalar");
    });

REGISTER_OP("tensor.div")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.div");
    });

REGISTER_OP("tensor.div_scalar")
    .set_op_category("TensorOp")
    .set_description("Element-wise division of tensor and scalar")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseScalarType(args, kwargs, "tensor.div_scalar");
    });

REGISTER_OP("tensor.maximum")
    .set_op_category("TensorOp")
    .set_description("Element-wise maximum of two tensors with broadcasting")
    .add_argument("lhs", "Left-hand side tensor (TensorType)")
    .add_argument("rhs", "Right-hand side tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceTensorOpElementwiseBinaryType(args, kwargs, "tensor.maximum");
    });

REGISTER_OP("tensor.where_tt")
    .set_op_category("TensorOp")
    .set_description("Element-wise selection based on condition: where(condition, x, y) returns x where condition is true, y otherwise")
    .add_argument("condition", "Condition tensor (TensorType, typically INT32 or BOOL)")
    .add_argument("x", "Tensor to select from when condition is true (TensorType)")
    .add_argument("y", "Tensor to select from when condition is false (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "The operator tensor.where requires exactly 3 arguments, but got "
                              << args.size();

      auto condition_type = As<TensorType>(args[0]->GetType());
      auto x_type = As<TensorType>(args[1]->GetType());
      auto y_type = As<TensorType>(args[2]->GetType());

      CHECK(condition_type) << "The operator tensor.where requires first argument (condition) to be a TensorType, but got "
                            << args[0]->GetType()->TypeName();
      CHECK(x_type) << "The operator tensor.where requires second argument (x) to be a TensorType, but got "
                    << args[1]->GetType()->TypeName();
      CHECK(y_type) << "The operator tensor.where requires third argument (y) to be a TensorType, but got "
                    << args[2]->GetType()->TypeName();

      // Promote data types between x and y
      auto result_dtype = PromoteDataTypes(x_type->dtype_, y_type->dtype_);
      CHECK(result_dtype) << "The operator tensor.where requires compatible data types for x and y, but got "
                          << args[1]->GetType()->TypeName() << " and " << args[2]->GetType()->TypeName();

      // Broadcast shapes: condition, x, y
      auto broadcast_xy = BroadcastShapes(x_type->shape_, y_type->shape_);
      CHECK(broadcast_xy.success) << "The operator tensor.where requires compatible shapes for x and y, but got "
                                   << x_type->shape_ << " and " << y_type->shape_;

      auto broadcast_result = BroadcastShapes(condition_type->shape_, broadcast_xy.shape);
      CHECK(broadcast_result.success) << "The operator tensor.where requires compatible shapes for condition and x/y, but got condition "
                                      << condition_type->shape_ << " and x/y " << broadcast_xy.shape;

      return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
    });

REGISTER_OP("tensor.where_ts")
    .set_op_category("TensorOp")
    .set_description("Element-wise selection: where(condition, x, y) with scalar y")
    .add_argument("condition", "Condition tensor (TensorType)")
    .add_argument("x", "Tensor to select from when condition is true (TensorType)")
    .add_argument("y", "Scalar to select from when condition is false (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "The operator tensor.where_ts requires exactly 3 arguments, but got "
                              << args.size();

      auto condition_type = As<TensorType>(args[0]->GetType());
      auto x_type = As<TensorType>(args[1]->GetType());
      auto y_type = As<ScalarType>(args[2]->GetType());

      CHECK(condition_type) << "The operator tensor.where_ts requires first argument (condition) to be a TensorType, but got "
                            << args[0]->GetType()->TypeName();
      CHECK(x_type) << "The operator tensor.where_ts requires second argument (x) to be a TensorType, but got "
                    << args[1]->GetType()->TypeName();
      CHECK(y_type) << "The operator tensor.where_ts requires third argument (y) to be a ScalarType, but got "
                    << args[2]->GetType()->TypeName();

      // Promote data types between x tensor and y scalar
      auto result_dtype = PromoteDataTypes(x_type->dtype_, y_type->dtype_);
      CHECK(result_dtype) << "The operator tensor.where_ts requires compatible data types for x and y, but got "
                          << args[1]->GetType()->TypeName() << " and " << args[2]->GetType()->TypeName();

      // Result shape is broadcast of condition and x
      auto broadcast_result = BroadcastShapes(condition_type->shape_, x_type->shape_);
      CHECK(broadcast_result.success) << "The operator tensor.where_ts requires compatible shapes for condition and x, but got "
                                      << condition_type->shape_ << " and " << x_type->shape_;

      return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
    });

REGISTER_OP("tensor.where_st")
    .set_op_category("TensorOp")
    .set_description("Element-wise selection: where(condition, x, y) with scalar x")
    .add_argument("condition", "Condition tensor (TensorType)")
    .add_argument("x", "Scalar to select from when condition is true (ScalarType)")
    .add_argument("y", "Tensor to select from when condition is false (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "The operator tensor.where_st requires exactly 3 arguments, but got "
                              << args.size();

      auto condition_type = As<TensorType>(args[0]->GetType());
      auto x_type = As<ScalarType>(args[1]->GetType());
      auto y_type = As<TensorType>(args[2]->GetType());

      CHECK(condition_type) << "The operator tensor.where_st requires first argument (condition) to be a TensorType, but got "
                            << args[0]->GetType()->TypeName();
      CHECK(x_type) << "The operator tensor.where_st requires second argument (x) to be a ScalarType, but got "
                    << args[1]->GetType()->TypeName();
      CHECK(y_type) << "The operator tensor.where_st requires third argument (y) to be a TensorType, but got "
                    << args[2]->GetType()->TypeName();

      // Promote data types between x scalar and y tensor
      auto result_dtype = PromoteDataTypes(x_type->dtype_, y_type->dtype_);
      CHECK(result_dtype) << "The operator tensor.where_st requires compatible data types for x and y, but got "
                          << args[1]->GetType()->TypeName() << " and " << args[2]->GetType()->TypeName();

      // Result shape is broadcast of condition and y
      auto broadcast_result = BroadcastShapes(condition_type->shape_, y_type->shape_);
      CHECK(broadcast_result.success) << "The operator tensor.where_st requires compatible shapes for condition and y, but got "
                                      << condition_type->shape_ << " and " << y_type->shape_;

      return std::make_shared<TensorType>(broadcast_result.shape, *result_dtype);
    });

REGISTER_OP("tensor.where_ss")
    .set_op_category("TensorOp")
    .set_description("Element-wise selection: where(condition, x, y) with both x and y as scalars")
    .add_argument("condition", "Condition tensor (TensorType)")
    .add_argument("x", "Scalar to select from when condition is true (ScalarType)")
    .add_argument("y", "Scalar to select from when condition is false (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 3) << "The operator tensor.where_ss requires exactly 3 arguments, but got "
                              << args.size();

      auto condition_type = As<TensorType>(args[0]->GetType());
      auto x_type = As<ScalarType>(args[1]->GetType());
      auto y_type = As<ScalarType>(args[2]->GetType());

      CHECK(condition_type) << "The operator tensor.where_ss requires first argument (condition) to be a TensorType, but got "
                            << args[0]->GetType()->TypeName();
      CHECK(x_type) << "The operator tensor.where_ss requires second argument (x) to be a ScalarType, but got "
                    << args[1]->GetType()->TypeName();
      CHECK(y_type) << "The operator tensor.where_ss requires third argument (y) to be a ScalarType, but got "
                    << args[2]->GetType()->TypeName();

      // Promote data types between x and y scalars
      auto result_dtype = PromoteDataTypes(x_type->dtype_, y_type->dtype_);
      CHECK(result_dtype) << "The operator tensor.where_ss requires compatible data types for x and y, but got "
                          << args[1]->GetType()->TypeName() << " and " << args[2]->GetType()->TypeName();

      // Result shape is the condition shape
      return std::make_shared<TensorType>(condition_type->shape_, *result_dtype);
    });

}  // namespace ir
}  // namespace pypto
