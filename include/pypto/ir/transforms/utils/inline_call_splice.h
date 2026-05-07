/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the License file in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_IR_TRANSFORMS_UTILS_INLINE_CALL_SPLICE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_INLINE_CALL_SPLICE_H_

#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto {
namespace ir {
namespace inline_splice {

/// Splice a callee body at a call site (same contract as InlineFunctions pass).
std::vector<StmtPtr> SpliceInlineCall(const FunctionPtr& callee, const std::vector<ExprPtr>& args,
                                    const VarPtr& lhs, const Span& call_site_span);

}  // namespace inline_splice
}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_INLINE_CALL_SPLICE_H_
