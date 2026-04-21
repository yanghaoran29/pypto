# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-exports from ``simpler.task_interface`` and ``simpler.worker``.

All C++ nanobind types (DataType, ChipCallable, ChipStorageTaskArgs, etc.) and
torch-aware helpers (make_tensor_arg, scalar_to_uint64) come from the
``simpler`` package installed via ``pip install simpler``.

Some torch helpers (``make_tensor_arg``, ``torch_dtype_to_datatype``) live in
``simpler_setup.torch_interop`` after simpler PR #618; older simpler versions
keep them in ``simpler.task_interface``. Try the legacy path first so we work
against both pinned (a9f3ea9) and post-#618 simpler.
"""

from simpler.task_interface import (  # pyright: ignore[reportMissingImports]
    ChipCallable,
    ChipCallConfig,
    ChipStorageTaskArgs,
    CoreCallable,
    scalar_to_uint64,
)

try:
    from simpler.task_interface import (  # type: ignore[attr-defined]  # pyright: ignore[reportMissingImports]
        make_tensor_arg,
    )
except ImportError:
    from simpler_setup.torch_interop import make_tensor_arg  # pyright: ignore[reportMissingImports]

from simpler.worker import Worker  # pyright: ignore[reportMissingImports]

__all__ = [
    "ChipCallable",
    "ChipCallConfig",
    "ChipStorageTaskArgs",
    "CoreCallable",
    "Worker",
    "make_tensor_arg",
    "scalar_to_uint64",
]
