# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the LICENSE file in the root directory of this source tree for more details.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the temporary A5 soft-SYNCALL PTO-ISA compatibility include."""

from pathlib import Path

import pytest
from pypto.runtime.kernel_compiler import KernelCompiler


def test_a5_soft_syncall_compiles_through_compat_wrapper(monkeypatch, tmp_path):
    calls = []

    def fake_compile(self, source_path, **kwargs):
        calls.append((Path(source_path), kwargs))
        return b"object"

    monkeypatch.setattr(KernelCompiler.__mro__[1], "compile_incore", fake_compile)
    source = tmp_path / "kernel.cpp"
    source.write_text("SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gm, ub, 4);\n", encoding="utf-8")

    result = KernelCompiler("a5sim").compile_incore(
        str(source), pto_isa_root="/pto-isa", extra_include_dirs=["/caller/include"]
    )

    assert result == b"object"
    compiled_source, kwargs = calls[0]
    assert compiled_source == tmp_path / ".kernel_a5_syncall_compat.cpp"
    assert compiled_source.read_text(encoding="utf-8") == (
        '#include "a5_syncall_compat.hpp"\n#include "kernel.cpp"\n'
    )
    include_dirs = kwargs["extra_include_dirs"]
    assert Path(include_dirs[0]).name == "incore"
    assert include_dirs[1] == "/caller/include"


def test_a2a3_soft_syncall_uses_generated_source_directly(monkeypatch, tmp_path):
    calls = []

    def fake_compile(self, source_path, **kwargs):
        calls.append((Path(source_path), kwargs))
        return b"object"

    monkeypatch.setattr(KernelCompiler.__mro__[1], "compile_incore", fake_compile)
    source = tmp_path / "kernel.cpp"
    source.write_text("SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(gm, 4);\n", encoding="utf-8")

    result = KernelCompiler("a2a3sim").compile_incore(str(source), pto_isa_root="/pto-isa")

    assert result == b"object"
    assert calls[0][0] == source
    assert calls[0][1]["extra_include_dirs"] is None
    assert not (tmp_path / ".kernel_a5_syncall_compat.cpp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
