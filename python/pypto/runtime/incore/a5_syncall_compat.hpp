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

#ifndef PYTHON_PYPTO_RUNTIME_INCORE_A5_SYNCALL_COMPAT_HPP_
#define PYTHON_PYPTO_RUNTIME_INCORE_A5_SYNCALL_COMPAT_HPP_

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#if defined(__CPU_SIM)
#define __aicore__
#else
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif
#endif

#include <pto/pto-inst.hpp>

// PTOAS v0.60's A5 verifier and C++ emitter still use the workspace-bearing
// soft-SYNCALL overloads. The pto-isa revision pinned by Simpler 0659e29 uses
// the newer raw-GM implementation helpers instead. Restore the two public A5
// overloads that PTOAS emits and bridge them to the pinned helper ABI until
// PTOAS and pto-isa converge.
namespace pto {

template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::AIVOnly, typename GlobalData,
          typename UbTileData,
          std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<UbTileData> &&
                               UbTileData::Loc == TileType::Vec,
                           int> = 0>
PTO_INST void SYNCALL(GlobalData& gmWorkspace, UbTileData& ubWorkspace, int32_t usedCores = 0) {
  if constexpr (Mode == SyncAllMode::Hard) {
    (void)gmWorkspace;
    (void)ubWorkspace;
    (void)usedCores;
    SYNCALL_IMPL<CoreType>();
  } else {
#ifndef __PTO_AUTO__
    (void)ubWorkspace;
    SYNCALL_SOFT_IMPL<CoreType>(gmWorkspace.data(), usedCores);
#endif
  }
}

template <SyncAllMode Mode, SyncCoreType CoreType = SyncCoreType::Mix, typename GlobalData,
          typename UbTileData, typename L1TileData,
          std::enable_if_t<is_global_data_v<GlobalData> && is_tile_data_v<UbTileData> &&
                               UbTileData::Loc == TileType::Vec && is_tile_data_v<L1TileData> &&
                               L1TileData::Loc == TileType::Mat,
                           int> = 0>
PTO_INST void SYNCALL(GlobalData& gmWorkspace, UbTileData& ubWorkspace, L1TileData& l1Workspace,
                      int32_t usedCores = 0) {
  if constexpr (Mode == SyncAllMode::Hard) {
    (void)gmWorkspace;
    (void)ubWorkspace;
    (void)l1Workspace;
    (void)usedCores;
    SYNCALL_IMPL<CoreType>();
  } else {
#ifndef __PTO_AUTO__
    (void)ubWorkspace;
    (void)l1Workspace;
    SYNCALL_SOFT_MIX_IMPL<CoreType>(gmWorkspace.data(), usedCores);
#endif
  }
}

}  // namespace pto

#endif  // PYTHON_PYPTO_RUNTIME_INCORE_A5_SYNCALL_COMPAT_HPP_
