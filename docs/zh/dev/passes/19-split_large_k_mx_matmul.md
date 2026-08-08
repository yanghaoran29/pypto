# SplitLargeKMxMatmul Pass

把静态大 K 的 MX matmul 切成 K=64 的 `matmul_mx` / `matmul_mx_acc` 链。

## 概述

当 `tile.matmul_mx` / `tile.matmul_mx_acc` / `tile.matmul_mx_bias` 的 lhs 具有静态 K，且 `K > 64`、`K % 64 == 0` 时，本 pass 将其改写为：

1. 首块：`matmul_mx`（或 `matmul_mx_bias`），操作数 slice 到 `K=64`，scale 切 `ceil(64/32)=2` 组；
2. 后续块：对剩余 K 切片调用 `matmul_mx_acc`。

动态 K 或非 64 对齐的 K 保持不变。改写后每个 MX matmul 的 K 均为 64，因此 pass 幂等。

**流水线位置**：紧接在 [`InferTileMemorySpace`](13-infer_tile_memory_space.md) 之后、[`InsertMxScaleAddr`](13-insert_mx_scale_addr.md) 之前，使每个 chunk 都能获得独立的 scale 地址绑定。

**前置 / 产出属性**：与 `InsertMxScaleAddr` 相同（`SSAForm`、`IncoreTileOps`、`SplitIncoreOrch`、`NormalizedStmtStructure`、`TileMemoryInferred`）。

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::SplitLargeKMxMatmul()` | `passes.split_large_k_mx_matmul()` | Program-level |

```python
from pypto.pypto_core import passes

after = passes.split_large_k_mx_matmul()(passes.infer_tile_memory_space()(program))
```

## 算法

对每个 InCore 变体函数体中的 MX matmul 赋值：

1. 解析 lhs `[M,K]`；若 K 非常量、`K <= 64` 或 `K % 64 != 0`，跳过。
2. 按 `k0 = 0, 64, …` 对 lhs/rhs 与 lhs_scale/rhs_scale 做 `tile.slice`（scale 偏移 `g0 = k0/32`）。
3. `ci==0` 发 `matmul_mx` / `matmul_mx_bias`；其后发 `matmul_mx_acc`。
4. 最后一块保留原结果 SSA 名，下游 use 无需改写。

## 相关

- 实现：`src/ir/transforms/split_large_k_mx_matmul_pass.cpp`
- 打包量化仅支持 `K=64`：[`ExpandMxPackedQuant`](12-expand_mx_packed_quant.md)
- 后续绑定：[`InsertMxScaleAddr`](13-insert_mx_scale_addr.md)
