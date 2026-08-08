# SplitLargeKMxMatmul Pass

Splits large-K MX matmul into a K=64 `matmul_mx` / `matmul_mx_acc` chain.

## Overview

When `tile.matmul_mx` / `tile.matmul_mx_acc` / `tile.matmul_mx_bias` has a static K with `K > 64` and `K % 64 == 0`, this pass rewrites it into:

1. First chunk: `matmul_mx` (or `matmul_mx_bias`) with operands sliced to `K=64` and scales sliced to `ceil(64/32)=2` groups;
2. Remaining chunks: `matmul_mx_acc` over the remaining K slices.

Dynamic K or K not divisible by 64 is left unchanged. After the rewrite every MX matmul has K=64, so the pass is idempotent.

**Pipeline position**: Immediately after [`InferTileMemorySpace`](13-infer_tile_memory_space.md) and before [`InsertMxScaleAddr`](13-insert_mx_scale_addr.md), so each chunk receives its own scale-address bindings.

**Requirements / produces**: Same property set as `InsertMxScaleAddr` (`SSAForm`, `IncoreTileOps`, `SplitIncoreOrch`, `NormalizedStmtStructure`, `TileMemoryInferred`).

## API

| C++ | Python | Level |
| --- | ------ | ----- |
| `pass::SplitLargeKMxMatmul()` | `passes.split_large_k_mx_matmul()` | Program-level |

```python
from pypto.pypto_core import passes

after = passes.split_large_k_mx_matmul()(passes.infer_tile_memory_space()(program))
```

## Algorithm

For each MX matmul assign in an InCore-variant function body:

1. Read lhs `[M,K]`; skip if K is dynamic, `K <= 64`, or `K % 64 != 0`.
2. For `k0 = 0, 64, …`, emit `tile.slice` on lhs/rhs and lhs_scale/rhs_scale (scale offset `g0 = k0/32`).
3. Emit `matmul_mx` / `matmul_mx_bias` for `ci==0`, then `matmul_mx_acc` for later chunks.
4. Keep the original result SSA name on the final chunk so downstream uses stay valid.

## Related

- Implementation: `src/ir/transforms/split_large_k_mx_matmul_pass.cpp`
- Packed quant is K=64 only: [`ExpandMxPackedQuant`](12-expand_mx_packed_quant.md)
- Downstream binding: [`InsertMxScaleAddr`](13-insert_mx_scale_addr.md)
