# LegalizeMixedMxScaleViaGm Pass

把 mixed kernel 上经 V2C 传输的 MX E8M0 A-scale 改写为 GM `store` + `MX_A_ZZ` `load`。紧接 [`ExpandMxPackedQuant`](12-expand_mx_packed_quant.md) 运行，复用其已产出的 packed ZZ 布局，不做 FP32 伪装或 pad。

## 概述

A5 上 V2C 传非均匀 E8M0 scale 数值不可靠（见[设计文档](../design/a5-mixed-mx-scale-transport.md)）。本 Program pass：

1. 在 AIV 中匹配 `tile.tpush_to_aic` 且 tile dtype 为 `FP8E8M0`
2. 在 AIC 中匹配对应 pipe id 的 `tile.tpop_from_aiv`（E8M0）
3. 注入共享 GM 参数 `__mx_a_scale_gm`（编排层 `tensor.create`）
4. AIV：`tpush(scale)` → `tile.store`；AIC：`tpop` → `tensor.view(MX_A_ZZ)` + `tile.load(Mat)`
5. FP8 data 的 V2C 不变；已无 E8M0 V2C 时幂等跳过

**前置条件 / 产生属性**：无（与 `ExpandMxPackedQuant` 一样为空属性集）。

## 运行时机

`tile_pto_passes` 中紧接 `expand_mx_packed_quant`，在 `lower_composite_ops` 之前。

## API

```python
from pypto.pypto_core import passes
passes.legalize_mixed_mx_scale_via_gm()
```

- 实现：`src/ir/transforms/legalize_mixed_mx_scale_via_gm_pass.cpp`
- 声明：`include/pypto/ir/transforms/passes.h`

## 另请参阅

- [A5 Mixed MX Scale 传输设计](../design/a5-mixed-mx-scale-transport.md)
- [`ExpandMxPackedQuant`](12-expand_mx_packed_quant.md)
- [`ExpandMixedKernel`](23-expand_mixed_kernel.md)（data 仍可走 V2C）
