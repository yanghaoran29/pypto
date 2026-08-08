# LegalizeMixedMxScaleViaGm Pass

Rewrite mixed-kernel MX E8M0 A-scale V2C (`tpush`/`tpop`) into GM `store` + `MX_A_ZZ` `load`. Runs immediately after [`ExpandMxPackedQuant`](12-expand_mx_packed_quant.md) and reuses its packed ZZ layout — no FP32 disguise or padding.

See the [design note](../design/a5-mixed-mx-scale-transport.md) (Chinese) for why V2C scale transport was abandoned.

## When it runs

Right after `expand_mx_packed_quant`, before `lower_composite_ops`.

## API

```python
from pypto.pypto_core import passes
passes.legalize_mixed_mx_scale_via_gm()
```
