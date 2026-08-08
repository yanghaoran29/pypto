# A5 Mixed Kernel: MX Scale Transport (V2C Failure → GM)

Chinese design note (full analysis): [zh version](../../../zh/dev/design/a5-mixed-mx-scale-transport.md).

**Decision:** keep FP8 **data** on V2C; force MX **scale** through GM in the packed ZZ layout produced by `ExpandMxPackedQuant`. New pass `LegalizeMixedMxScaleViaGm` runs immediately after that expansion. Do **not** FP32-disguise or pad scale for V2C.
