# SPMD `gm_pipe_buffer` Overlap Analysis and Fix Rationale

## Background

The following down-projection pattern uses `pl.spmd(...)` and mixed-kernel
matmul/matmul_acc in one orchestration scope:

```python
# Stage 7 & 8: Down projection + final residual writeback.
for db0 in pl.spmd((HIDDEN // DOWN_N_CHUNK) // 2, name_hint="down_proj_residual_spmd"):
    db = db0 * 2
    for di in pl.range(db, db + 2):
        d0 = di * DOWN_N_CHUNK
        mlp_chunk_0 = mlp_tile[:, 0 : DOWN_K_CHUNK]
        w_down_chunk_0 = w_down[0 : DOWN_K_CHUNK, d0 : d0 + DOWN_N_CHUNK]
        resid1_tile_chunk = resid1_tile[:, d0 : d0 + DOWN_N_CHUNK]
        down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0, out_dtype=pl.FP32)
        mlp_chunk_1 = mlp_tile[:, DOWN_K_CHUNK : 2 * DOWN_K_CHUNK]
        w_down_chunk_1 = w_down[DOWN_K_CHUNK : 2 * DOWN_K_CHUNK, d0 : d0 + DOWN_N_CHUNK]
        down_acc = pl.matmul_acc(down_acc, mlp_chunk_1, w_down_chunk_1)
        for ob in pl.pipeline(2, INTERMEDIATE // DOWN_K_CHUNK, stage=2):
            o0 = ob * DOWN_K_CHUNK
            down_mlp_chunk = mlp_tile[:, o0 : o0 + DOWN_K_CHUNK]
            w_down_chunk = w_down[o0 : o0 + DOWN_K_CHUNK, d0 : d0 + DOWN_N_CHUNK]
            down_acc = pl.matmul_acc(down_acc, down_mlp_chunk, w_down_chunk)
        out_chunk = pl.add(down_acc, resid1_tile_chunk)
        out = pl.assemble(out, pl.cast(out_chunk, target_type=pl.BF16), [0, d0])
```

For this pattern, codegen injects and uses `__gm_pipe_buffer`. The failure mode
appears only in SPMD form, while equivalent non-SPMD form can pass.

## Why non-SPMD can pass while SPMD fails

### Key runtime behavior from Simpler source (with direct code excerpts)

1. **Allocation path: `alloc_tensors` enters runtime and performs real allocation**

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`:

```cpp
static inline TaskOutputTensors alloc_tensors(const Arg &args) {
    PTO2Runtime *rt = pto2_current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->alloc_tensors(rt, args);
}
```

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`:

```cpp
static TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const Arg &args) {
    return pto2_alloc_tensors(&rt->orchestrator, args);
}
```

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`:

```cpp
TaskOutputTensors pto2_alloc_tensors(PTO2OrchestratorState *orch, const Arg &args) {
    // ...
    PTO2OutputLayout layout = pto2_calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!pto2_prepare_task(orch, args, layout.total_output_size, 0, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    // ...
    payload.init(args, outputs, prepared.alloc_result, layout, false);
    // ...
}
```

This shows `alloc_tensors` is not a frontend-only abstraction: it reaches runtime
allocation code and binds allocation results into task payload.

2. **SPMD dispatch is one task split into many logical blocks**

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp`:

```cpp
void SchedulerContext::build_payload(
    PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot,
    PTO2DeferredCompletionIngressBuffer *deferred_ingress
) {
    auto &payload = *slot_state.payload;
    int n = 0;
    for (int32_t i = 0; i < payload.tensor_count; i++) {
        dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
    }
    // ...
    dispatch_payload.local_context.block_idx = slot_state.next_block_idx;
    dispatch_payload.local_context.block_num = slot_state.logical_block_num;
    // ...
}
```

Tensor argument addresses are reused for the task, while each dispatch updates
`block_idx/block_num`.

3. **Kernel-side block identity is read from per-dispatch local context**

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h`:

```cpp
struct LocalContext {
    int32_t block_idx;  // Logical block index within the task [0, block_num)
    int32_t block_num;  // How many logical blocks this task requires.
    // ...
};

static __aicore__ inline int32_t get_block_idx(__gm__ int64_t *args) {
    __gm__ LocalContext *ctx =
        reinterpret_cast<__gm__ LocalContext *>(static_cast<uint64_t>(args[SPMD_LOCAL_CONTEXT_INDEX]));
    return ctx->block_idx;
}
```

Different logical blocks therefore share one task-level argument layout but run
with different `block_idx` values from local context.

### Consequence

- In non-SPMD form (`block_num=1`), only one logical block uses the workspace at
  a time, so overlap does not happen.
- In SPMD form (`block_num>1`), if `__gm_pipe_buffer` is not both:
  - allocated with enough total capacity for all logical blocks, and
  - indexed to a per-block slice in kernel wrapper,
  then multiple logical blocks will read/write overlapping workspace regions.

## What this change set fixes

1. **Allocation-side scaling (host/orchestration side)**  
   For injected gm pipe tensor.create, allocation shape is scaled by launch
   `core_num` during orchestration codegen (instead of changing IR visible shape).

2. **Wrapper-side sharding (kernel arg unpacking side)**  
   For SPMD kernels, generated wrapper maps `__gm_pipe_buffer` pointer to:
   `base + block_idx * elems_per_block`.
   This guarantees each logical block gets a disjoint workspace slice.

3. **Sample alignment with block-level sharding contract**  
   The added ST sample uses SPMD by block index and does **not** combine
   `UP_DOWN` split for this gm pipe case, because current sharding contract is
   block-level (not lane-level subblock sharding).

## Evidence from the added sample

Sample test:
`tests/st/runtime/test_spmd.py::TestSPMDOperations::test_spmd_gm_pipe_buffer_golden`

- **Before latest sample correction (historical run):**
  - result: FAIL
  - symptom: golden mismatch (`1024/2048` mismatched elements)
- **After latest commit and reinstall (current run):**
  - result: PASS
  - command mode: `task-submit` + pytest ST

This delta is consistent with the overlap hypothesis and confirms the necessity
of:
- per-block capacity guarantee at allocation time, and
- per-block pointer sharding at kernel entry,
- plus sample-side execution pattern aligned to current sharding semantics.

