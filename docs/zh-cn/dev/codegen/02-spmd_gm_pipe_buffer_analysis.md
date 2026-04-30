# SPMD `gm_pipe_buffer` 重叠问题分析与修改必要性

## 背景

下面这段 down projection 代码使用了 `pl.spmd(...)`，并在同一编排作用域内执行 mixed-kernel 的
matmul/matmul_acc：

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

该模式下，代码生成会注入并使用 `__gm_pipe_buffer`。问题只在 SPMD 形态下暴露，而改成非 SPMD
形态通常可通过。

## 为什么非 SPMD 能过，而 SPMD 会失败

### 来自 Simpler/runtime 源码的关键行为（含完整依赖代码）

1. **内存分配链路：`alloc_tensors` 进入 runtime 并执行真实分配**

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/orchestration/pto_orchestration_api.h`：

```cpp
static inline TaskOutputTensors alloc_tensors(const Arg &args) {
    PTO2Runtime *rt = pto2_current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->alloc_tensors(rt, args);
}
```

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2.cpp`：

```cpp
static TaskOutputTensors alloc_tensors_impl(PTO2Runtime *rt, const Arg &args) {
    return pto2_alloc_tensors(&rt->orchestrator, args);
}
```

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_orchestrator.cpp`：

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

以上代码说明 `alloc_tensors` 不是语法糖，而是进入 runtime 的真实分配路径，并将分配结果绑定到 task payload 中。

2. **SPMD 是“一个任务 + 多个逻辑 block 分发”**

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp`：

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

这段代码表明：同一 task 的 tensor 参数地址会被复用，而每次 dispatch 只更新 `block_idx/block_num`。

3. **kernel 侧 block 身份来自 per-dispatch local context**

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h`：

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

也就是说，不同逻辑 block 共享同一 task 的参数布局，但通过 local context 读取不同 `block_idx` 执行各自子任务。

### 直接后果

- 非 SPMD（`block_num=1`）时，同一时刻只有一个逻辑 block 使用工作区，不会重叠。
- SPMD（`block_num>1`）时，如果 `__gm_pipe_buffer` 没有同时满足：
  - 分配容量按逻辑 block 总数扩容；
  - kernel 入口按 block 切片索引；
  就会发生多个 block 读写同一片工作区，导致结果错误。

## 本次修改如何解决

1. **分配侧扩容（host/orchestration）**  
   对注入的 gm pipe `tensor.create`，在 orchestration codegen 阶段按 `core_num` 扩展分配形状
   （而不是改 IR 可见 shape）。

2. **wrapper 侧切片（kernel 参数解包）**  
   在 SPMD wrapper 中，将 `__gm_pipe_buffer` 指针映射为：  
   `base + block_idx * elems_per_block`，确保每个逻辑 block 用独立切片。

3. **样例与当前分片语义对齐**  
   新增 ST 样例中，gm pipe 场景使用 block 级 SPMD，不再叠加 `UP_DOWN` split。当前语义是
   block 级分片（不是 lane/subblock 级分片）。

## 基于新增样例的运行证据

样例用例：
`tests/st/runtime/test_spmd.py::TestSPMDOperations::test_spmd_gm_pipe_buffer_golden`

- **未做最新样例修正时（历史运行）：**
  - 结果：FAIL
  - 现象：golden mismatch（`1024/2048` 元素不一致）
- **最新提交并重装后（当前运行）：**
  - 结果：PASS
  - 执行方式：`task-submit` + pytest ST

这个差异与“SPMD 下 gm_pipe 工作区重叠”假设一致，支持本次修改的必要性：
- 分配侧保证每 block 容量；
- wrapper 侧按 block 切片；
- 样例执行形态与当前分片契约一致。

