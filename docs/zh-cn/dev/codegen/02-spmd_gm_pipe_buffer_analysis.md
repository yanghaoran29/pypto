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

### 来自 Simpler/runtime 源码的关键行为

1. **每个编排作用域只做一次输出/工作区分配**  
   在生成的 orchestrator 中，`alloc_tensors(...)` 调用一次，返回的 tensor 在任务提交中复用。  
   示例文件：
   `build_output/gm_pipe_golden_saved/spmd_gm_pipe_buffer_128x16/orchestration/orchestrator.cpp`
   可见：
   - 一次 `alloc_tensors(gm_pipe_buffer_0_ci)`
   - 一次 `params_t0.add_output(gm_pipe_buffer_0)`
   - 一次 `params_t0.launch_spec.set_block_num(2)` 提交。

2. **SPMD 是“一个任务 + 多个逻辑 block 分发”**  
   调度器构建 payload 时，tensor 参数地址保持不变，只修改本次分发的 block 身份：
   - `dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);`
   - `dispatch_payload.local_context.block_idx = slot_state.next_block_idx;`
   见 `runtime/.../scheduler/scheduler_dispatch.cpp`。

3. **kernel 侧的 block_idx 来自每次分发的 local context**  
   `get_block_idx(args)` 从 `LocalContext.block_idx` 读取，不同 block 值不同，证明同一任务内存在多 block 分发。  
   见 `runtime/.../common/intrinsic.h`。

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

