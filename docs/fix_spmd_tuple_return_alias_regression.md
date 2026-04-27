# SPMD tuple return alias 报错复盘（fix-spmd）

## 背景

在运行 `qwen3_32b_decode_scope1_spmd_4.py` 时，编译阶段出现如下报错：

`PartialCodegenError: Internal error: tuple element index 0 out of range for kv_proj (has 0 Out/InOut params)`

错误位置落在 `src/codegen/orchestration/orchestration_codegen.cpp` 的 `GenerateTupleReturnAliases()`。

## 错误描述

`GenerateTupleReturnAliases()` 在为 tuple 返回值建立 alias 映射时，使用了 `callee->param_directions_`（经 `GetEffectiveDirections(callee)`）来计算输出参数下标 `out_indices`。

这在理想情况下可用，但在 SPMD/outline/normalize 后，调用点的方向信息可能已经被重写（例如 `output_existing`），此时：

- `call` 侧方向：包含 `ArgDirection::OutputExisting`，语义上是“输出到已存在缓冲区”；
- `callee` 侧方向：仍可能看起来没有 `Out/InOut`。

结果就是 `out_indices` 被错误算成空，后续按 tuple 元素索引访问时触发越界检查并报错。

## 错因分析

原方案里，中间存在以下问题叠加：

1. 调用点在 `DeriveCallDirections` 后被自动标注为 `output_existing`（即写入已有缓冲区）；
2. 但在 outline/normalize 后，`callee->param_directions_` 不能稳定反映这类 call-site 的最终方向；
3. `GenerateTupleReturnAliases()` 当时只信 `callee->param_directions_`，于是把本应属于输出的参数漏掉；
4. 该样例里 `kv_proj` 又正好走到 tuple 返回 alias 映射路径，最终出现“按 tuple 第 0 个输出取索引，但输出列表为空”的报错。

也就是说，**这个样例并不是算子数学逻辑错了，而是“调用点方向信息”和“callee 方向信息”在该重写路径下出现偏差，触发了 codegen 的输出参数识别缺陷**。  
后续我们把 K/V 合并为单个 `kv_proj` SPMD 块后，前端结构更干净，但真正根因仍是 codegen 里对方向来源的选择错误；因此最终修复落在 `GenerateTupleReturnAliases()`。

## 代码执行证据链（为什么会触发）

以失败样例 `Qwen3Scope1_20260425_140548` 为例，编译产物可以直接看到这条链路：

1. `kv_proj` 是 tuple 返回函数，且存在 tuple 元素访问：
   - `passes_dump/30_after_Simplify.py` 中 `def kv_proj(...) -> tuple[...]`
   - 同文件中 `k_proj__rv_v4 = ret__tmp_v0[0]`、`v_proj__rv_v4 = ret__tmp_v0[1]`
2. 外层调用同样走 tuple 返回路径：
   - `ret__tmp_v0 = self.kv_proj(...)`
   - 后续 `ret__tmp_v0[0]` / `ret__tmp_v0[1]`
3. `kv_proj` 调用点已带 `arg_directions=[output_existing, output_existing, ...]`，但 `kv_proj` callee 签名参数是普通 `pl.Tensor[...]`（不是 `pl.Out[...]`）。
4. codegen 报错文本与 `GenerateTupleReturnAliases()` 的边界检查完全一致：
   - 报错：`tuple element index 0 out of range for kv_proj (has 0 Out/InOut params)`
   - 代码位置：`orchestration_codegen.cpp` 中 `GenerateTupleReturnAliases()` 的 `elem.index < out_indices.size()` 检查。

这说明问题不是“没有方向信息”，而是“tuple alias 阶段选错了方向来源”。

## 源码判定：什么时候会走 tuple 返回路径

在 orchestration codegen 中，tuple 返回路径不是“所有 Call 都走”，而是满足以下条件才会触发：

0. **Call 本身被静态类型系统判定为 tuple 返回（多返回）**  
   判定标准不是变量名，而是 `call->GetType()`。当被调用函数签名为多返回（例如 `-> tuple[T1, T2, ...]`）时，`call->GetType()` 才会是 `TupleType`。  
   代码中对应判断为：`if (As<TupleType>(call->GetType())) { ... }`。

1. **Call 的返回类型是 tuple（即多返回）**  
   在 `orchestration_analysis.cpp` 的 `OrchestrationInfoCollector::VisitStmt_(AssignStmt)` 中，只有当：
   - 右值是非 builtin 的函数调用；
   - 且 `call->GetType()` 是 `TupleType`；  
   才会给该 Call 记录 `call_to_tuple_key`。

2. **后续存在 tuple 元素访问（TupleGetItem）**  
   同一 collector 在遇到 `ret[i]`（`TupleGetItemExpr`）时，会把 `{index, var}` 记录到 `call_tuple_elements`。

3. **codegen 阶段两张表都命中**  
   `GenerateTupleReturnAliases()` 里必须同时找到：
   - `call_to_tuple_key_[call]`
   - `tuple_var_to_elements_[key]`  
   否则直接 `return`，不会执行 tuple alias 映射。

因此，只有“**tuple-return call + tuple element uses**”这一组合才会进入该路径。

## 为什么旧方案下 KV 拆开能过、合并过不了

旧方案下这两种写法走到的 codegen 风险面不同：

1. **KV 拆开时（K/V 各自单返回）**  
   常见形态是 `k_proj`、`v_proj` 分别提交，不经过 `ret__tmp_v0[0]/[1]` 这种 tuple 元素映射链路。  
   即使 wrapper 层参数显示为普通 `pl.Tensor[...]`，调用点仍可通过 `arg_directions=output_existing` 正常写回，且 orchestration 侧常直接表现为 `add_output(...)` 提交，通常不会触发 tuple alias 的越界检查。

2. **KV 合并时（单个 kv_proj 返回 tuple）**  
   会出现：
   - `kv_proj(...) -> tuple[k_proj, v_proj]`
   - `ret__tmp_v0[0]` / `ret__tmp_v0[1]`  
   从而进入 `GenerateTupleReturnAliases()`。旧实现仅依据 `callee->param_directions_` 推导输出位，未优先读取 call-site `arg_directions`，当两者不一致时就可能把输出位漏掉，最终报：
   `tuple element index 0 out of range for kv_proj (has 0 Out/InOut params)`。

总结：旧方案“拆开能过”不是因为方向信息一定更正确，而是多数情况下绕开了 tuple alias 的脆弱路径；“合并过不了”是因为恰好触发了该路径并暴露了方向来源选择错误。

## 为什么之前的修改方案能修复

之前的主修复正好命中上述根因：

1. `GenerateTupleReturnAliases()` 改为优先使用 `call->GetArgDirections()`，直接读取调用点上由 `DeriveCallDirections` 写入的方向信息；
2. 把 `ArgDirection::OutputExisting` 纳入“输出参数”集合，避免被遗漏；
3. 仅在调用点方向缺失时才回退到 `callee->param_directions_`；
4. `InOut` 判断与边界检查也改为优先对齐调用点语义与 `call->args_.size()`。

因此即便 wrapper 层签名里参数显示为普通 `pl.Tensor[...]`，只要调用点方向是 `output_existing`，tuple alias 仍能正确建立，不会再出现 `out_indices` 为空导致的越界。

## 为什么 SPMD 会“多一层”：分层结构详解

这个“多一层”不是 Python 源码里平白多写了一个函数，而是编译 pass 在 SPMD 场景下为了并行切分与调度插入的中间 Group wrapper。  
从 `down_proj_residual` 链路看，非 SPMD 与 SPMD 的函数分层如下。

### A. 非 SPMD（`qwen3_32b_decode.py`）典型分层

1. **用户 DSL 层（CORE_GROUP block）**  
   `name_hint="down_proj_residual"` 对应一段普通 CORE_GROUP 代码块。

2. **Outline 后 Group 层（单层 wrapper）**  
   生成 `def down_proj_residual(...)` 这层 Group 函数。

3. **Kernel 层（最终执行单元）**  
   Group 内直接调用 `down_proj_residual_aic` / `down_proj_residual_aiv`。  
   这一层是非 Group callee（AIC/AIV kernel）。

4. **方向推导可见性**  
   `ComputeGroupEffectiveDirections()` 在扫描 Group body 时，能直接看到第 3 层 kernel 的
   `param_directions_`，因此可把外层首参稳定合并为 Out/OutputExisting。

### B. SPMD（`qwen3_32b_decode_spmd_4.py`）典型分层

1. **用户 DSL 层（SPMD + CORE_GROUP 复合）**  
   外层有 `pl.spmd(4)`，内部再包 `CORE_GROUP` 计算块。

2. **外层 Group（例如 `down_proj_residual`）**  
   对应业务语义函数，参数仍然是这层函数的形式参数。

3. **中间 Group wrapper（例如 `qwen3_decode_incore_14`）**  
   由 SPMD/outline 拆分后引入，承接子任务组织与调度语义。  
   这是“多一层”的关键。

4. **Kernel 层（`qwen3_decode_incore_14_aic/aiv`）**  
   真正执行并写回输出的 AIC/AIV 调用发生在这一层。

5. **方向推导盲区产生位置**  
   现实现里 `ComputeGroupEffectiveDirections()` 的 `InnerCallFinder` 只收集“非 Group” inner call。  
   当外层 Group 的 body 里先看到的是“中间 Group wrapper（第 3 层）”时，这层会被跳过，
   导致外层无法把写回语义继续向上传递，出现 Out 被误并成 In 的风险。

### C. 为什么这会影响 `arg_directions`

`DeriveCallDirections` 依赖 `ResolveCalleeDirections -> ComputeGroupEffectiveDirections` 的结果。  
如果该结果把某参数留在 `ParamDirection::In`，那么调用点就会被写成 `pl.adir.input`，最终在
orchestration 里表现为 `add_input(...)` 而不是 `add_output(...)` / `add_inout(...)`。

因此，“SPMD 多一层”本质是：**多了一个 Group-to-Group 的方向传播环节**。  
当前规则未递归穿透这一环节时，就可能在外层丢失写回语义。

### D. 函数名级别分层对照表（`down_proj_residual` 链路）

| 层级 | 非 SPMD (`qwen3_32b_decode.py`) | SPMD (`qwen3_32b_decode_spmd_4.py`) | 方向语义风险 |
|---|---|---|---|
| L0 用户 DSL | `name_hint="down_proj_residual"` 的 CORE_GROUP 代码块 | `pl.spmd(4)` 内 `name_hint="down_proj_residual"` 代码块 | 无 |
| L1 外层 Group | `down_proj_residual(...)` | `down_proj_residual(...)` | 需要从内层汇总参数方向 |
| L2 中间封装层 | （通常无额外 Group） | `qwen3_decode_incore_14(...)`（Group wrapper） | **关键：若不穿透 Group，会在这里丢写回语义** |
| L3 Kernel 调用层 | `down_proj_residual_aic/aiv(...)` | `qwen3_decode_incore_14_aic/aiv(...)` | 真正定义 Out 写回语义的位置 |
| L4 Orchestration 参数提交 | `add_output(out)` / `output_existing` 语义保留 | 异常时变成 `add_input(ext_out)` | 结果张量不被写回，出现全 0 |

表中可以看出：SPMD 比非 SPMD 多出的核心不是算子本身，而是 L2 的 Group wrapper。  
当前实现若只在 Group 内扫描到“下一层也是 Group”就停止向下合并，就会把 L3 的真实 Out 语义隔断。

## 这是否能解释“KV 合并后 codegen 不通过”？

可以，而且是直接解释。

1. **KV 拆开（单返回）时**  
   通常不会形成 `ret = kv_proj(...); ret[0]/ret[1]` 这类 tuple 元素访问链路，因而大多不会进入
   `GenerateTupleReturnAliases()` 的敏感路径。

2. **KV 合并（多返回）时**  
   会形成：
   - `ret__tmp_v0 = self.kv_proj(...)`
   - `k_proj = ret__tmp_v0[0]`
   - `v_proj = ret__tmp_v0[1]`  
   该形态必然触发 tuple alias 生成逻辑。

3. **旧实现失败点**  
   旧版 `GenerateTupleReturnAliases()` 仅依赖 `callee->param_directions_` 计算输出位，未优先读取
   call-site `arg_directions`（其中包含 `output_existing`）。当两者在重写后不一致时，输出位会被漏掉，
   最终出现：
   `tuple element index 0 out of range for kv_proj (has 0 Out/InOut params)`。

结论：**“KV 合并后 codegen 不通过”与 tuple alias 路径中的方向来源选择错误完全一致。**

## 与“最终输出全 0”问题的关系（避免混淆）

两者同属“方向语义传递问题”，但属于不同阶段、不同触发链路：

1. **KV 合并 codegen 报错（编译期失败）**  
   - 触发点：`GenerateTupleReturnAliases()`  
   - 条件：tuple-return call + tuple element uses  
   - 现象：编译期越界检查失败，直接报错终止。

2. **`down_proj_residual` 全 0（运行期数值错误）**  
   - 触发点：`DeriveCallDirections/ComputeGroupEffectiveDirections` 在 Group-to-Group 场景未把 Out 语义向上传递  
   - 现象：`arg_directions` 被误推为 `input`，orchestration 里 `add_input(ext_out)`，导致结果缓冲区未被写回。

因此，这两次问题可以统一归为“方向信息传播链不完整”，但并非同一个函数里的同一个 bug。
