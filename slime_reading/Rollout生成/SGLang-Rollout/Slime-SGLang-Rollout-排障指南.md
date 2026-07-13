---
title: "SGLang-Rollout · 排障指南"
type: troubleshooting
framework: slime
topic: "SGLang-Rollout"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# SGLang-Rollout · 排障指南

## 你为什么要读

这篇按症状排障。先判断问题落在哪个边界：rollout function、custom generate、Sample 账本、dynamic filter、partial abort、metrics，还是 eval。

## 症状一：不知道该改 `rollout-function-path` 还是 `custom-generate-function-path`

判断：如果只想改变单条 sample 如何生成 response，改 `custom-generate-function-path`。如果要替换补样、filter、abort、返回包装整套 orchestration，才改 `rollout-function-path`。

```python
# 定位骨架（据 `slime/utils/arguments.py` L328-L339 删节）：
            parser.add_argument(
                "--rollout-function-path",
                type=str,
                default="slime.rollout.sglang_rollout.generate_rollout",
                help=(
                    "Path to the rollout generation function."
                    "You should use this model to create your own custom rollout function, "
                    "and then set this to the path of your custom rollout function. "
                    "The signature of the function should be "
                    "`def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput`"
                    "and within the output sample, you should at least set `tokens`, `response_length`, `reward` "
                    "and `status`."
                ),
            )
```

```python
# 定位骨架（据 `slime/utils/arguments.py` L473-L480 删节）：
            parser.add_argument(
                "--custom-generate-function-path",
                type=str,
                default=None,
                help=(
                    "Only substitue the `def generate(args, sample, sampling_params)` function within the example rollout function. "
                    "This should be useful if you need to implement some special rollout logic, e.g. multi-turn, function calling."
                ),
            )
```

处理：

- agent/tool calling：优先 `custom-generate-function-path`。
- SFT、fully-async、磁盘 replay：使用专用 `rollout-function-path`。
- eval 某个 dataset 特殊生成：使用 dataset 级 `custom_generate_function_path`，它最终写入 `sample.generate_function_path`。

验证：用 `plugin_generate_contracts` 测 custom generate，用 `plugin_rollout_contracts` 测整段 rollout function。

## 症状二：custom generate 被配置了，但没有走预期函数

判断：sample 级 `generate_function_path` 优先于全局 `args.custom_generate_function_path`。eval dataset 注入的 path 会覆盖全局 custom generate。

```python
# 来源：slime/rollout/sglang_rollout.py L249-L261
        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)
```

处理：

- 检查 sample 上是否已有 `generate_function_path`。
- 检查 custom generate 是否是 async 函数。
- 如果函数需要知道 eval/train，签名里显式加入 `evaluation`。

验证：`test_generate_and_rm_prefers_per_sample_generate_function` 会验证 sample 级 override 优先级。

## 症状三：custom generate 返回后 RM 或训练报字段错

判断：custom generate 返回的是 `Sample` 或 `list[Sample]`，但字段必须满足训练契约。至少要有 token/response 长度、status、reward 或能让后续 RM 补 reward。

```python
# 来源：slime/tests/plugin_contracts/test_plugin_generate_contracts.py L71-L87
async def custom_generate(args, sample: Sample, sampling_params: dict):
    sample.tokens = [11, 12, 13]
    sample.response = "generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.25
    sample.status = Sample.Status.COMPLETED
    return sample


async def custom_generate_with_evaluation(args, sample: Sample, sampling_params: dict, evaluation: bool = False):
    sample.tokens = [21, 22]
    sample.response = "eval-generated" if evaluation else "train-generated"
    sample.response_length = len(sample.tokens)
    sample.reward = 0.5 if evaluation else 0.75
    sample.status = Sample.Status.COMPLETED
    sample.metadata["evaluation"] = evaluation
    return sample
```

处理：

- 模型生成 token 建议走 `sample.append_response_tokens(args, tokens=tokens, trainable=True, log_probs=log_probs, meta_info=meta_info)`。
- 工具或环境 token 走 `trainable=False`，不要手写 `loss_mask` 和 offsets。
- fan-out 返回 `list[Sample]` 时，每个 sibling 都要满足字段契约。

验证：跑 `test_rollout_metrics.py` 检查 append 不变量，跑 `test_plugin_generate_contracts.py` 检查返回形状。

## 症状四：rollout 卡住，一直补样不结束

判断：主循环只在 `len(data) == rollout_batch_size` 时结束。dynamic filter drop 的 group 不进入 `data`，只降低水位并继续补样。

```python
# 定位骨架（据 `slime/rollout/sglang_rollout.py` L408-L439 删节）：
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)
```

处理：

- 降低 filter 严格度，或检查 reward 是否总是相同导致 filter 全 drop。
- 检查 DataSource 是否能持续提供 group。
- 确认参数归一化确实执行过：`over_sampling_batch_size=None` 会回填为 `rollout_batch_size`，随后 validator 强制前者不小于后者。绕过统一参数入口的手工 Namespace 才可能漏掉这道门禁。

验证：看 dynamic filter drop metrics 和日志中的 first/finish rollout sample；drop 原因应能解释为什么有效 group 不增长。

## 症状五：top-p 指标为空或计算异常

判断：只有 `rollout_top_p != 1.0` 时，`GenerateState` 才请求 top-p token replay。之后 offsets 必须与 `response_length` 对齐。

```python
# 来源：slime/rollout/sglang_rollout.py L107-L108
        if args.rollout_top_p != 1.0:
            self.sampling_params["custom_params"] = {"return_top_p_token_ids": True}
```

```python
# 定位骨架（据 `slime/ray/rollout.py` L1427-L1454 删节）：
def _compute_top_p_kept_vocab_metrics(args, all_samples: list[Sample]):
    total_kept = 0
    total_tokens = 0
    for sample in all_samples:
        offsets = sample.rollout_top_p_token_offsets
        if offsets is None or sample.response_length == 0:
            continue
        offsets = torch.as_tensor(offsets, dtype=torch.int64)
        if offsets.numel() == 0:
            continue
        assert (
            offsets.numel() == sample.response_length + 1
        ), f"top-p token offsets length {offsets.numel()} != response length + 1 {sample.response_length + 1}"
        if sample.remove_sample:
            continue
        if sample.loss_mask is None:
            total_kept += int(offsets[-1] - offsets[0])
            total_tokens += sample.response_length
            continue
```

处理：

- `rollout_top_p=1.0` 时 metric 为空是正常现象。
- custom multi-turn generate 必须用 `append_response_tokens` 合并 top-p offsets。
- non-trainable tool token 应让 offsets padding，而不是伪造 top-p token ids。

验证：`test_append_response_tokens_merges_top_p_tensors`、`test_top_p_kept_vocab_metric_uses_loss_mask`、`test_append_response_tokens_pads_top_p_for_non_trainable_tokens`。

## 症状六：routing replay 字段 shape 错

判断：默认 generate 只有在 `use_rollout_routing_replay` 开启时请求 routed experts。Sample 解码时需要 `args.num_layers` 和 `args.moe_router_topk`。

```python
# 来源：slime/rollout/sglang_rollout.py L174-L181
    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True
```

```python
# 来源：slime/utils/types.py L352-L360
        routed_experts = decode_int32_meta_array(meta_info, "routed_experts")
        if routed_experts is not None:
            if args is None:
                raise ValueError("args is required to decode routed experts metadata.")
            self.rollout_routed_experts = routed_experts.reshape(
                len(self.tokens) - 1,
                args.num_layers,
                args.moe_router_topk,
            )
```

处理：

- 确认 generate payload 带 `return_routed_experts=True`。
- 确认 args 里 layer 数和 topk 与模型一致。
- 检查首维按整条 `tokens` 的 next-token 长度 `len(tokens)-1` 对齐，而不是按 `response_length`；多段生成要传当前完整快照，因为 `_apply_meta_info` 会覆盖旧值而不增量 merge。
- custom generate 不要手写不同 shape 的 routed experts。

验证：`test_append_response_tokens_decodes_routed_experts`。

## 症状七：partial rollout 续写后旧 token 仍参与训练

判断：只有同时开启 `partial_rollout` 和 `mask_offpolicy_in_partial_rollout`，已有 response 的 loss mask 才会被置 0。

```python
# 来源：slime/rollout/sglang_rollout.py L230-L239
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample
```

处理：

- 检查两个参数是否都开启。
- 检查回灌样本的 `response_length` 是否大于 0。
- 确认 custom generate 后续 append 的新 token 使用 `trainable=True` 和 logprobs。

验证：断点查看进入 `generate_and_rm` 前后的 `sample.loss_mask`。

## 症状八：abort 后没有样本回到 buffer

判断：`abort` 只有在 partial 模式下收集 pending task 结果；同步入口收到 `aborted_samples` 后才调用 `data_source.add_samples`。

```python
# 定位骨架（据 `slime/rollout/sglang_rollout.py` L336-L372 删节）：
async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    await abort_servers_until_idle(urls)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)
```

处理：

- 开启 `partial_rollout`。
- 使用支持 `add_samples` 的 DataSource。
- 检查有效 batch 满时是否真的还有 pending task。
- 注意没有 response 的 sample 不会写 `start_rollout_id`，但 group 仍可能被 append。
- 如果 custom generate 会 fan-out，检查 `group` 的 leaf 是否已经变成 `list[Sample]`；当前 abort 循环直接访问 `sample.response`，不兼容这层额外嵌套。

验证：看 `Collected <count> partial samples into the data buffer` 日志；下一轮 DataSource 应先弹出 buffer。

## 症状九：eval 路径和训练路径行为不一致

判断：这是设计。eval 不走训练水位和 dynamic filter；它展开固定 eval dataset，并复用 `generate_and_rm`。

```python
# 定位骨架（据 `slime/rollout/sglang_rollout.py` L561-L582 删节）：
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )
```

处理：

- 不要期待 eval 产生 `rollout_batch_size` 组。
- eval dataset 的 custom generate path 会写入 sample 级 override。
- eval 不支持 group RM。

验证：查看 eval 输出字典是否包含 `rewards`、`truncated`、`samples`。

## 症状十：fan-out 单测通过，但 group RM 或 partial abort 崩溃

判断：现有 contract test 只直接调用 `generate_and_rm(..., group_rm=True)`，证明它能原样返回 `list[Sample]`；它没有把结果继续送进 `generate_and_rm_group`。真实组路径中，一个输入 sample fan-out 后会让 group 变成 `list[list[Sample]]`，而组级 reward 赋值和 partial abort 仍按内层元素是 `Sample` 编写。

处理：

- fan-out 与 `group_rm=True` 组合前，给 `generate_and_rm_group` 写端到端嵌套测试，不要只复用现有单层 contract 名称作结论。
- fan-out 与 `partial_rollout=True` 组合前，覆盖 abort 后 `response/metadata/start_rollout_id` 回灌路径。
- 若下游不支持嵌套，选择在 custom generate 内部聚合成单个 Sample，或同时替换整个 rollout function 并显式定义 flatten/rollout-id 语义。

预期：普通 group 的 leaf 始终是 `Sample`；若允许更深嵌套，则 RM、filter、abort、排序、hook 和 RolloutManager compact 校验都必须对同一 shape 达成一致。

## 症状十一：设置了 DP 规模，却发现 `dp_rank_context` 没有定向请求

判断：该 context 只做进程内最小计数选择并写 `GenerateState.dp_rank`。默认 `generate()` 仍访问 `sglang_router_ip:sglang_router_port/generate`，payload/header 没有 DP rank 字段。

处理：

- 若依赖 router 自身负载均衡，就看 router worker 分发日志，不要把 `state.dp_rank` 当作网络路由证据。
- 若 custom generate 主动读取 `GenerateState.dp_rank` 并定向 engine，要在扩展实现里记录最终 URL/worker id。

预期：默认路径的路由所有权属于 router；只有扩展代码显式消费 `state.dp_rank` 时，这个值才参与实际目标选择。

## 症状十二：sample filter 后样本还在 group 或 advantage 统计里

判断：这是两层过滤语义。dynamic sampling filter 决定整个 group 是否计入有效 batch；`rollout_sample_filter_path` 在 batch 已凑满后运行，正式契约是设置叶子 Sample 的 `remove_sample`。该标记会在训练数据转换时把 loss mask 置零，但不会把对象从 group 删除，也不决定更早的 advantage normalization。

处理：让 dynamic sampling filter 承担“是否计入有效 batch”的筛选；让 sample filter 只原地设置 `remove_sample`，不要修改 group 数量或嵌套形状。若业务要求过滤也影响 advantage normalization，应改更早的 reward/filter 流程，而不是依赖这个最终 loss mask hook。

预期：交给 RolloutManager 的 group 数仍等于 `rollout_batch_size`；被标记样本仍存在，但其训练 `loss_mask` 最终全 0。

## 症状十三：HTTP 有 response 文本，但训练侧 response_length 为 0

判断：默认路径只从 `meta_info.output_token_logprobs` 提取 token ids 和 logprobs。该字段缺失时两者都变成空列表，文本仍会追加，finish reason 也仍可能把状态设为完成。

处理：检查 SGLang 响应是否真的包含 `output_token_logprobs`，以及请求是否保持 `return_logprob=True`；不要只以 HTTP 200 或非空 `text` 判定训练样本有效。

预期：每个可训练 response token 都有对应 rollout logprob，`response_length` 与 response 侧数组一致。

## 症状十四：明明有完成且通过 filter 的 group，却没进入训练也没回 buffer

判断：`asyncio.wait(..., FIRST_COMPLETED)` 返回的 `done` 可能同时包含多个 task。循环先把每个完成 group 放进 `all_data`，但只有 `len(data) < rollout_batch_size` 时才追加到训练数据；同一批较早的 keep 已填满目标后，后续 keep group 会成为“完成但未使用”结果。abort 只处理仍 pending 的 task，不会回灌这些已完成 group。

处理：通过 `rollout_all_samples_process_path` 记录或显式处理 `all_data` 与 `data` 的差集；若必须零浪费复用，需在自定义 rollout/DataSource 协议中定义完成 group 的回灌与去重规则。

预期：能分别解释 dynamic-filter drop、目标已满的 unused completed group、以及 pending abort partial group 三类去向，不再把它们都归为“被 abort”。

## 测试矩阵

| 问题 | 推荐测试 |
|------|----------|
| Sample top-p/loss mask/routing 字段 | `python -m pytest slime/tests/test_rollout_metrics.py -q` |
| custom generate 优先级与单层 fan-out 返回 | `python -m pytest slime/tests/plugin_contracts/test_plugin_generate_contracts.py -q` |
| rollout function 签名与返回包装 | `python -m pytest slime/tests/plugin_contracts/test_plugin_rollout_contracts.py -q` |
| DataSource `add_samples/get_samples` 与 hook 路径签名 | `python -m pytest slime/tests/plugin_contracts/test_plugin_path_loading_contracts.py -q` |

当前机器的实际 collection 结果是 plugin contracts 缺 `httpx`，最小 stub 后继续缺 `pylatexenc` 并触发 PyArrow/Torch—NumPy ABI 问题；rollout metrics 还缺 `ray`。这些是环境阻塞，但测试名也不能外推覆盖面：当前 generate contract 没有覆盖 fan-out 进入 group RM、partial abort、dynamic filter 和最终 hook 的组合路径。本轮 5 项 AST/隔离执行检查已覆盖这些静态边界，真实服务仍需完整环境。
