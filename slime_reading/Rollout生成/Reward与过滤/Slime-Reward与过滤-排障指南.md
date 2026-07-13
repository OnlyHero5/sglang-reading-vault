---
title: "Reward与过滤 · 排障指南"
type: troubleshooting
framework: slime
topic: "Reward与过滤"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# Reward与过滤 · 排障指南

## 读者任务

这篇按症状排障：reward 为什么没写上、`dapo` 为什么要 `reward_key`、`group_rm` 为什么签名变了、dynamic filter 为什么让 rollout 变慢。

## 速查表

| 症状 | 优先看 | 可能原因 | 验证抓手 |
|------|--------|----------|----------|
| 未指定 RM 时报错 | `async_rm` | `rm_type` 和 `custom_rm_path` 都为空 | 是否触发 `NotImplementedError` |
| `dapo` 后 filter 报 tensor 转换错 | `get_reward_value` | reward 是 dict，但未设 `--reward-key score` | 打印 `sample.reward` |
| `--group-rm` 后 custom RM TypeError | `batched_async_rm` | 函数签名仍是 `(args, sample)` | batch 模式应接 `samples` |
| rollout 很慢但 SGLang 不慢 | `generate_rollout_async` | dynamic filter drop 率高，一直补样 | 看 `rollout/dynamic_filter/drop_*` |
| remote RM 卡住或失败 | `remote_rm` | 服务超时、HTTP 错误、返回 dict 未配置 key | 看 retry 日志和返回 JSON |
| eval reward key 不对 | eval output | `eval_reward_key`/`reward_key` 配置错 | 看 eval 返回的 rewards |
| `boxed_math` 明明有答案却总是 0 | `boxed_` 与 scorer 的组合 | 预抽取后 `math` 又要求 `\boxed` | 分别执行前缀和后缀 scorer |
| custom batch RM 后部分 reward 仍是 None | reward 回填 `zip` | 返回长度小于输入长度 | 断言输入输出等长 |
| filter 一直 drop 且永不退出 | 补样 while 循环 | 没有最大 drop/尝试门禁 | 观察 keep rate 并设置外部 watchdog |

## Q1：`math` 与 `dapo` 该选哪个？

| 维度 | `math` | `dapo` |
|------|--------|--------|
| 返回形状 | `0/1` | dict |
| 错题 reward | `0` | `score=-1.0` |
| 答案提取 | boxed answer | 内置分发固定走默认 Minerva `Answer:`；函数级 API 才可传 strict box |
| 长 response | 全文找 boxed | 只看最后 300 字符 |
| 常见配置 | `--rm-type math` | `--rm-type dapo --reward-key score` |

`math` 入口：

```python
# 来源：slime/rollout/rm_hub/math_utils.py L484-L493
def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)
```

`dapo` 入口：

```python
# 定位骨架（据 `slime/rollout/rm_hub/math_dapo_utils.py` L279-L292 删节）：
solution_str = solution_str[-300:]

correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

reward = 1.0 if correct else -1.0
acc = correct

return {
    "score": reward,
    "acc": acc,
    "pred": pred,
}
```

测试明确锁定差异，避免以后误合并：

```python
# 定位骨架（据 `tests/test_rm_math_dapo.py` L205-L229 删节）：
def test_compute_score_correct_returns_dict_with_reward_one():
    out = compute_score(r"\boxed{42}", "42", strict_box_verify=True)
    assert out == {"score": 1.0, "acc": True, "pred": "42"}

def test_compute_score_incorrect_returns_minus_one():
    out = compute_score(r"\boxed{43}", "42", strict_box_verify=True)
    assert out["score"] == -1.0
    assert out["acc"] is False

def test_compute_score_only_uses_last_300_chars():
    sol = r"\boxed{42}" + (" filler" * 60)
    out = compute_score(sol, "42", strict_box_verify=True)
    assert out["score"] == -1.0
```

## Q2：为什么 `dapo` 不设 `--reward-key score` 会出问题？

因为 `dapo` 的 reward 是 dict。filter 和训练侧都要通过 `get_reward_value` 得到标量。

```python
# 来源：slime/utils/types.py L246-L247
def get_reward_value(self, args) -> float:
    return self.reward if not args.reward_key else self.reward[args.reward_key]
```

内置 dynamic filter 会把 reward values 转成 tensor：

```python
# 来源：slime/rollout/filter_hub/dynamic_sampling_filters.py L9-L11
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
```

正确配置：

```bash
--rm-type dapo --reward-key score
```

remote RM 返回 dict 时同理。

若希望内置 `dapo` 走 strict-box，单写 CLI `--rm-type dapo` 不够：`async_rm` 调用 `compute_score_dapo(response, label)` 时没有传 `strict_box_verify=True`。需要 custom RM 包装器显式调用函数级 API。

## Q3：`custom_rm_path` 的单条和 batch 签名怎么区分？

默认单条路径调用 `async_rm(args, sample)`，custom 函数应接 `(args, sample, **kwargs)`。

`group_rm=True` 或 batch custom path 调用 `batched_async_rm(args, samples)`，custom 函数应接 `(args, samples, **kwargs)`。

但优先级有一个容易忽略的断点：只要设置了全局 `args.custom_rm_path`，batch 入口就直接调用它，不再逐条查看 `sample.custom_rm_path`。此外返回 list 必须与 samples 等长；调用侧的 `zip(strict=False)` 不会替你报长度不匹配。

源码分支：

```python
# 定位骨架（据 `slime/rollout/rm_hub/__init__.py` L99-L110 删节）：
async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
```

插件契约测试也按这两个签名检查：

```python
# 来源：tests/plugin_contracts/test_plugin_path_loading_contracts.py L319-L336
def test_custom_rm_path_aligns_with_expected_format():
    path = get_contract_path("CUSTOM_RM_PATH")
    if get_contract_path("GROUP_RM") == "1":
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "samples")
        rewards = asyncio.run(
            batched_async_rm(
                make_args(
                    group_rm=True,
                    custom_rm_path=path or "plugin_contracts.test_plugin_path_loading_contracts.reference_batched_rm",
                ),
                [make_sample(0), make_sample(1)],
            )
        )
        assert isinstance(rewards, list) and len(rewards) == 2
    else:
        fn = load_function(path or "plugin_contracts.test_plugin_path_loading_contracts.reference_single_rm")
        assert tuple(inspect.signature(fn).parameters)[:2] == ("args", "sample")
```

## Q4：`group_rm` 什么时候需要？

需要整组上下文时才打开：

- listwise RM 要比较同一 prompt 的多条 response。
- 自定义远程 RM 函数希望自己把一组 response 编成一次 batch 请求；内置 `remote_rm` 仍是一条 sample 一次 HTTP POST。
- 普通同构 `list[Sample]` 需要统一打分；若 custom generate 已产生嵌套 fan-out，当前 group RM 不能直接组合。

不需要整组上下文时，不要为了“更快”盲目打开。默认 `batched_async_rm` 已经会并发单条 `async_rm`。

eval 路径明确断言 `group_rm` 为 false，所以 batch/listwise RM 不能直接沿默认 eval rollout 复用。

源码入口：来源：slime/rollout/sglang_rollout.py L326-L331

## Q5：dynamic filter 一直 drop，rollout 看起来卡住怎么办？

现象：模型过强或过弱时，同一 prompt 的多条 response reward 完全相同，`check_reward_nonzero_std` 会持续 drop。

filter 判定：

```python
# 来源：slime/rollout/filter_hub/dynamic_sampling_filters.py L9-L15
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
```

drop 后补样：

```python
# 来源：slime/rollout/sglang_rollout.py L429-L433
dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
if not dynamic_filter_output.keep:
    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
    state.remaining_batch_size -= 1
    continue
```

排查顺序：

- 看 `rollout/dynamic_filter/drop_zero_std_*` metrics。
- 临时关闭 `--dynamic-sampling-filter-path` 验证 SGLang 生成速度。
- 提高采样多样性或换 curriculum。
- 调大 `--over-sampling-batch-size` 提高补样吞吐。
- 写自定义 filter，保留一部分 zero-std groups。

若 `n_samples_per_prompt=1`，内置 `torch.std()` 得到 `nan`，也会落入 drop 分支。若 filter 对所有后续 prompt 都拒绝，当前主循环没有最大补样次数，会持续请求新数据；生产运行应监控 keep rate、drop 总数和 wall-clock deadline，而不是只看进度条。

## Q6：`remote_rm` 的请求和重试策略是什么？

payload 只含 prompt、response、label：

```python
# 来源：slime/rollout/rm_hub/__init__.py L34-L45
async def remote_rm(args, sample: Sample, max_retries: int = 10):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session = _get_shared_session()
    for attempt in range(max_retries):
        try:
            async with session.post(args.rm_url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
```

失败会重试，最终失败会 raise：

```python
# 来源：slime/rollout/rm_hub/__init__.py L46-L52
except Exception as e:
    if attempt + 1 >= max_retries:
        logger.warning(f"remote_rm failed after {attempt + 1} attempts: {e}")
        raise
    backoff = min(2**attempt, 30) + random.random()
    logger.info(f"remote_rm: {type(e).__name__}, retrying in {backoff:.1f}s ({attempt + 1}/{max_retries})")
    await asyncio.sleep(backoff)
```

如果服务返回 dict，仍要配置 `--reward-key`。

单次请求 timeout 是 120 秒，最多重试 10 次，且有最高约 30 秒的退避；持续故障的总体等待可能远超 120 秒。共享 `ClientSession` 也没有在本模块暴露显式 close 生命周期。排障时要同时看单次 HTTP timeout、累计重试时间和进程退出时的 session warning。

## Q7：`boxed_math` 这类前缀怎么理解？

`boxed_` 是局部预处理前缀，不是经过统一验证的 scorer 装饰器。源码先从 response 中提取 boxed answer，再把后缀当成 `rm_type`，但不同后缀消费输入的方式并不一致。

```python
# 来源：slime/rollout/rm_hub/__init__.py L69-L71
if rm_type.startswith("boxed_"):
    response = extract_boxed_answer(response) or ""
    rm_type = rm_type[len("boxed_") :]
```

不能把 `boxed_math` 当作正确示例：它先把 `\boxed{42}` 压成 `42`，随后 `math` 的 `grade_answer_verl` 又从输入里寻找 `\boxed`，因此得到 0。`boxed_deepscaler` 通常也因丢失分隔符/box 得到 0；`boxed_remote_rm` 则仍把原始 `sample.response` 放进 HTTP payload，预处理结果没有传给服务。使用任何 `boxed_*` 组合前都应写一个直接行为测试。

## Q8：`deepscaler` 为什么明明答案正确也给 0？

DeepScaler scorer 先要求 response 中有 `</think>` 或 `###Response` 分隔符；没有就直接 0。

```python
# 来源：slime/rollout/rm_hub/deepscaler.py L4-L14
def get_deepscaler_rule_based_reward(response, label):
    if "</think>" in response:
        model_solution = response.split("</think>")[-1]
    elif "###Response" in response:
        model_solution = response.split("###Response")[1]
    else:
        return 0

    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
```

如果你的模型输出没有这些分隔符，用 `math` 或自定义 scorer 更合适。

## Q9：如何验证这块没有改坏？

CPU scorer 单测：

```powershell
Set-Location 'F:\源码阅读\slime'
python -m pytest tests/test_rm_math_dapo.py -q
```

插件契约：

```powershell
Set-Location 'F:\源码阅读\slime'
python -m pytest tests/plugin_contracts/test_plugin_path_loading_contracts.py -k "custom_rm or dynamic_filter" -q
```

文档证据：

```powershell
node maintenance/audit_source_evidence.mjs --note slime_reading/Rollout生成/Reward与过滤/Slime-Reward与过滤-源码走读.md
```

## Q10：Slime RM Hub 与外部 serving RM 的边界是什么？

Slime RM Hub 是 rollout 进程内的打分入口；外部 RM 是它通过 `remote_rm` 调用的 HTTP 服务。前者决定 payload、重试和 reward 写入，后者决定模型推理和 JSON 返回格式。

关键边界：外部服务返回什么，Slime 就写入 `sample.reward`；如果不是标量，Slime 需要 `reward_key` 才能把它用于 filter 和训练。
