---
title: "SGLang-Rollout · 学习检查"
type: exercise
framework: slime
topic: "SGLang-Rollout"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# SGLang-Rollout · 学习检查

这份清单用来判断你是否真的理解默认 SGLang Rollout，而不是只知道 `generate` 会发 HTTP。

## 读者能画出来

- [ ] 能画出 `RolloutManager -> generate_rollout -> generate_rollout_async -> generate_and_rm_group -> generate_and_rm -> generate -> Sample.append_response_tokens` 主线。
- [ ] 能在图上标出 DataSource 补样、dynamic filter、abort 回灌、RolloutFnTrainOutput 四个边界。
- [ ] 能画出 train 和 eval 两条路径的区别：train 走有效 batch 水位，eval 走固定 eval dataset 展开。

## 读者能解释清楚

- [ ] `rollout_batch_size` 为什么是有效 group 数。
- [ ] `remaining_batch_size` 为什么 filter drop 后要减少。
- [ ] 为什么 `remaining_batch_size` 包含已 keep group 与 pending group，不能当成 `len(pendings)`。
- [ ] 为什么 task 粒度是 group，组内再并发 sample。
- [ ] `GenerateState` 为什么要同时保存 semaphore、pending set、sampling params 和 abort 标志。
- [ ] 为什么单例第一次 args 会冻结 tokenizer、semaphore 和 sampling template，以及 `reset()` 为什么不能替代 cancel/drain task。
- [ ] `--rollout-function-path` 与 `--custom-generate-function-path` 的替换边界。
- [ ] 为什么 custom generate 最好走 `Sample.append_response_tokens`，而不是手写多个字段。
- [ ] 为什么 `dp_rank_context` 在默认 HTTP 路径中不是强制 DP 定向路由。

## 读者能排障

- [ ] rollout 长时间不结束时，能检查 dynamic filter drop、DataSource 补样、`over_sampling_batch_size` 和 reward 分布。
- [ ] custom generate 不生效时，能检查 sample 级 `generate_function_path` 是否覆盖全局参数。
- [ ] top-p metric 为空时，能判断是 `rollout_top_p=1.0` 的正常结果，还是 offsets 没写入。
- [ ] partial rollout 没回收时，能检查 `partial_rollout`、pending task、DataSource `add_samples`。
- [ ] eval 与 train 行为不一致时，能说明它们本来不共享训练水位和 dynamic filter。
- [ ] fan-out 单测通过但 group RM/partial abort 崩溃时，能检查 leaf 从 `Sample` 变成 `list[Sample]` 的位置。
- [ ] HTTP 有文本但训练 token 为空时，能检查 `output_token_logprobs`，而不是只看状态码和 `text`。
- [ ] 能解释最终 sample filter 为什么应设置 `remove_sample` 而不是删除 group，以及它为什么不回滚已发生的 advantage normalization。

## 读者能做最小验证

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/test_rollout_metrics.py -q
python -m pytest slime/tests/plugin_contracts/test_plugin_generate_contracts.py -q
python -m pytest slime/tests/plugin_contracts/test_plugin_rollout_contracts.py -q
```

预期现象：

- `test_rollout_metrics.py` 验证 Sample 响应字段不变量。
- `test_plugin_generate_contracts.py` 验证 custom generate 优先级和 `generate_and_rm` 的单层 fan-out 返回；不证明 group RM、partial abort、filter/hook 组合闭合。
- `test_plugin_rollout_contracts.py` 验证整段 rollout function 的签名、train/eval 输出和 legacy 包装。

当前环境的实际结果是：三份 plugin contract 直接 collection 均缺 `httpx`；最小 stub 后继续暴露缺 `pylatexenc` 和 PyArrow/Torch 对 NumPy 2.x 的 ABI 问题，`test_rollout_metrics.py` 还缺 `ray`。静态替代应从当前 AST 检查关键控制流，但必须明确它没有启动真实 router、HTTP client、RM 或 Ray。

## 复盘迁移

- [ ] 接到 [[Slime-数据源]] 时，能说明 `get_samples` 提供的是 prompt group，SGLang Rollout 只负责把 group 变成 generated group。
- [ ] 接到 [[Slime-Sample数据契约]] 时，能说明 `append_response_tokens` 如何保持 tokens、logprobs、loss mask、top-p offsets、status 对齐。
- [ ] 接到 [[Slime-Reward与过滤]] 时，能区分 sample RM、group RM 和 dynamic filter 的时机。
- [ ] 接到 [[Slime-其他Rollout路径]] 时，能判断一个变体替换的是整段 rollout，还是只替换 sample generate。

## 通过标准

全部满足后，应该能做到：

- 能先依靠笔记复述一次训练 rollout 的生命周期；遇到修改、版本漂移或组合边界时，能回到当前源码基线逐项核证。
- 根据症状判断问题在 DataSource、SGLang HTTP、Sample 账本、RM/filter、partial abort 还是 eval。
- 写一个 custom generate，并知道哪些字段必须由它或后续 RM 补齐。
