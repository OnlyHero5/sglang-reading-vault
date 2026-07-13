---
title: "其他Rollout路径 · 学习检查"
type: exercise
framework: slime
topic: "其他Rollout路径"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 其他Rollout路径 · 学习检查

这份清单用来判断你是否真的会选择和修改替代 rollout，而不是只记住几个函数路径。

## 读者能画出来

- [ ] 能画出默认 `RolloutManager -> rollout function -> Sample group -> train data` 出口。
- [ ] 能标出 fully-async 替换整段 rollout function，streaming 只替换单 sample generate。
- [ ] 能画出 `train_async.py` 的 step 间重叠，以及 fully-async worker 的跨 step 热队列。
- [ ] 能画出 SFT 从 messages 到 `tokens/loss_mask/response_length` 的数据流。
- [ ] 能画出 OPD 从 teacher server 到 `sample.teacher_log_probs` 的数据流。
- [ ] 能画出 forge load 从 dump 到 `RolloutFnTrainOutput` 的路径，并标出不覆盖 `rollout_id`。

## 读者能解释清楚

- [ ] 为什么整段 rollout 替换必须满足 `generate_rollout(args, rollout_id, data_source, evaluation=False)`。
- [ ] 为什么 fully-async 不支持 evaluation。
- [ ] 为什么 fully-async 的 ABORTED group 回灌 DataSource，而不是进入 output queue。
- [ ] 能解释为什么一次 drain 超过 target 会丢弃多余完成 group，以及 task/回灌异常为何不具备 exactly-once。
- [ ] 为什么 streaming 的 chunk 处理要先恢复 base Sample 再 append。
- [ ] 为什么 SFT rollout 可以 `reward=0`，但不能错写 `loss_mask`。
- [ ] 为什么 OPD 标量 reward 为 0 仍可能有学习信号。
- [ ] 能解释 SFT 与 OPD 中 `[-0:]` 为什么返回全量，并给出零 response 的 fail-fast 策略。
- [ ] 为什么 forge load 不是 `load_debug_rollout_data` 的同义词。

## 读者能排障

- [ ] fully-async 队列不增长时，能检查 DataSource、worker crash、SGLang 健康、custom generate 卡死。
- [ ] eval 报 fully-async 不支持时，能把训练 rollout 和 eval function 分开配置。
- [ ] streaming partial 没保存时，能检查 SSE chunk 是否返回、HTTP client 是否初始化、partial rollout 是否触发 abort。
- [ ] SFT loss 全零时，能用最小 messages 解码 masked token 验证 assistant/user 边界。
- [ ] OPD reward 均值全零时，能检查 `teacher_log_probs` 而不是只看 reward。
- [ ] forge load 触发分组断言时，能检查 `sample.rollout_id` 与 `sample.index`。
- [ ] streaming 排障时能同时核对 cumulative/incremental 模式以及 text/token/logprob 四者对齐。

## 读者能做最小验证

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/plugin_contracts/test_plugin_rollout_contracts.py -q
python -m pytest slime/tests/plugin_contracts/test_plugin_generate_contracts.py -q
python -m pytest slime/tests/gemma4/test_gemma4_sft_rollout.py -q
```

预期现象：

- rollout contract 测试验证整段 rollout function 的签名和返回包装。
- generate contract 测试验证单 sample generate 的优先级、fan-out 和 evaluation 参数。
- Gemma4 SFT 测试在 checkpoint 存在时验证 messages mask；checkpoint 不存在时应 skip。

端到端验证要用真实训练环境：

- [ ] fully-async smoke：`tests/test_qwen2.5_0.5B_fully_async_short.py`，预期日志出现 `fully-async rollout <id>: target=<count>`。
- [ ] streaming partial smoke：`tests/test_qwen3_4B_streaming_partial_rollout.py`，预期 oversampling + partial rollout 会触发 abort 并回收 partial group。

## 复盘迁移

- [ ] 接到 [[Slime-SGLang-Rollout]] 时，能说明哪些默认能力被 fully-async 保留，哪些没有保留。
- [ ] 接到 [[Slime-数据源]] 时，能说明 fully-async 和 forge 分别怎样使用 DataSource。
- [ ] 接到 [[Slime-Sample数据契约]] 时，能说明 streaming、SFT、OPD 都必须最终落到 Sample 字段契约。
- [ ] 接到 [[Slime-Reward与过滤]] 时，能区分 OPD teacher scorer 与普通 reward scorer。

## 通过标准

全部满足后，应该能做到：

- 根据需求选择替换整段 rollout、单 sample generate、reward hook 或样本来源。
- 写一个新的 rollout function，并知道哪些默认能力需要自己补回来。
- 接入 streaming、SFT、OPD、forge 时，能提前指出最可能破坏的字段不变量。
