---
title: "Agent轨迹 · 学习检查"
type: exercise
framework: slime
topic: "Agent轨迹"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# Agent轨迹 · 学习检查

## 读者能做什么

- [ ] 能画出 `wire request -> chat-template messages -> prompt_ids -> TurnRecord -> MessageNode tree -> _SampleBuilder -> Sample`。
- [ ] 能解释为什么训练目标必须保留 SGLang `output_token_logprobs` 中的 token ids，而不是 decode 后重 tokenize。
- [ ] 能区分 tree fork 和 token drift fork：前者由 message dict equality 决定，后者由 prompt id common prefix 决定。
- [ ] 能说明 `CLEAN`、`REALIGN`、`FORK` 三种 drift 处理条件。
- [ ] 能解释 `response_trained` 如何避免 sibling leaf 重复训练同一 assistant 前缀。
- [ ] 能说出 adapter 为什么先 flush wire response，再 `record_turn`。
- [ ] 能说明 `finish_session` 为什么是一次性消费，以及 custom generate 应如何缓存返回 samples。
- [ ] 能指出 OpenAI adapter 只实现 Chat Completions，并解释为什么当前 reply 只保留第一个 tool call。
- [ ] 能说明缺省 sid=`default` 会如何混合 store、turn cap 和 trajectory tree。
- [ ] 能用数值例子证明 REALIGN 阈值比较新 output 总长度，而非 drift tail 长度。
- [ ] 能审计 `max_sample_tokens < leading_prompt_len`、截断后无训练 token 和负 `response_length` 三个边界。
- [ ] 能解释 fan-out 为什么给每个 sample 完整 reward，以及下游必须按 `rollout_id` 核对聚合口径。
- [ ] 能描述 `response_trained` 提前原地修改导致 finish 异常重试不等价的状态账。

## 可执行验证

核心 trajectory 和 adapter 验证：

```powershell
Push-Location slime
python -m pytest tests/test_agent/test_trajectory_manager_branching.py tests/test_agent/test_adapters.py -q
Pop-Location
```

完整 agent CPU 验证：

```powershell
Push-Location slime
python -m pytest tests/test_agent/ -q
Pop-Location
```

静态契约定位：

```powershell
rg -n 'TurnRecord|MessageNode|response_trained|finish_session|record_turn|output_token_logprobs' slime/slime slime/tests/test_agent
rg -n 'tool_calls\[:1\]|return .*"default"|max_sample_tokens|leading_prompt_len|s\.reward = reward' slime/slime/agent
```

边界微实验（当前仓库测试未覆盖，应新增或在隔离脚本中验证）：

1. 构造包含两个 parsed tool uses 的 OpenAI reply，确认当前 wire/manager 只有一个 call。
2. 两个客户端都不传 sid，确认它们都解析为 `default`，并说明为什么生产应拒绝这种用法。
3. 令 `max_sample_tokens` 分别小于、等于、略大于首轮 prompt 长度，检查 `response_length`、loss-mask 总和与 tokens 对齐。
4. 在第一个 leaf 线性化后人为抛错，确认 tree 未 pop 但共享 node 的 `response_trained` 已改变。

## 预期现象

- branching 测试覆盖 single turn、多轮 tool、tree fork、drift realign/fork、rewrite merge、cross-leaf dedup、logprob 长度不匹配、empty prompt。
- adapter 测试覆盖 Anthropic/OpenAI session id、wire translation、stream/non-stream response、multi-turn tool roundtrip、turn cap、parsing fallback。
- 在未传或安全设置 `max_sample_tokens` 的路径，每个 emitted sample 满足 `len(loss_mask) == len(rollout_log_probs) == response_length`，并且至少有一个训练 token；微实验应反证 cap 切入 prompt 时当前实现会破坏该不变量。
- 静态结果能串出 wire response、turn 记录、tree/fork、sample 构建和一次性 session 收口。
- 能明确列出现有两份测试未覆盖的并行 tool calls、Responses API、默认 sid 隔离、同 sid 并发、context cap 和 finish 异常重试边界。
- 当前环境核心两文件为 49 passed/1 skipped，完整 agent CPU 目录为 62 passed/1 skipped；skip 是缺 SGLang 的 qwen3 reasoning parser 用例，不得冒充通过。

## 下游衔接

| 下一篇 | 为什么读 |
|--------|----------|
| [[Slime-自定义扩展]] | 看 adapter 和 trajectory 如何挂到 `--custom-generate-function-path`、`--custom-rm-path` |
| [[Slime-插件与示例]] | 看 Search-R1、coding agent、多 agent 如何把这些对象用于真实示例 |
