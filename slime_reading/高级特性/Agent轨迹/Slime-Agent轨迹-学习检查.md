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
updated: 2026-07-10
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

## 可执行验证

核心 trajectory 和 adapter 验证：

```powershell
python -m pytest tests/test_agent/test_trajectory_manager_branching.py tests/test_agent/test_adapters.py -q
```

完整 agent CPU 验证：

```powershell
python -m pytest tests/test_agent/ -q
```

专题源码引用审计：

```powershell
node maintenance/audit_source_evidence.mjs --note 'slime_reading/高级特性/Agent轨迹/Slime-Agent轨迹-源码走读.md'
```

专题旧结构残留：

```powershell
$old = 'Explain','Code','Comment' | ForEach-Object { [regex]::Escape($_) + '[:：]' }
rg -n ($old -join '|') 'slime_reading/高级特性/Agent轨迹'
$dots = ([string][char]46) * 3
$cn = ([string][char]0x2026) * 2
rg -n ([regex]::Escape($dots) + '|' + [regex]::Escape($cn)) 'slime_reading/高级特性/Agent轨迹'
```

全 vault 链接审计：

```powershell
node maintenance/audit_wikilinks.mjs
```

## 预期现象

- branching 测试覆盖 single turn、多轮 tool、tree fork、drift realign/fork、rewrite merge、cross-leaf dedup、logprob 长度不匹配、empty prompt。
- adapter 测试覆盖 Anthropic/OpenAI session id、wire translation、stream/non-stream response、multi-turn tool roundtrip、turn cap、parsing fallback。
- 每个 emitted sample 满足 `len(loss_mask) == len(rollout_log_probs) == response_length`，并且至少有一个训练 token。
- 专题旧结构搜索没有命中。
- 全 vault wikilink 审计 broken target 为 0。

## 下游衔接

| 下一篇 | 为什么读 |
|--------|----------|
| [[Slime-自定义扩展]] | 看 adapter 和 trajectory 如何挂到 `--custom-generate-function-path`、`--custom-rm-path` |
| [[Slime-插件与示例]] | 看 Search-R1、coding agent、多 agent 如何把这些对象用于真实示例 |
