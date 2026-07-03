---
type: batch-doc
module: 20-Sampling
batch: "20"
doc_type: checkpoint
title: "Sampling 验收清单"
tags:
 - sglang/batch/20
 - sglang/module/sampling
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# Sampling 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能画出 GrammarManager 异步编译 → apply_mask → sample 完整链路
- [ ] 能对比 xgrammar/outlines/llguidance 三后端差异
- [ ] 能说明 penalty 与 grammar mask 的施加顺序
- [ ] 能解释 `--grammar-backend none` 与 thinking_budget 行为
- [ ] 能说出 json_schema 与 regex 的 cache key 区别
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论（3 句话）

1. **GrammarManager 用 ThreadPoolExecutor 异步编译约束**，cache miss 入 grammar_queue 轮询，DP 组 all_gather 同步就绪状态。
2. **采样前 `apply_logits_bias` 严格按 penalty → grammar mask → logit_bias 顺序修改 logits**，mask 用后立即释放防 VRAM 泄漏。
3. **Sampler 根据 `is_all_greedy` 短路 argmax 或走 temperature + FlashInfer top_p 路径**，grammar 采样后 accept_token 推进状态机。
