---
title: "Sampling · 学习检查"
type: exercise
framework: sglang
topic: "Sampling"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# Sampling · 学习检查

## 读完应能回答

- [ ] Sampling 为什么不是 API 参数表，而是从 logits 到 next token 的生产线。
- [ ] `SamplingParams.__post_init__`、`verify` 和 `normalize` 分别解决什么问题。
- [ ] `temperature=0` 在哪个阶段被翻译成 `top_k=1`，为什么不是除以零。
- [ ] 带 `json_schema` 的请求为什么可能先进入 `grammar_queue`。
- [ ] `SamplingBatchInfo` 中哪些字段是 `[bs]`，哪些字段会扩展到 `[bs, vocab]`。
- [ ] penalty、grammar mask、logit bias、custom processor 按什么顺序改写 logits。
- [ ] 全 batch greedy 为什么能跳过 softmax 和概率采样。
- [ ] `top_p/top_k/min_p` 最终在哪个函数里选择 backend 分支。
- [ ] 采样后为什么还要推进 grammar 和 penalty 状态。
- [ ] 约束解码下为什么需要同步 TP token ids。
- [ ] custom logit processor 为什么可能破坏已经施加的 grammar mask。
- [ ] deterministic inference、默认 seed、FlashInfer 和 min-p 的兼容边界。

## 打开源码后的定位题

| 问题 | 应定位到 |
|------|----------|
| 参数越界 | `SamplingParams.verify` |
| stop 字符串在 `skip_tokenizer_init` 下失败 | `raise_if_tokenizer_required` |
| JSON schema 请求不进 batch | `GrammarManager.process_req_with_grammar` |
| 空 `structural_tag` 异常 | `process_req_with_grammar` 的外层 `is not None` 与内层 truthy 判断 |
| grammar timeout | `GrammarManager.get_ready_grammar_requests` |
| batch 不是 greedy | `SamplingBatchInfo.from_schedule_batch` 的 `is_all_greedy` |
| 过滤后仍未走 greedy fast path | `SamplingBatchInfo.filter_batch` 不重算 batch flags |
| grammar token 被挡 | `update_regex_vocab_mask` 和 backend `apply_vocab_mask` |
| repetition penalty 不生效 | `BatchedPenalizerOrchestrator.cumulate_output_tokens/apply` |
| top-p 没走预期 backend | `Sampler._sample_from_probs` |
| grammar + TP hang | `Sampler._sync_token_ids_across_tp` |
| spec decode 输出非法后缀 | `_accept_grammar_tokens` |
| seed 打开后 backend assert | `Sampler._sample_from_probs` 与 PyTorch min-p 分支 |

## 可观测验证

**操作：** 按参数层、grammar 层、batch 层、logits 层、sampler 层逐级记录状态；每次只改变一个采样条件。

**预期：** 你应能指出行为第一次偏离预期的层级，并用该层的字段解释原因，而不是只看最终 token 猜测。

1. 参数层：构造非法 `top_p`、非法 `logit_bias` token id、同时传 `json_schema/regex`，确认在参数校验阶段失败；再确认 `temperature=0` 构造后变成 `temperature=1.0、top_k=1`。
2. Grammar 层：对一个复杂 schema 打印 `req.grammar_key`、`grammar_wait_ct`、cache hit 状态。
3. Batch 层：打印 `SamplingBatchInfo.is_all_greedy`、`need_top_p_sampling`、`need_min_p_sampling`、`temperatures.shape`。
4. Logits 层：分别在 ModelRunner 预处理后和 custom processor 后比较目标 token 的 logits，确认 penalty、mask、bias 是否生效，以及扩展是否重新开放 mask 位。
5. Sampler 层：分别跑 greedy、top-p、min-p、deterministic sampling，确认分支和 backend；特别记录 FlashInfer complex sampling 与 PyTorch min-p 对 seed 的限制。
6. 状态层：生成多步后检查 penalty orchestrator 是否累计了输出 token，grammar 是否接受了每步 token。
7. TP 层：约束解码下确认各 rank 的 `batch_next_token_ids` 同步。

## 自我复述

Sampling 的核心不是“随机抽 token”，而是“把请求意图、结构约束、历史惩罚、扩展处理器和后端能力组合成下一 token 决策”。请合上文档复述：参数如何变成 batch 状态，logits 按什么顺序被改写，token 如何被提交，以及这一步的输出怎样成为下一步 penalty 的历史。
