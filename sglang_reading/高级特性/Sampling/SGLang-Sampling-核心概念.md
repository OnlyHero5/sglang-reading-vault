---
title: "Sampling · 核心概念"
type: concept
framework: sglang
topic: "Sampling"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# Sampling · 核心概念

## 先回答为什么读

当用户说“我设置了 `json_schema` 但输出还是不合法”“`temperature=0` 为什么这么快”“`top_p` 没生效”“grammar 卡住了”，这些问题都不能只看 API 层。Sampling 是从 logits 到 next token 的最后一段控制系统。

读完本篇，你应该能把问题放进正确阶段：

| 症状 | 更可能的阶段 |
|------|--------------|
| 参数越界、stop 字符串不能用 | `SamplingParams.verify/normalize` |
| JSON schema 请求迟迟不进 batch | `GrammarManager` 编译队列 |
| 结构化输出 token 被挡掉 | grammar vocab mask |
| 重复惩罚或 presence/frequency 不生效 | penalty orchestrator |
| greedy 和随机采样性能差异大 | `Sampler.forward` 的 greedy short path |
| 多 TP rank 约束解码 hang | grammar 后的 token id 同步 |

## 心理模型：下一 token 生产线

Sampling 不是一个单独函数，而是一条流水线：

```text
单请求参数
  -> 是否需要 grammar 编译
  -> batch 级张量和状态
  -> logits 原地改写
  -> greedy 或概率采样
  -> grammar/penalty 状态推进
```

### 1. 参数先被校验和规范化

`verify` 会检查数值范围、logit bias token id 和 grammar 字段互斥关系。这里仍是 CPU 侧的请求语义检查。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L176-L232
    def verify(self, vocab_size):
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError(
                f"temperature must be a non-negative finite number, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < 1 or self.top_k == -1:
            raise ValueError(
                f"top_k must be -1 (disable) or at least 1, got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 < self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in (0, 2] (1.0 = no penalty), "
                f"got {self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in [0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in [0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )

        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")
```

注意 `structural_tag` 没在这个互斥列表里；后面 `GrammarManager` 会按 `json_schema -> regex -> ebnf -> structural_tag` 的顺序选 key。

### 2. Grammar 是请求准入前的异步关口

约束解码需要先把 schema/regex/ebnf/structural tag 编译成 grammar object。cache miss 时，请求会进入 `grammar_queue`，不会马上进入 waiting queue。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L89-L140
    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
        ):
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)

                value, cache_hit = self.grammar_backend.get_cached_or_future_value(
                    key, req.require_reasoning
                )
                req.grammar = value

                if not cache_hit:
                    req.grammar_key = key
                    add_to_grammar_queue = True
                else:
                    if isinstance(
                        value, InvalidGrammarObject
                    ):  # We hit a cached invalid grammar.
                        error_msg = (
                            f"Failed to compile {key[0]} grammar: {value.error_message}"
                        )
                        req.set_finish_with_abort(error_msg)
                    else:
                        self._apply_request_reasoning_budget(req)
        elif self._enable_strict_thinking:
            grammar_obj = self.grammar_backend.init_strict_reasoning_grammar(
                req.require_reasoning
            )
            if grammar_obj is not None:
                req.grammar = grammar_obj
                self._apply_request_reasoning_budget(req)

        if add_to_grammar_queue:
            self.grammar_queue.append(req)

        return add_to_grammar_queue
```

这解释了为什么结构化输出请求的慢点常常不在 sampler kernel，而在 grammar 编译和排队。

### 3. Batch 信息把标量变成张量

Scheduler 组 batch 后，`SamplingBatchInfo` 把每个 request 的温度、top-p、top-k、min-p 等参数搬到设备上，并计算 batch 级开关。

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L184-L203
        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
        )
        ret.adjusted_from_schedule_batch(batch, vocab_size)
        return ret
```

`is_all_greedy` 是 batch 级开关：只要 batch 中有一行不是 greedy，整个 batch 就不能走纯 argmax fast path。

### 4. Logits 被原地改写

`ModelRunner` 在调用 Sampler 前，会先更新 grammar mask，然后调用 `apply_logits_bias`。这个函数内部还会施加 penalty、grammar mask 和 logit bias。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L3143-L3158
    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None
```

因此“某 token 为什么不可能被选中”要先看 logits 预处理，而不是只看最终 top-p 采样。

### 5. Sampler 决定 greedy 还是随机

`Sampler.forward` 会先处理 custom logit processor 和 NaN，然后根据 `is_all_greedy` 分支。全 greedy 时直接 argmax；否则进入温度缩放、softmax 和概率采样路径。

```python
# 来源：python/sglang/srt/layers/sampler.py L116-L145
        logits = logits_output.next_token_logits

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:
            if _use_aiter and not _disable_aiter_greedy_sample:
                batch_next_token_ids = torch.empty(
                    logits.shape[0], device=logits.device, dtype=torch.int32
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)
            else:
                batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                original_logprobs = logprobs = torch.nn.functional.log_softmax(
                    logits, dim=-1
                )
        else:
            simple_sampling_case = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )

            # If requested, cache original logprobs before temperature scaling.
            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:
                original_logprobs = torch.log_softmax(logits, dim=-1)

            # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
            logprobs_via_logsoftmax_kernel = None
```

## 复盘迁移

读 Sampling 时可以把问题翻译成一句话：在下一 token 生产线的哪一站，候选 token 集合或分布被改变了。

| 改变方式 | 站点 |
|----------|------|
| 参数无效或 alias 清理 | `SamplingParams` |
| 请求暂不进入 batch | `GrammarManager` |
| 每行参数变成 GPU tensor | `SamplingBatchInfo` |
| 合法 token 集合被裁剪 | grammar vocab mask |
| token 概率被惩罚或加 bias | penalty / logit bias |
| 分布被抽样 | `Sampler.forward` |
| grammar 状态推进 | result processor |
