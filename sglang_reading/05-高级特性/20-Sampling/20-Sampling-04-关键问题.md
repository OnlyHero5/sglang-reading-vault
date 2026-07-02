---
type: batch-doc
module: 20-Sampling
batch: "20"
doc_type: faq
title: "Sampling：关键问题"
tags:
 - sglang/batch/20
 - sglang/module/sampling
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# Sampling：关键问题

## Q1：`--grammar-backend none` 会怎样？

**Explain：** 启动时 `create_grammar_backend` 返回 `None`，GrammarManager.grammar_backend 为 None。请求携带 json_schema/regex/ebnf/structural_tag 时，`process_req_with_grammar` 直接 `set_finish_with_abort`，不会 silent fallback 到无约束采样。若同时开启 `--enable-strict-thinking`，启动阶段就会 raise ValueError，因为 strict thinking 依赖 token filter。

**Code：**

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L98-L101
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
```

**易错对比：**

```python
# ❌ 误以为 none 只是慢——实际直接 abort
# 启动: --grammar-backend none
# 请求: {"json_schema": "{\"type\":\"object\"}"} → 请求失败

# ✅ 需要约束解码时必须显式指定 backend
# 启动: --grammar-backend xgrammar
```

---

## Q2：json_schema 与 regex 有什么区别？

**Explain：** 两者走同一 GrammarManager 路径，但 cache key 类型不同：`("json", schema_str)` vs `("regex", pattern_str)`。json_schema 由 backend 的 `dispatch_json` 编译为 JSON FSM，保证输出合法 JSON；regex 编译为正则自动机，只约束字符序列格式、不保证 JSON 语义。同一请求若同时设置两者，if-elif 链优先 json_schema，regex 被忽略。

**Code：**

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L102-L109
                if req.sampling_params.json_schema is not None:
                    key = ("json", req.sampling_params.json_schema)
                elif req.sampling_params.regex is not None:
                    key = ("regex", req.sampling_params.regex)
                elif req.sampling_params.ebnf is not None:
                    key = ("ebnf", req.sampling_params.ebnf)
                elif req.sampling_params.structural_tag:
                    key = ("structural_tag", req.sampling_params.structural_tag)
```

**易错对比：**

```python
# ❌ 用 regex 约束 JSON 结构——只能匹配字符模式，无法表达嵌套 schema
sampling_params.regex = r'\{.*\}'

# ✅ 结构化输出用 json_schema
sampling_params.json_schema = '{"type":"object","properties":{"name":{"type":"string"}}}'
```

---

## Q3：thinking_budget 如何生效？

**Explain：** 通过 `SamplingParams.custom_params["thinking_budget"]` 传入整数；GrammarManager 在 grammar 就绪后调用 `_apply_request_reasoning_budget`，若 grammar 是 `ReasonerGrammarObject` 则设置 `max_think_tokens`。超出 budget 后 grammar 强制结束 reasoning 阶段，只允许输出正文 token。需配合 `reasoning_parser` 和 ReasonerGrammarBackend 使用。

**Code：**

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L75-L96
    def _get_request_thinking_budget(self, req: Req) -> int | None:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        thinking_budget = custom_params.get("thinking_budget")
        return thinking_budget if isinstance(thinking_budget, int) else None

    def _apply_request_reasoning_budget(self, req: Req) -> None:
        thinking_budget = self._get_request_thinking_budget(req)
        if thinking_budget is None:
            return
        if isinstance(req.grammar, ReasonerGrammarObject):
            req.grammar.max_think_tokens = thinking_budget

    def process_req_with_grammar(self, req: Req) -> bool:
        # Init grammar cache for this request
        add_to_grammar_queue = False
        if (
            req.sampling_params.json_schema is not None
            or req.sampling_params.regex is not None
            or req.sampling_params.ebnf is not None
            or req.sampling_params.structural_tag is not None
```

**Comment：**
- 非 ReasonerGrammarObject 时 budget 被静默忽略
- strict_thinking 模式下即使用户未传 json_schema 也会初始化 reasoning grammar

---

## Q4：greedy vs sampling 如何短路？

**Explain：** `from_schedule_batch` 检查所有 req 的 temperature==0 且无 top_p/k/min_p 需求，则 `is_all_greedy=True`；Sampler 直接 `argmax` 跳过 softmax 与 FlashInfer 采样 kernel。任一 req 需要随机采样则整 batch 走完整路径（per-row temperature 仍可用）。

**Code：**

```python
# 来源：python/sglang/srt/layers/sampler.py L121-L128
        if sampling_info.is_all_greedy:
            if _use_aiter and not _disable_aiter_greedy_sample:
                batch_next_token_ids = torch.empty(
                    logits.shape[0], device=logits.device, dtype=torch.int32
                )
                _aiter_greedy_sample(batch_next_token_ids, logits)
            else:
                batch_next_token_ids = torch.argmax(logits, -1)
```

---

## Q5：grammar 编译超时怎么办？

**Explain：** `get_ready_grammar_requests` 对每个未 ready 的 req 递增 `grammar_wait_ct`；超过 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` 则 cancel Future、缓存 `InvalidGrammarObject("Grammar preprocessing timed out")` 并 abort 请求。多 rank 时 failed 集合取 union，任一 rank 超时会同步 abort。

**Code：**

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L176-L235
        for i, req in enumerate(self.grammar_queue):
            if i not in ready_req_idxs:
                self.grammar_queue[i].grammar_wait_ct += 1
                if (
                    self.grammar_queue[i].grammar_wait_ct
                    >= self.SGLANG_GRAMMAR_MAX_POLL_ITERATIONS
                ):
                    # Timeout after max poll iterations
                    # The actual waiting time is SGLANG_GRAMMAR_MAX_POLL_ITERATIONS * max(SGLANG_GRAMMAR_POLL_INTERVAL, GPU_forward_batch_latency)
                    failed_req_idxs.add(i)

        # Sync ready and failed requests across all ranks
        if self.grammar_sync_size == 1:
            synced_ready_req_idxs = ready_req_idxs
            synced_failed_req_idxs = failed_req_idxs
        else:
            all_gather_output = [None] * self.grammar_sync_size
            torch.distributed.all_gather_object(
                all_gather_output,
                (ready_req_idxs, failed_req_idxs),
                group=self.grammar_sync_group,
            )
            synced_ready_req_idxs = set.intersection(*[x[0] for x in all_gather_output])
            synced_failed_req_idxs = set.union(*[x[1] for x in all_gather_output])

        # Return ready requests
        return_reqs: List[Req] = []
        for i in synced_ready_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)
            if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                continue

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            try:
                req.grammar = req.grammar.result()
            except Exception as e:
                logger.error(
                    f"Grammar compilation raised an exception: {e}, "
                    f"grammar_key={req.grammar_key}"
                )
                req.grammar = InvalidGrammarObject(f"Grammar compilation failed: {e}")
            self.grammar_backend.set_cache(req.grammar_key, req.grammar.copy())
            self._apply_request_reasoning_budget(req)
            if isinstance(req.grammar, InvalidGrammarObject):
                error_msg = f"Failed to compile {req.grammar_key[0]} grammar: {req.grammar.error_message}"
                req.set_finish_with_abort(error_msg)

        # Return failed requests
        for i in synced_failed_req_idxs:
            req = self.grammar_queue[i]
            return_reqs.append(req)

            assert isinstance(req.grammar, futures.Future) and req.grammar_key
            req.grammar.cancel()
            self.grammar_backend.set_cache(
                req.grammar_key, InvalidGrammarObject("Grammar preprocessing timed out")
            )
            error_msg = f"Grammar preprocessing timed out: {req.grammar_key=}"
            req.set_finish_with_abort(error_msg)
```

**Comment：**
- 复杂 schema 首次编译可能较慢，后续同 schema 走 cache hit
- 调大 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` 可延长等待

---

## Q6：skip_tokenizer_init 的限制？

**Explain：** 无 tokenizer 时无法 decode token 为字符串做 stop_str 匹配，也无法获取 eos_token_id 做 min_new_tokens penalty。`raise_if_tokenizer_required` 在 normalize 阶段直接 raise ValueError，而非运行时 silent 忽略。

**Code：**

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L45-L72
def raise_if_tokenizer_required(
    tokenizer, stop_strs, stop_regex_strs, min_new_tokens=0
):
    """Raise ValueError if tokenizer-dependent features are used without a tokenizer.

    String-based stop conditions (stop_strs, stop_regex_strs) require tokenizer.decode()
    to convert output token IDs to text for matching. min_new_tokens requires the
    tokenizer's eos_token_id to penalize. When skip_tokenizer_init=True, these cannot
    be used.
    """
    if tokenizer is not None:
        return

    if stop_strs:
        raise ValueError(
            f"stop={stop_strs!r} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer to decode tokens to text for matching)."
        )
    if stop_regex_strs:
        raise ValueError(
            f"stop_regex={stop_regex_strs!r} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer to decode tokens to text for matching)."
        )
    if min_new_tokens > 0:
        raise ValueError(
            f"min_new_tokens={min_new_tokens} is unavailable when skip_tokenizer_init=True "
            "(requires tokenizer for eos_token_id)."
        )
```

---

## 验证建议（零基础可试）

1. **对比有/无 json_schema 的延迟** 
 - 操作：同一 prompt 分别请求 `response_format: json_object` 与纯文本 generate。 
 - 预期：带 schema 的首条可能更慢；Prometheus 中 `num_grammar_queue_reqs` 可能升高。 
 - 对应：用户故事「JSON 缺括号」

2. **观察 temperature=0 的确定性** 
 - 操作：连续 3 次相同 prompt，`temperature: 0`。 
 - 预期：输出 token 序列一致（无 sampling 随机性）。 
 - 对应：[[20-Sampling-01-核心概念|01-核心概念 §2]]

3. **stop 字符串截断** 
 - 操作：`"stop": ["\n\n"]`，观察流式是否在双换行处结束。 
 - 预期：SSE 流提前 `[DONE]`；Detokenizer 侧完成 finish reason。 
 - 对应：[[20-Sampling-03-数据流与交互|03-数据流与交互]]
