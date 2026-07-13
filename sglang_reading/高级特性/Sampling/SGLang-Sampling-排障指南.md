---
title: "Sampling · 排障指南"
type: troubleshooting
framework: sglang
topic: "Sampling"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Sampling · 排障指南

## 怎么读这篇

这篇按症状排障。先判断问题发生在哪个阶段，再去对应源码入口：

| 阶段 | 典型症状 |
|------|----------|
| 参数规范化 | 参数越界、stop 失效、`skip_tokenizer_init` 下报错 |
| Grammar 准入 | JSON schema 请求不进 batch、grammar timeout |
| Logits 预处理 | 合法 token 被挡、重复惩罚不生效、custom processor 破坏 mask |
| Sampler | greedy/top-p/min-p 分支不符合预期 |
| 状态推进 | grammar 后续 token 错、TP hang、spec decode 输出非法后缀 |

## Q1：`--grammar-backend none` 下带 schema 的请求会怎样？

不会静默降级为普通采样。只要请求带 `json_schema`、`regex`、`ebnf` 或 `structural_tag`，而 `grammar_backend` 是 `None`，请求会被 abort。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L98-L101
            if self.grammar_backend is None:
                error_msg = "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not supported when the server is launched with --grammar-backend none"
                req.set_finish_with_abort(error_msg)
            else:
```

验证：启动时禁用 grammar backend，再发带 `json_schema` 的请求。预期不是无约束输出，而是请求失败。

## Q2：同时传 `json_schema` 和 `regex`，谁生效？`structural_tag` 呢？

先说结论：`SamplingParams.verify` 会禁止 `json_schema`、`regex`、`ebnf` 多者同时存在；但 `structural_tag` 不在互斥列表里。如果它与前三者之一并存，校验不会为此报错，`GrammarManager` 会按 `json_schema`、`regex`、`ebnf`、`structural_tag` 的顺序选择，导致 `structural_tag` 被静默遮蔽。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L226-L232
        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")
```

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

验证：构造同时带 `json_schema` 和 `regex` 的请求，应该在参数校验阶段失败；再构造 `json_schema + structural_tag`，观察 `grammar_key` 选择 `("json", ...)`。不要传空字符串 `structural_tag`：它能通过外层 `is not None` 判断，却不能给内部 `key` 赋值，当前基线会在 grammar lookup 前失败。

## Q3：`thinking_budget` 为什么有时不生效？

`thinking_budget` 来自 `sampling_params.custom_params`，并且只有当 `req.grammar` 是 `ReasonerGrammarObject` 时才会设置 `max_think_tokens`。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L75-L87
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
```

验证：打印 `type(req.grammar)` 和 `req.grammar.max_think_tokens`。如果没有 reasoning grammar，仅传 `custom_params` 不会产生预算效果。

## Q4：为什么 `temperature=0` 不会除零，却仍不一定走最快路径？

构造 `SamplingParams` 时，接近零的非负 temperature 会先被改写为 `temperature=1.0、top_k=1`，所以后续不会除零。greedy fast path 再看 batch 级 `is_all_greedy`：只要构造 batch 时有请求的 `top_k > 1`，整个 batch 就走非 greedy 路径。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L168-L174
        # Process some special cases
        if 0 <= self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL  # whole vocabulary
```

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L184-L193
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
```

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

验证：先打印构造后的 `temperature/top_k`，再打印 batch 内每个 request 的 `top_k` 和 `sampling_info.is_all_greedy`。还要注意当前 `filter_batch()` 不重算这个标志：非 greedy 行后来被过滤掉，剩余行也可能继续走概率路径，只是结果仍受 `top_k=1` 约束。

## Q5：Grammar 编译超时后会怎样？

超时请求会 cancel Future，把 invalid grammar 放进 cache，并 abort 请求。多 rank 下 failed 集合取 union，所以任一 rank 超时都会导致同步失败。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L224-L235
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

验证：观察 `grammar_wait_ct`、`SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` 和日志中的 timeout 信息。不要把这种情况误判成 GPU forward 慢。

## Q6：`skip_tokenizer_init=True` 下哪些采样功能会直接报错？

字符串 stop、regex stop、`min_new_tokens` 都依赖 tokenizer。没有 tokenizer 时会在 normalize 阶段报错。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L58-L72
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

验证：开启 `skip_tokenizer_init` 后分别发送 stop string、stop regex、`min_new_tokens>0` 的请求，应该在请求规范化阶段失败。

## Q7：为什么设置了 top-p，却看起来没有走 top-p kernel？

先看 `need_top_p_sampling`。如果 batch 中没有任何请求的 `top_p != 1.0`，Sampler 会把它当作简单采样；如果需要 top-p，还要看 backend 是 `flashinfer` 还是 `pytorch`。

```python
# 来源：python/sglang/srt/layers/sampler.py L226-L264
        if simple_sampling_case:
            batch_next_token_ids = sampling_from_probs_torch(
                probs,
                sampling_seed=sampling_info.sampling_seed,
                positions=positions,
            )
        else:
            backend = get_global_server_args().sampling_backend
            if backend == "flashinfer":
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )
            elif backend == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                    sampling_info.sampling_seed,
                    positions,
                )
            else:
                raise ValueError(f"Invalid sampling backend: {backend}")
        return batch_next_token_ids
```

验证：打印 `sampling_info.need_top_p_sampling`、`need_top_k_sampling`、`need_min_p_sampling` 和 `sampling_backend`。

## Q8：为什么 grammar 下 TP rank 更容易 hang？

因为 grammar 状态依赖每一步接受的 token id。约束解码下 SGLang 会同步 token id，避免不同 TP rank 因非确定性采样进入不同 grammar 状态。

```python
# 来源：python/sglang/srt/layers/sampler.py L382-L395
        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )
```

验证：出现 grammar 相关 TP hang 时，先确认 `sampling_info.grammars` 是否非空，以及 token ids 是否执行了 `MIN all_reduce`。这不是 rank 0 广播；若各 rank 局部结果不同，最终统一为其中最小的 token id。

## Q9：spec decode 下 grammar 为什么会截断草稿 token？

spec decode 可能一次接受多个 token。Grammar 终止后，后面的 over-drafted suffix 不能进入 KV，也不能输出给用户，所以 `_accept_grammar_tokens` 会在 grammar terminated 后停止保留。

```python
# 来源：python/sglang/srt/managers/scheduler_components/batch_result_processor.py L586-L607
    def _accept_grammar_tokens(
        self, req: Req, tokens: Union[int, List[int]]
    ) -> List[int]:
        """Advance the grammar over the accepted token(s), stopping at the token
        that terminates it.

        ``tokens`` is a single sampled token (normal decode) or the whole
        verified run (spec decode). Returns the retained prefix; for spec the
        suffix past grammar completion is dropped so it is never committed to KV
        nor emitted. Advances the grammar FSM only -- ``grammar.finished`` is
        synced by the caller once the finish state is updated.
        """
        if isinstance(tokens, int):
            tokens = [tokens]
        retained = []
        try:
            for token_id in tokens:
                req.grammar.accept_token(token_id)
                retained.append(token_id)
                if req.grammar.is_terminated():
                    break
        except ValueError as e:
```

验证：在 spec decode + grammar 的请求中打印 accepted token run 和 retained prefix。超过 grammar 终止点的 token 不应被提交。

## Q10：为什么开了 deterministic inference，FlashInfer 或 min-p 反而 assert？

`sampling_seed` 只有服务器开启 deterministic inference 时才会创建；请求没显式给 seed 的行也会填默认值 `42`。simple sampling 可以走基于 seed 和 position 的 Gumbel trick，但 backend 支持并不统一：FlashInfer 的 complex sampling 明确拒绝非空 seed；PyTorch 的 top-k/top-p complex 路径可带 seed，batch 只要需要 min-p 就会拒绝，因为过滤概率尚未重归一化。

```python
# 来源：python/sglang/srt/layers/sampler.py L234-L250
            if backend == "flashinfer":
                assert (
                    sampling_info.sampling_seed is None
                ), "Sampling seed is not supported for flashinfer backend"
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs.contiguous(),
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )
```

验证：同时记录 `enable_deterministic_inference`、batch 的 `sampling_seed`、`sampling_backend` 和 `need_min_p_sampling`。不要把“请求没传 seed”误认为 batch seed 一定是 `None`。

## Q11：grammar mask 已经生效，为什么仍可能采到非法 token？

`ModelRunner._preprocess_logits` 先施加 penalty、grammar mask 和 logit bias；随后 `Sampler._preprocess_logits` 才调用 custom logit processor。自定义 processor 如果把 mask 位重新写成有限值，就能重新开放非法 token。它是可信代码扩展点，不是天然受 grammar 保护的沙箱。其后的 `sanitize_nan_logits` 只有相应环境开关开启时才会把 NaN/±Inf 改成有限哨兵值，不能替代约束正确性检查。

```python
# 来源：python/sglang/srt/layers/sampler.py L84-L91
    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """Apply custom logit processors and sanitize non-finite logits."""
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(logits, sampling_info)
        sanitize_nan_logits(logits, "sampler: next_token_logits")
        return logits
```

验证：先在 ModelRunner 预处理后检查非法位置是否为 `-inf`，再在 custom processor 返回时复查同一位置；不要把随后可选的 Inf 哨兵化误判成 processor 行为。若 processor 已把 mask 位改成普通有限分数，应修 processor，而不是更换 grammar backend。

## 运行验证

Sampling FAQ 的源码复核入口可以覆盖 grammar 准入、参数规范化、batch 级采样开关、sampler backend 分支、TP token 同步和 grammar token 接受。

```powershell
rg -n 'GrammarManager|process_req_with_grammar|def normalize|def update_regex_vocab_mask|def forward|sampling_backend|need_top_p_sampling|need_top_k_sampling|need_min_p_sampling|def _sync_token_ids_across_tp|def _accept_grammar_tokens' sglang/python/sglang/srt/constrained/grammar_manager.py sglang/python/sglang/srt/sampling/sampling_params.py sglang/python/sglang/srt/sampling/sampling_batch_info.py sglang/python/sglang/srt/layers/sampler.py sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py
```

如果采样问题无法通过用户参数解释，按命中顺序继续查：先看 request 是否进 grammar queue，再看 `SamplingBatchInfo` 的 flags/seed 是否与当前 batch 一致，然后按 ModelRunner 预处理、custom processor、sampler backend、TP 同步和 `_accept_grammar_tokens` 的顺序定位第一次偏离。
