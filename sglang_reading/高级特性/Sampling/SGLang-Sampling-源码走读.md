---
title: "Sampling · 源码走读"
type: walkthrough
framework: sglang
topic: "Sampling"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# Sampling · 源码走读

## 场景主线

这篇追一个 decode step：请求已经进入 serving，模型已经算出 `next_token_logits`，SGLang 要决定每个 batch 行的下一个 token id。

```text
SamplingParams.__post_init__/verify/normalize
  -> GrammarManager.process_req_with_grammar
  -> Scheduler waiting queue
  -> SamplingBatchInfo.from_schedule_batch
  -> ModelRunner._preprocess_logits
  -> Sampler.forward
  -> BatchResultProcessor commits token and advances grammar
  -> ScheduleBatch accumulates the token for next-step penalties
```

这条线只表示共同骨架。普通 decode、overlap、speculative verify、Ascend 与 RL on-policy 会在骨架内分叉，不能拿一条箭头链替代实际控制流。

读这条线时要分清两件事：`SamplingParams` 描述单个请求想要什么，`SamplingBatchInfo` 描述一个 batch 在目标设备上怎样执行这些要求。

## 长文读法

这篇按“单请求采样意图如何变成 batch 级设备决策”读：`SamplingParams` 先处理特殊值、校验并规范化 stop/tokenizer 依赖；grammar 可能让请求暂留队列；进入 batch 后 `SamplingBatchInfo` 汇总温度、top-p/top-k/min-p、seed、logit bias、penalty、grammar 和 custom processor 元数据；ModelRunner 与 Sampler 分两层预处理 logits；Sampler 再决定 greedy、Ascend、RL log-softmax 或普通后端分支。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立 decode step 主线 | 场景主线、1 到 4 | 请求参数先变成 batch tensor，真正采样在模型 forward 之后 |
| 排查 stop/regex/min_new_tokens | 1 | tokenizer 依赖在 normalize 阶段就会被检查，后续看内部字段 |
| 排查 grammar 卡队列 | 2 到 3 | grammar 未 ready 时请求不进 waiting queue，ready 后才回到调度主线 |
| 排查 batch 采样参数 | 4 到 6 | `SamplingBatchInfo` 是设备侧 batch 的采样状态，不是单请求对象的拷贝 |
| 排查 logits 与 penalty | 5 到 8 | ModelRunner 先预处理 logits，decode 准备再累计 penalty 所需的输出 token |
| 排查采样后状态 | 8 | Sampler 只产 token id；结果处理与下一轮 decode 准备共同完成状态衔接 |

读的时候先判断问题发生在“请求参数规范化、grammar readiness、batch 信息构造、logits 预处理、采样 kernel、结果接受”哪一段。

## 1. 参数先经历构造、校验和规范化

`temperature=0` 的 greedy 语义在 `__post_init__` 就完成：内部改写为 `temperature=1.0、top_k=1`。因此后面不会发生除零，也不能只检查 normalize 来解释 greedy。

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

`normalize` 会把 stop 字段整理成内部字段，并检查 tokenizer 依赖。`skip_tokenizer_init=True` 时，字符串 stop、regex stop 和 `min_new_tokens` 都不能无条件使用。

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L234-L276
    def normalize(self, tokenizer):
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len

        # Process stop regex strings
        if self.stop_regex_strs is None:
            self.stop_regex_strs = []
            self.stop_regex_max_len = 0
        else:
            if isinstance(self.stop_regex_strs, str):
                self.stop_regex_strs = [self.stop_regex_strs]

            stop_regex_max_len = 0
            for stop_regex in self.stop_regex_strs:
                stop_regex_max_len = max(
                    stop_regex_max_len, get_max_seq_length(stop_regex)
                )

            self.stop_regex_max_len = stop_regex_max_len

        # Validate tokenizer is available for tokenizer-dependent features
        raise_if_tokenizer_required(
            tokenizer, self.stop_strs, self.stop_regex_strs, self.min_new_tokens
        )

        # Clear API input aliases so omit_defaults=True drops them from the wire.
        self.stop = None
        self.stop_regex = None
        self.is_normalized = True
```

这里的关键是 alias 清理：后续传输和调度看的是内部字段，不再看原始 `stop` 或 `stop_regex`。`verify` 只互斥 `json_schema/regex/ebnf`，未把 `structural_tag` 纳入，因此后者与其他字段并存时会被 GrammarManager 的优先级选择遮蔽。

## 2. 约束请求先过 GrammarManager

Scheduler 接到请求后会调用 `process_req_with_grammar`。如果请求带 `json_schema`、`regex`、`ebnf` 或 `structural_tag`，就需要 grammar backend 参与。

```python
# 来源：python/sglang/srt/managers/scheduler.py L2248-L2250
        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)
```

当 grammar cache miss 时，请求进入 `grammar_queue`，而不是立即进入 waiting queue。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L111-L140
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

这一步解释了结构化输出的第一类延迟：请求还没开始 forward，就可能在等 grammar 编译。

## 3. Grammar 队列每轮被轮询并回到 waiting queue

Scheduler 在取新 prefill batch 前，会先检查 grammar 队列里有没有已就绪的请求。

```python
# 来源：python/sglang/srt/managers/scheduler.py L2741-L2749
    def _get_new_batch_prefill_raw(
        self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor]
    ) -> Optional[ScheduleBatch]:
        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)
```

`get_ready_grammar_requests` 同时处理 ready、failed 和 timeout。DP 并行时会同步 ready 和 failed 集合，避免不同 rank 的准入状态不一致。

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L175-L199
        # Check failed requests
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
```

如果某个 rank grammar 编译失败，最终会通过 failed union 同步失败，而不是让一部分 rank 继续跑。

## 4. Batch 级采样信息一次性构造

`SamplingBatchInfo.from_schedule_batch` 从 `ScheduleBatch.reqs` 读取每个请求的 sampling params，并搬到设备 tensor。温度被 reshape 成 `[bs, 1]`，方便和 `[bs, vocab]` logits 广播。

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L81-L123
        reqs = batch.reqs
        device = batch.device
        _pin = is_pin_memory_available(device)
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
                pin_memory=_pin,
            )
            .to(device, non_blocking=True)
            .view(-1, 1)
        )
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs],
            dtype=torch.int32,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                pin_memory=_pin,
            ).to(device, non_blocking=True)
            if enable_deterministic
            else None
        )
```

这一步之后，核心采样参数已经不再是 Python 标量，而是设备上的 batch 张量。只有开启 `enable_deterministic_inference` 才创建 `sampling_seed`；请求未给 seed 时该行填 `42`。这不是“设置 seed 就自动开启 deterministic”，方向恰好相反。

## 5. Grammar mask 和 penalty 在采样前改写 logits

`update_regex_vocab_mask` 为每个 batch 行填合法 token mask。

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L222-L247
    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask_func = None
            return

        # Find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # TODO(lianmin): Maybe we can reuse the existing mask?
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask_func = (
            first_grammar.apply_vocab_mask
        )  # force to use static method

        # Apply the mask
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(self.vocab_mask, i)

        # Move the mask to the device if needed
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)
```

然后 `apply_logits_bias` 把 overlap 预累积 penalty、非 overlap penalty、grammar mask 和 logit bias 依次施加到 logits 上。

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L266-L283
    def apply_logits_bias(self, logits: torch.Tensor):
        if self.acc_additive_penalties is not None:
            # Used in the overlap mode
            logits.add_(self.acc_additive_penalties)

        if self.acc_scaling_penalties is not None:
            # Used in the overlap mode
            apply_scaling_penalties(logits, self.acc_scaling_penalties)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

        if self.logit_bias is not None:
            logits.add_(self.logit_bias)
```

这段的顺序很重要，但还不是最终状态：`Sampler._preprocess_logits` 随后才运行 custom logit processor，并调用 NaN 检测/按环境开关清理。custom processor 能看见 penalty/mask/bias 后的 logits，也能把它们再次改写；这是结构化输出与自定义扩展组合时最危险的边界。

## 6. ModelRunner 调用 Sampler

`ModelRunner.sample` 先预处理 logits，再调用 `self.sampler` 得到 next token ids。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L3160-L3191
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids
```

这里是模型执行和采样执行的分界线：`LogitsProcessorOutput` 进入，`next_token_ids` 出来。

## 7. Sampler 先按 batch 分 greedy，再按执行模式分叉

全 greedy batch 直接走 argmax；否则不是只有一条“温度缩放 + softmax”路径：Ascend 从缩放后的 logits 进入专用实现；RL on-policy + deterministic + simple case 从 log-softmax 用 Gumbel trick 采样；其余路径才先 softmax 再交给 `_sample_from_probs`。

```python
# 来源：python/sglang/srt/layers/sampler.py L121-L145
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

非 greedy 标准路径会原地除 temperature、softmax，然后调用 `_sample_from_probs`。

```python
# 来源：python/sglang/srt/layers/sampler.py L178-L195
            else:
                # Standard path: do softmax and sample from probs.
                logits.div_(sampling_info.temperatures)

                # In-place op to save memory
                logits[:] = torch.softmax(logits, dim=-1)
                probs = logits

                batch_next_token_ids = self._sample_from_probs(
                    probs, sampling_info, positions, simple_sampling_case
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = (
                        logprobs_via_logsoftmax_kernel
                        if logprobs_via_logsoftmax_kernel is not None
                        else torch.log(probs)
                    )
                del probs
```

`_sample_from_probs` 再根据 backend 和 `top_k/top_p/min_p` 选择实现。

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

所以“top-p 是否生效”最终要看 batch need flags、执行模式、backend 和 `_sample_from_probs` 分支。simple case 即使配置 backend 为 FlashInfer，也会调用 `sampling_from_probs_torch`。

Seed 的兼容边界同样在这里暴露：deterministic inference 使 `sampling_seed` 非空；FlashInfer 的 complex top-k/top-p/min-p 路径直接 assert 不支持 seed；PyTorch complex 路径允许 top-k/top-p + seed，但只要 batch 需要 min-p，也会 assert，避免在未重归一化的过滤概率上给出错误 deterministic 结果。

Logprob 也有两个口径：默认非 greedy 路径记录温度缩放后的 softmax logprob，且不是 top-k/top-p/min-p 过滤后的重归一化分布；设置 `SGLANG_RETURN_ORIGINAL_LOGPROB` 时，则在温度缩放前缓存原始 logits 的 log-softmax。

## 8. 采样后：先提交与推进 grammar，再为下一步累计 penalty

采样出 token 不是终点。结果处理器把 token 写入 `Req.output_ids` 并推进 grammar；下一轮 decode 准备再从 Req 读取最新 token，累计到 penalizer，供随后 logits 改写。两者因果相连，但不是同一个函数完成。

```python
# 来源：python/sglang/srt/managers/scheduler_components/batch_result_processor.py L485-L497
    def _apply_prefill_grammar(self, *, req: Req, next_token_id: int) -> None:
        # FIXME: this try-except block is for handling unexpected xgrammar issue.
        try:
            req.grammar.accept_token(next_token_id)
        except ValueError as e:
            # Grammar accept_token can raise ValueError if the token is not in the grammar.
            # This can happen if the grammar is not set correctly or the token is invalid.
            logger.error(
                f"Grammar accept_token failed for req {req.rid} with token {next_token_id}: {e}"
            )
            self.abort_request(AbortReq(rid=req.rid))
        req.grammar.finished = req.finished()
```

`prepare_for_decode` 中 penalty token 会从 `Req` 里取最新输出，再累积到 orchestrator；注释特别说明 overlap 下 `batch.input_ids` 只是占位符，因此必须以 Req 为准。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2597-L2616
    def cumulate_penalty_output_tokens(self):
        # Under overlap batch.input_ids is just a placeholder here -- the
        # real token is relayed via future_map and resolved at forward
        # entry. So take the last output token from Req directly
        # (origin_input_ids[-1] on the first decode, before any output).
        last_tokens = [
            req.output_ids[-1] if len(req.output_ids) else req.origin_input_ids[-1]
            for req in self.reqs
        ]
        # Non-blocking H2D so this per-step copy doesn't sync behind the forward.
        # pin_memory (matching the prefill-path tensors) keeps the copy async;
        # is_pin_memory_available falls back to pageable on unsupported devices.
        latest_output_ids = torch.tensor(
            last_tokens,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)
        self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            latest_output_ids
        )
```

这解释了为什么 penalty 不是一次性参数，而是跨 decode step 的状态。

## 运行验证

静态验证时按三层查：

1. 请求层：`SamplingParams.verify/normalize` 后字段是否符合预期。
2. Batch 层：`SamplingBatchInfo` 中 `is_all_greedy`、need flags、`sampling_seed`、`grammars`、`logit_bias`、custom processor 是否符合请求；若发生 filter，记住 flags 不会重算。
3. 执行层：`ModelRunner._preprocess_logits`、`Sampler._preprocess_logits` 和 `Sampler.forward` 分别走了哪个分支。

动态验证至少构造六组条件：纯 greedy、greedy 与随机混 batch、普通 top-p、json_schema、带 repetition penalty、deterministic + min-p。分别记录规范化参数、batch flags/seed、两层 logits 预处理结果、backend 分支和最终 token；再过滤掉混 batch 中的随机行，确认 flags 不会自动重算。
