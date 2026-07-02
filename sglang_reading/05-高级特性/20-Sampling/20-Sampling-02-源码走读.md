---
type: batch-doc
module: 20-Sampling
batch: "20"
doc_type: walkthrough
title: "Sampling · 源码走读"
tags:
 - sglang/batch/20
 - sglang/module/sampling
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Sampling · 源码走读

> 走读顺序：`SamplingParams` → `GrammarManager`（异步编译）→ `SamplingBatchInfo`（mask/penalty）→ `ModelRunner.sample` → `Sampler`

---

## 1. sampling_params.py — 约束字段入口

### 1.1 SamplingParams 结构

**Explain：** `SamplingParams` 是 msgspec Struct，API 层传入 `json_schema`/`regex`/`ebnf`/`structural_tag` 等约束字段；`normalize()` 会把 `stop`/`stop_regex` 别名拷贝到内部字段并清空 API 别名，保证 Scheduler IPC 序列化稳定。约束字段不为空时，下游 `GrammarManager.process_req_with_grammar` 才会触发 grammar 编译队列。

**Code：**

```python
# 来源：python/sglang/srt/sampling/sampling_params.py L75-L120
class SamplingParams(msgspec.Struct, kw_only=True, omit_defaults=True):
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.sglang.io/backend/sampling_params.html
    for the documentation.
    """

    # --- API parameters (set by callers) ---
    max_new_tokens: Optional[int] = 128
    stop: Optional[Union[str, List[str]]] = (
        None  # API input alias, copied to stop_strs then cleared in normalize()
    )
    stop_token_ids: Optional[Set[int]] = None
    stop_regex: Optional[Union[str, List[str]]] = (
        None  # API input alias, copied to stop_regex_strs then cleared in normalize()
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = TOP_K_ALL
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 0
    n: int = 1
    json_schema: Optional[str] = None
    regex: Optional[str] = None
    ebnf: Optional[str] = None
    structural_tag: Optional[str] = None
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    no_stop_trim: bool = False
    custom_params: Optional[Dict[str, CustomParamValue]] = None
    stream_interval: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    sampling_seed: Optional[int] = None

    # --- Internal fields (populated by __post_init__ or normalize(), not API-facing) ---
    stop_strs: Optional[Union[str, List[str]]] = None  # from stop
    stop_regex_strs: Optional[Union[str, List[str]]] = None  # from stop_regex
    stop_str_max_len: int = 0  # set by normalize()
    stop_regex_max_len: int = 0  # set by normalize()
    is_normalized: bool = False  # set by normalize()
```

**Comment：**
- `TOP_K_ALL = 1 << 30` 表示不限制 top_k
- `custom_params.thinking_budget` 供 ReasonerGrammar 限制思考 token 数

---

## 2. base_grammar_backend.py — 后端工厂与异步编译

### 2.1 create_grammar_backend

**Explain：** Scheduler 启动时按 `--grammar-backend` 实例化 xgrammar/outlines/llguidance/none；若配置了 `reasoning_parser`，再包一层 `ReasonerGrammarBackend` 处理 `` 标签过滤。xgrammar 初始化失败且未开 strict_thinking 时会降级为 none 并打 warning。

**Code：**

```python
# 来源：python/sglang/srt/constrained/base_grammar_backend.py L223-L313
def create_grammar_backend(
    server_args: ServerArgs,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
    think_end_id: Optional[int] = None,
) -> Optional[BaseGrammarBackend]:
    name = server_args.grammar_backend

    # Custom grammar backend has the highest priority
    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            server_args, tokenizer, vocab_size, eos_token_ids
        )

    # Default grammar backends
    if name == "outlines":
        from sglang.srt.constrained.outlines_backend import OutlinesGrammarBackend

        grammar_backend = OutlinesGrammarBackend(
            tokenizer,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "xgrammar":
        from sglang.srt.constrained.xgrammar_backend import (
            TokenizerNotSupportedError,
            XGrammarGrammarBackend,
        )

        # Convert Set[int] to List[int] if needed
        eos_list = list(eos_token_ids) if eos_token_ids else None

        try:
            grammar_backend = XGrammarGrammarBackend(
                tokenizer,
                vocab_size=vocab_size,
                model_eos_token_ids=eos_list,
                any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            )
        except TokenizerNotSupportedError as e:
            if server_args.enable_strict_thinking:
                raise ValueError(
                    f"--enable-strict-thinking requires a grammar backend with "
                    f"token filtering support, but XGrammar failed to initialize: "
                    f"{e}. Cannot fall back to grammar_backend='none' with strict "
                    f"thinking enabled."
                ) from e
            logger.warning(
                f"Grammar backend disabled because tokenizer is not supported by XGrammar: {e}. "
                "Falling back to grammar_backend='none'. "
                "Structured outputs (JSON schema, regex, EBNF) will not be available."
            )
            server_args.grammar_backend = "none"
            return None
    elif name == "llguidance":
        from sglang.srt.constrained.llguidance_backend import GuidanceBackend

        grammar_backend = GuidanceBackend(
            tokenizer=tokenizer,
            any_whitespace=not server_args.constrained_json_disable_any_whitespace,
            whitespace_pattern=server_args.constrained_json_whitespace_pattern,
        )
    elif name == "none":
        if server_args.enable_strict_thinking:
            raise ValueError(
                "--enable-strict-thinking requires a grammar backend that supports "
                "token filtering, but grammar_backend='none' was specified. Use "
                "--grammar-backend xgrammar or another backend that supports token "
                "filtering."
            )
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    if server_args.reasoning_parser and think_end_id is not None:
        from sglang.srt.constrained.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        reasoning_parser = ReasoningParser(
            model_type=server_args.reasoning_parser, stream_reasoning=False
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend,
            reasoning_parser,
            tokenizer,
            enable_strict_thinking=server_args.enable_strict_thinking,
        )

    return grammar_backend
```

**Comment：**
- `GRAMMAR_BACKEND_REGISTRY` 支持插件式自定义 backend
- strict_thinking 强制要求 token filter 能力，不能与 none 共存

### 2.2 get_cached_or_future_value — 异步编译入口

**Explain：** 每个约束 key `(type, string)` 先查 `cache`；命中则 `copy()` 并 `maybe_init_reasoning`；未命中则 `ThreadPoolExecutor.submit(_init_value_dispatch)` 返回 `Future`，由 GrammarManager 放入 `grammar_queue` 轮询。编译完成后 `set_cache` 写入缓存供后续请求复用。

**Code：**

```python
# 来源：python/sglang/srt/constrained/base_grammar_backend.py L178-L210
    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        s = time.perf_counter()
        key_type, key_string = key
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        return grammar

    def get_cached_or_future_value(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Tuple[BaseGrammarObject | Future[BaseGrammarObject], bool]:
        value = self.cache.get(key)
        if value:
            copied_value = value.copy()
            copied_value.maybe_init_reasoning(require_reasoning)
            return copied_value, True
        value = self.executor.submit(self._init_value_dispatch, key, require_reasoning)
        return value, False

    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        self.cache[key] = value
```

---

## 3. grammar_manager.py — 编译队列与 DP 同步

### 3.1 process_req_with_grammar

**Explain：** 请求入队时检测四类约束字段；`grammar_backend=None`（即 `--grammar-backend none`）直接 abort。否则构造 cache key，cache miss 时 `req.grammar` 暂存为 `Future` 并 append 到 `grammar_queue`；cache hit 且为 `InvalidGrammarObject` 同样 abort。`enable_strict_thinking` 且无显式约束时，会初始化纯 reasoning 过滤 grammar。

**Code：**

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

### 3.2 get_ready_grammar_requests — 轮询 + all_gather

**Explain：** Scheduler 每轮调用此函数，在 `SGLANG_GRAMMAR_POLL_INTERVAL` 内 poll `Future.done()`；超时 `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` 次则 cancel 并缓存 `InvalidGrammarObject`。多 rank 时 `all_gather_object` 取 ready 交集、failed 并集，保证 DP 组内 grammar 就绪状态一致后才移入 waiting_queue。

**Code：**

```python
# 来源：python/sglang/srt/constrained/grammar_manager.py L142-L243
    def get_ready_grammar_requests(self) -> List[Req]:
        """
        Move requests whose grammar objects are ready from grammar_queue to waiting_queue.

        Rank i returns two sets ready_reqs_i, failed_reqs_i
        ready_reqs_all = all_gather(ready_reqs_i)
        failed_reqs_all = all_gather(failed_reqs_i)

        ready_reqs = intersect(ready_reqs_all)
        failed_reqs = union(failed_reqs_all)
        """
        assert self.grammar_backend
        ready_req_idxs: set[int] = set()
        failed_req_idxs: set[int] = set()

        # Poll for ready requests
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < self.SGLANG_GRAMMAR_POLL_INTERVAL:
            for i, req in enumerate(self.grammar_queue):
                if i in ready_req_idxs:
                    continue

                if req.finished() or req.grammar is None:  # It is aborted by AbortReq
                    ready_req_idxs.add(i)
                    continue

                assert isinstance(req.grammar, futures.Future), f"{req=}"
                if req.grammar.done():
                    ready_req_idxs.add(i)

            # Sleep a bit to avoid busy waiting
            time.sleep(self.SGLANG_GRAMMAR_POLL_INTERVAL / 10)

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

        # Remove finished requests from grammar_queue
        self.grammar_queue = [
            req
            for i, req in enumerate(self.grammar_queue)
            if i not in synced_ready_req_idxs and i not in synced_failed_req_idxs
        ]
        return return_reqs
```

**Comment：**
- `_apply_request_reasoning_budget` 读取 `custom_params.thinking_budget` 设置 `max_think_tokens`
- abort 时 cancel 未完成的 Future，避免线程泄漏

---

## 4. sampling_batch_info.py — 批量化与 mask 构建

### 4.1 from_schedule_batch

**Explain：** 每个 decode step 从 `ScheduleBatch.reqs` 收集 per-req 采样参数，pin_memory + non_blocking H2D 搬到 GPU；同时收集 `grammars` 列表（每个 req 的 grammar object 或 None）。`is_all_greedy` 等 flag 用于 Sampler 短路，避免无谓 top_p kernel。

**Code：**

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L76-L145
    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        global_server_args = get_global_server_args()
        enable_deterministic = global_server_args.enable_deterministic_inference

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

        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    for key, value in r.sampling_params.logit_bias.items():
                        logit_bias[i, int(key)] = value

        # Check if any request has custom logit processor
        has_custom_logit_processor = (
            global_server_args.enable_custom_logit_processor
            and any(r.custom_logit_processor for r in reqs)  # check the flag first.
        )  # then check the requests.

        if has_custom_logit_processor:
            # Merge the same type of custom logit processors together
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
```

### 4.2 update_regex_vocab_mask + apply_logits_bias

**Explain：** 采样前为 batch 分配 `[bs, vocab_size]` 的 bitmask；每个 grammar 调用 `fill_vocab_mask` 写入合法 token。`apply_logits_bias` 按序施加 additive/scaling penalty → grammar mask → logit_bias；mask 通过 backend 的 `apply_vocab_mask` 将非法 token logits 置 -inf。

**Code：**

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py L222-L283
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

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            self.acc_additive_penalties = torch.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            self.penalizer_orchestrator.accumulate_additive_penalties(
                self.acc_additive_penalties
            )
            self.acc_scaling_penalties = (
                self.penalizer_orchestrator.accumulate_scaling_penalties()
            )
        else:
            self.acc_additive_penalties = None
            self.acc_scaling_penalties = None

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

---

## 5. xgrammar_backend.py — apply_mask 实现

**Explain：** xgrammar 默认走 Triton `apply_token_bitmask_inplace_triton`（CUDA）；ROCm 用 CUDA kernel。与 outlines 的 `masked_fill_` 相比，bitmask kernel 对大 vocab 更高效。采样后 grammar object 通过 `accept_token` 推进状态，下一步 decode 重新 `fill_vocab_mask`。

**Code：**

```python
# 来源：python/sglang/srt/constrained/xgrammar_backend.py L238-L245
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type in {"cuda", "npu", "xpu", "musa"}:
            if _is_hip:
                apply_token_bitmask_inplace_cuda(logits, vocab_mask)
            else:
                apply_token_bitmask_inplace_triton(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")
```

---

## 6. model_runner.py — sample 主链路

### 6.1 _preprocess_logits

**Explain：** forward 产出 logits 后、Sampler 之前调用；先 `update_regex_vocab_mask` 再 `apply_logits_bias`。mask 用完后立即 `vocab_mask = None` 释放 GPU 显存，防止 overlap scheduling 下 closure 持有 mask 导致 structured output 场景 VRAM 泄漏。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L3143-L3191
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

---

## 7. sampler.py — top_p 采样

### 7.1 forward 分支

**Explain：** `is_all_greedy=True` 时直接 argmax；否则 logits 除以 temperature 后 softmax 得 probs，再调 `_sample_from_probs`。FlashInfer 的 `top_k_top_p_sampling_from_probs` 融合 top_k + top_p 重归一化与采样；采样完成后 TP group 可选 sync token id。

**Code：**

```python
# 来源：python/sglang/srt/layers/sampler.py L121-L212
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
            if self.rl_on_policy_target is not None:
                # TODO: use more inplace ops to save memory
                logits_div_temperature = (
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(
                    logits_div_temperature, dim=-1
                )
                del logits_div_temperature

            if self.use_ascend_backend:
                # Ascend backend: sample from logits directly.
                batch_next_token_ids, logprobs = self._forward_ascend_backend(
                    logits,
                    sampling_info,
                    simple_sampling_case,
                    return_logprob,
                    positions,
                )
            elif (
                self.use_log_softmax_logprob
                and self.enable_deterministic
                and simple_sampling_case
            ):
                # RL on-policy path: sample from logprobs to match the trainer.
                batch_next_token_ids = self._sample_from_logprobs(
                    logprobs_via_logsoftmax_kernel,
                    sampling_info,
                    positions,
                )
                if return_logprob and not SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = logprobs_via_logsoftmax_kernel
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

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if SGLANG_RETURN_ORIGINAL_LOGPROB:
                logprobs = original_logprobs
            self._attach_logprobs_to_output(
                logits_output,
                logprobs,
                top_logprobs_nums,
                token_ids_logprobs,
                sampling_info,
                batch_next_token_ids,
            )

        self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)

        return batch_next_token_ids
```

---

## 8. penaltylib/orchestrator.py — Penalty 编排

**Explain：** `BatchedPenalizerOrchestrator` 在构造时 lazy `prepare_if_required` 各 penalizer（frequency/presence/repetition/min_new_tokens）；`apply` 按注册顺序 in-place 修改 logits。speculative decoding 时 `repeat` 参数将 per-req penalty 按 draft token layout 扩展。

**Code：**

```python
# 来源：python/sglang/srt/sampling/penaltylib/orchestrator.py L13-L70
class BatchedPenalizerOrchestrator:
    def __init__(
        self,
        vocab_size: int,
        batch: ScheduleBatch,
        penalizers: Set[Type[_BatchedPenalizer]],
    ):
        self.vocab_size = vocab_size
        self._batch_ref = weakref.ref(batch)
        self.device = batch.device
        self.penalizers = {Penalizer: Penalizer(self) for Penalizer in penalizers}

        is_required = False
        for penalizer in self.penalizers.values():
            pen_is_required = penalizer.prepare_if_required()
            is_required |= pen_is_required
        self.is_required = is_required

    @property
    def batch(self) -> ScheduleBatch | None:
        return self._batch_ref()

    @batch.setter
    def batch(self, value: Optional[ScheduleBatch]):
        if value is None:
            self._batch_ref = lambda: None
        else:
            self._batch_ref = weakref.ref(value)

    def reqs(self):
        return self.batch.reqs

    def cumulate_output_tokens(self, output_ids: torch.Tensor):
        """
        Feed the output tokens to the penalizers.

        Args:
            output_ids (torch.Tensor): The output tokens.
        """
        for penalizer in self.penalizers.values():
            penalizer.cumulate_output_tokens(output_ids=output_ids)

    def apply(self, logits: torch.Tensor, repeat: Optional[int] = None):
        """
        Apply all penalizers to the logits in-place.

        Args:
            logits: The logits tensor to apply penalties to.
            repeat: If set (speculative decoding), per-request penalties are
                expanded via repeat_interleave to match the draft token layout.
                Additive penalties are captured into a zeros tensor, expanded,
                then added; scaling penalties are accumulated, expanded, then
                applied directly.
        """
        if repeat is None:
            for penalizer in self.penalizers.values():
                penalizer.apply(logits)
        else:
```

**Comment：**
- penalty 在 grammar mask **之前**施加（见 `apply_logits_bias` 顺序）
- weakref 持有 ScheduleBatch，避免循环引用
