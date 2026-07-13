---
title: "OpenAI-API · 源码走读"
type: walkthrough
framework: sglang
topic: "OpenAI-API"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-11
---
# OpenAI-API · 源码走读

## 场景主线

这篇只追一个场景：业务用 OpenAI SDK 发起 `POST /v1/chat/completions`，`stream=true`，SGLang 返回一串 SSE chunk。源码主线是：

```text
http_server route
  -> OpenAIServingChat.handle_request
  -> _convert_to_internal_request
  -> TokenizerManager.generate_request
  -> _generate_chat_stream
  -> _generate_stream_content
  -> build_sse_content
```

读这条线时不要把 endpoint 当入口终点。endpoint 只是门牌；真正的协议转换在 serving handler，真正的生成边界在 TokenizerManager。

## 心理模型

OpenAI API 层像一个双向翻译器：

| 方向 | 输入 | 输出 | 核心风险 |
|------|------|------|----------|
| 入站 | OpenAI/Ollama JSON | `GenerateReqInput` / `EmbeddingReqInput` | 字段语义丢失、模板处理错误 |
| 出站 | 默认累积 chunk，或 incremental 模式的分段 chunk | OpenAI SSE / Ollama NDJSON | 上下游 chunk 语义不匹配、delta 重复、reasoning/tool call 状态错位 |

## 长文读法

这篇按协议翻译边界读：FastAPI route 只是把请求交给 `app.state` 上的 serving handler，`OpenAIServingBase` 统一做校验、日志、内部请求转换和 stream 分支，chat/completion handler 再分别处理模板、sampling、logprob、tool/reasoning 状态，真正进入生成系统的边界仍是 `TokenizerManager.generate_request`。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 首次建立 OpenAI 兼容主线 | 场景主线、心理模型、1 到 2 | endpoint 不是逻辑中心，serving handler 才是协议转换层 |
| 排查 completion 字段映射 | 3 到 4 | OpenAI prompt、logprobs、response_format、LoRA、routing header 都被压进 `GenerateReqInput` 或 sampling params |
| 排查 chat 模板和工具调用 | 5 | `_process_messages` 决定 chat template、input_ids、reasoning、tool constraint 和 multimodal 入口 |
| 确认请求何时进入 runtime | 6 | `TokenizerManager.generate_request` 才开始 normalize、LoRA 校验、tokenize 和 scheduler 发送 |
| 排查 stream 报错时机 | 7 | handler 先 kick-start generator，能在 HTTP 200 前把验证错误转成普通错误响应 |
| 排查 SSE delta、reasoning、tool call、usage | 8 到 9 | chat stream 是多状态表，`build_sse_content` 只负责最终 OpenAI chunk 外形 |
| 区分 Ollama 和 OpenAI | 10、运行验证 | Ollama 复用 tokenizer/runtime，但输出是 NDJSON，逻辑不是 OpenAI handler 子类 |

读的时候保持两次转换分开：入站把 OpenAI/Ollama 请求改写为内部请求，出站把内部 chunk 改写为外部协议形状。`TokenizerManager` 中间的生成语义不属于 OpenAI 层私有逻辑。

下面按调用顺序读。

## 源码证据

### 1. 路由只委托 handler

`http_server.py` 的 OpenAI route 只把 Pydantic 请求交给 `app.state` 上的 handler。这里没有 prompt 模板、sampling、stream delta 逻辑。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1598-L1613
@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

因此排查 `/v1/chat/completions` 行为时，路由只能确认“请求进了哪个 handler”，不能解释输出内容为什么长那样。

### 2. `OpenAIServingBase` 固定四步模板

所有 OpenAI handler 共享同一个入站模板：校验、记录、转换、按 stream 分支。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_base.py L73-L109
    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Union[Any, StreamingResponse, ErrorResponse]:
        """Handle the specific request type with common pattern
        If you want to override this method, you should be careful to record the validation time.
        """
        received_time = monotonic_time()

        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Log the raw OpenAI request payload before conversion to tokenized form.
            request_logger = self.tokenizer_manager.request_logger
            if request_logger.log_requests and request_logger.log_requests_level >= 2:
                request_logger.log_openai_received_request(request, request=raw_request)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(
                request, raw_request
            )

            if isinstance(adapted_request, (GenerateReqInput, EmbeddingReqInput)):
                # Only set timing fields if adapted_request supports them
                adapted_request.received_time = received_time

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )
```

这个模板解释了一个常见现象：stream 和 non-stream 的上游转换相同，差异主要在输出包装和错误时机。

### 3. Completion 把 OpenAI 字段压成 `GenerateReqInput`

Completion 是最直接的翻译样本：prompt 转成 text 或 input ids，sampling 字段重建为内部采样参数，LoRA、DP rank、routing key 等 SGLang 扩展同时进入 `GenerateReqInput`。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_completions.py L65-L135
    def _convert_to_internal_request(
        self,
        request: CompletionRequest,
        raw_request: Request = None,
    ) -> tuple[GenerateReqInput, CompletionRequest]:
        """Convert OpenAI completion request to internal format"""
        # NOTE: with openai API, the prompt's logprobs are always not computed
        if request.echo and request.logprobs:
            logger.warning(
                "Echo is not compatible with logprobs. "
                "To compute logprobs of input prompt, please use the native /generate API."
            )
        # Process prompt
        prompt = request.prompt
        if self.template_manager.completion_template_name is not None:
            prompt = generate_completion_prompt_from_request(request)

        # Set logprob start length based on echo and logprobs
        if request.echo and request.logprobs:
            logprob_start_len = 0
        else:
            logprob_start_len = -1

        # Build sampling parameters
        sampling_params = self._build_sampling_params(request)

        # Determine prompt format
        if isinstance(prompt, str) or (
            isinstance(prompt, list) and isinstance(prompt[0], str)
        ):
            prompt_kwargs = {"text": prompt}
        else:
            prompt_kwargs = {"input_ids": prompt}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            sampling_params=sampling_params,
            return_logprob=request.logprobs is not None,
            top_logprobs_num=request.logprobs if request.logprobs is not None else 0,
            logprob_start_len=logprob_start_len,
            return_text_in_logprobs=True,
            stream=request.stream,
            lora_path=lora_path,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            routed_dp_rank=effective_routed_dp_rank,
            disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
            return_hidden_states=request.return_hidden_states,
            return_routed_experts=request.return_routed_experts,
            routed_experts_start_len=request.routed_experts_start_len,
            rid=request.rid,
            session_id=request.session_id,
            extra_key=self._compute_extra_key(request),
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
        )
```

注意 `echo + logprobs` 的 warning。它说明 OpenAI 兼容语义和 SGLang 原生 `/generate` 的能力并不是完全等价。

### 4. Sampling 参数在兼容层重建

OpenAI 请求里的 `max_tokens`、`temperature`、`response_format` 等不会原样传给 Scheduler，而是被组装成内部 sampling 参数。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_completions.py L139-L185
    def _build_sampling_params(self, request: CompletionRequest) -> Dict[str, Any]:
        """Build sampling parameters for the request"""
        # Start with common parameters
        sampling_params = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens,
            "min_new_tokens": request.min_tokens,
            "stop": request.stop,
            "stop_token_ids": request.stop_token_ids,
            "stop_regex": request.stop_regex,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repetition_penalty,
            "regex": request.regex,
            "json_schema": request.json_schema,
            "ebnf": request.ebnf,
            "n": request.n,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
            "logit_bias": request.logit_bias,
            "custom_params": request.custom_params,
            "sampling_seed": request.seed,
        }

        # Handle response_format constraints
        if request.response_format and request.response_format.type == "json_schema":
            json_schema = request.response_format.json_schema
            schema = getattr(json_schema, "schema_", None)
            if schema is None:
                raise ValueError(
                    "schema_ is required for json_schema response format request."
                )
            sampling_params["json_schema"] = convert_json_schema_to_str(schema)
        elif request.response_format and request.response_format.type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif (
            request.response_format and request.response_format.type == "structural_tag"
        ):
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                request.response_format.model_dump(by_alias=True)
            )

        return sampling_params
```

排查结构化输出时，先看这里有没有把外部 `response_format` 翻译成内部约束。

### 5. Chat 的复杂性来自 `_process_messages`

Chat 不是把 `messages` 简单 join 成字符串。工具、reasoning、模板、stop、multimodal 都在 `_process_messages` 中形成 `MessageProcessingResult`。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L626-L695
    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""
        # GptOss model needs to keep special tokens for harmony parsing
        if self.is_gpt_oss or self.is_gemma4:
            request.skip_special_tokens = False

        self._patch_reasoning_skip_special_tokens(request)

        thinking_mode = self._get_reasoning_from_request(request)
        # SGLang's ReasonerGrammarBackend owns the reasoning prefix
        # when --reasoning-parser is configured, so builtin xgrammar
        # tags must describe only the post-reasoning tool-call suffix.
        xgrammar_reasoning = thinking_mode and (
            self.tokenizer_manager.server_args.reasoning_parser is None
        )
        tool_call_constraint = None

        # Apply chat template and its stop strings
        tools = None
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False
            if not isinstance(request.tool_choice, str):
                tools = [
                    item.model_dump()
                    for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                tools = [item.model_dump() for item in request.tools]
            if self.tool_call_parser:
                parser = FunctionCallParser(request.tools, self.tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                    thinking_mode=xgrammar_reasoning,
                )
            # Fallback: use generic JSON schema for required/named tool choice
            # only when no parser-specific constraint was set
            if tool_call_constraint is None and (
                request.tool_choice == "required"
                or isinstance(request.tool_choice, ToolChoice)
            ):
                json_schema = get_json_schema_constraint(
                    request.tools,
                    request.tool_choice,
                    parallel_tool_calls=request.parallel_tool_calls,
                )
                tool_call_constraint = ("json_schema", json_schema)

        # When input_ids are provided, skip template tokenization entirely;
        # only stop tokens and tool_call_constraint are needed.
        if request.input_ids is not None:
            result = MessageProcessingResult(
                prompt="",
                prompt_ids=request.input_ids,
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=[],
                stop=request.stop or [],
            )
        elif self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result
```

如果 Chat 输出格式错，不要先看 `TokenizerManager`；先看这一步到底给模型喂了什么 prompt 和约束。还要区分“约束生成形状”和“解析外部响应”两件事：fallback JSON schema 可以约束 required/named tool choice，但当前 stream/non-stream 的 tool-call 解析入口仍以 `self.tool_call_parser` 为门槛；没有 parser 时，JSON 可能留在普通 content。

### 6. TokenizerManager 是真正进入生成系统的边界

Serving 层完成协议转换后，生成请求进入 `TokenizerManager.generate_request`。这里开始出现 normalize、priority、pause、LoRA resolve、tokenize、send scheduler、wait response。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L589-L646
    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()

        # Normalize the request
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)

        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                logger.debug(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        self._init_req_state(obj, request)
        try:
            if self.server_args.language_only:
                self._handle_epd_disaggregation_encode_request(obj)

            # Log the request
            self.request_logger.log_received_request(obj, self.tokenizer, request)

            async with self.is_pause_cond:
                await self.is_pause_cond.wait_for(lambda: not self.is_pause)

            async with self.model_update_lock.reader_lock:
                await self._validate_and_resolve_lora(obj)

                # Tokenize the request and send it to the scheduler
                if obj.is_single:
                    tokenized_obj = await self._tokenize_one_request(obj)
                    state = self.rid_to_state[obj.rid]
                    if obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_obj.input_ids)
                    self._send_one_request(tokenized_obj)
                    async for response in self._wait_one_response(obj, request):
                        yield response
                else:
                    async for response in self._handle_batch_request(obj, request):
                        yield response
        except Exception:
            # _init_req_state created a rid_to_state entry per (sub-)request up
            # front. The normal remover is the scheduler-response path
            # (_handle_batch_output), so a failure *before* a request reaches the
            # scheduler -- e.g. input-length validation rejecting an over-context
            # request -- would otherwise leak those entries forever. Drop any that
            # are still pending; entries already removed on the normal completion
            # path are left untouched (pop is a no-op).
            self._discard_pending_req_states(obj)
            raise
```

这段是阅读边界：OpenAI 兼容层到此为止，后面进入 tokenization、scheduler、model worker。

### 7. Stream 先 kick-start，再返回 HTTP 200

Chat stream 会先拉出第一个 chunk。这样上下文过长等 `ValueError` 还能以 HTTP error 返回，而不是已经发出 200 后再塞进 SSE。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L975-L1001
    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming chat completion request"""
        generator = self._generate_chat_stream(adapted_request, request, raw_request)

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        # If validation fails (e.g., context length exceeded), we can still return
        # a proper HTTP 400 error response instead of streaming it as SSE payload.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )
```

Completion stream 也采用同样模式。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_completions.py L187-L213
    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: CompletionRequest,
        raw_request: Request,
    ) -> Union[StreamingResponse, ErrorResponse]:
        """Handle streaming completion request"""
        generator = self._generate_completion_stream(
            adapted_request, request, raw_request
        )

        # Kick-start the generator to trigger validation before HTTP 200 is sent.
        try:
            first_chunk = await generator.__anext__()
        except ValueError as e:
            return self.create_error_response(str(e))

        async def prepend_first_chunk():
            yield first_chunk
            async for chunk in generator:
                yield chunk

        return StreamingResponse(
            prepend_first_chunk(),
            media_type="text/event-stream",
            background=self.tokenizer_manager.create_abort_task(adapted_request),
        )
```

所以“为什么 stream 请求有时返回 JSON error 而不是 SSE error”不是偶然，而是这个 kick-start 设计。首包之后的错误语义还要继续分：显式带 `HTTPStatus` 的 abort 会变成 streaming error；Chat 主循环只捕获 `ValueError`，其他 parser/runtime 异常可能直接中断连接，而 Completion 主循环捕获一般 `Exception` 后会尽量发送 error chunk。

### 8. Chat stream 是多状态表

Chat 的 `_generate_stream_content` 同时维护 text offset、reasoning parser、tool parser、logprobs flush 和 usage。它不是简单的 `yield text`。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L339-L461
    async def _generate_stream_content(
        self,
        content: Dict[str, Any],
        index: int,
        request: ChatCompletionRequest,
        stream_offsets: Dict[int, int],
        reasoning_parser_dict: Dict,
        parser_dict: Dict,
        has_tool_calls: Dict[int, bool],
        choice_logprobs: Optional[Dict],
        finish_reason_type: Optional[str],
        continuous_usage_stats: bool,
        prompt_tokens: Dict[int, int],
        reasoning_tokens: Dict[int, int],
        completion_tokens: Dict[int, int],
    ) -> AsyncGenerator[str, None]:
        """Generate SSE chunks for streaming content."""
        offset = stream_offsets.get(index, 0)
        if self.tokenizer_manager.server_args.incremental_streaming_output:
            delta = content["text"]
        else:
            delta = content["text"][offset:]
            stream_offsets[index] = len(content["text"])

        # Attach logprobs to the first chunk emitted this step (reasoning,
        # tool-call, or content) so they aren't dropped when a parser is active
        # nor duplicated across chunks; flush any leftover at the end.
        remaining_logprobs = choice_logprobs

        # Handle reasoning content
        if self.reasoning_parser and request.separate_reasoning:
            reasoning_text, delta = self._process_reasoning_stream(
                index, delta, reasoning_parser_dict, content, request
            )
            if reasoning_text:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    reasoning_content=reasoning_text,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None

        # Handle tool calls
        if request.tool_choice != "none" and request.tools and self.tool_call_parser:
            async for chunk in self._process_tool_call_stream(
                index,
                delta,
                parser_dict,
                content,
                request,
                has_tool_calls,
                continuous_usage_stats,
            ):
                if chunk:
                    yield chunk

            # Send any remaining tool call arguments when generation finishes
            if finish_reason_type is not None and index in parser_dict:
                parser = parser_dict[index]
                remaining_chunk = self._check_for_unstreamed_tool_args(
                    parser, content, request, index
                )
                if remaining_chunk:
                    yield remaining_chunk

        else:
            # Regular content
            if delta:
                usage = None
                if continuous_usage_stats:
                    usage = UsageProcessor.calculate_token_usage(
                        prompt_tokens=prompt_tokens.get(index, 0),
                        reasoning_tokens=reasoning_tokens.get(index, 0),
                        completion_tokens=completion_tokens.get(index, 0),
                    ).model_dump()

                yield build_sse_content(
                    chunk_id=content["meta_info"]["id"],
                    created=int(time.time()),
                    model=request.model,
                    index=index,
                    content=delta,
                    logprobs=remaining_logprobs,
                    usage=usage,
                )
                remaining_logprobs = None

        # Flush logprobs still unattached this step — only when a parser is
        # active, since _process_tool_call_stream may consume the delta and emit
        # no content chunk. On the plain path an empty-delta step has no chunk
        # to attach to either way, and a standalone empty-delta logprobs chunk
        # is not a shape clients expect.
        if remaining_logprobs is not None and (
            self.reasoning_parser or self.tool_call_parser
        ):
            usage = None
            if continuous_usage_stats:
                usage = UsageProcessor.calculate_token_usage(
                    prompt_tokens=prompt_tokens.get(index, 0),
                    reasoning_tokens=reasoning_tokens.get(index, 0),
                    completion_tokens=completion_tokens.get(index, 0),
                ).model_dump()

            yield build_sse_content(
                chunk_id=content["meta_info"]["id"],
                created=int(time.time()),
                model=request.model,
                index=index,
                logprobs=remaining_logprobs,
                usage=usage,
            )
```

这里是排查 stream 形状的第一现场：`stream_offsets` 控制普通文本，`reasoning_parser_dict` 控制思考内容，`parser_dict` 控制工具调用。还要注意源码注释表达的是“logprobs 挂到本 step 第一类输出”的目标，但 `_process_tool_call_stream` 本身没有接收 `remaining_logprobs`；纯 tool-call step 的 logprobs 会在后面的 flush 分支中形成独立、content 为空的 chunk，而不是附着到 tool-call chunk。客户端若假设每个 logprob chunk 都有正文或 tool call，会误判为空包。

### 9. SSE helper 固定外部形状

Chat stream 最终调用 `build_sse_content`，用 `msgspec` 构造 OpenAI 风格 chunk。

```python
# 来源：python/sglang/srt/entrypoints/openai/sse_utils.py L83-L99
    delta = StreamDelta(role=role, content=content, reasoning_content=reasoning_content)
    choice = StreamChoice(
        index=index,
        delta=delta,
        logprobs=logprobs,
        finish_reason=finish_reason,
        matched_stop=matched_stop,
    )
    chunk = StreamChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[choice],
        usage=usage,
    )
    return (_SSE_DATA_B + _stream_encoder.encode(chunk) + _SSE_NL_B).decode()
```

如果 SDK 解析失败，要先确认 `data: ` 前缀、双换行、`choices[].delta` 结构是否还符合这个出口。

### 10. Ollama 是平行适配，不是 OpenAI handler 的子类

Ollama chat 直接用 tokenizer 的 chat template，构造 `GenerateReqInput`，然后按 NDJSON 流式返回。

```python
# 来源：python/sglang/srt/entrypoints/ollama/serving.py L68-L103
    async def handle_chat(
        self, request: OllamaChatRequest, raw_request: Request
    ) -> Union[OllamaChatResponse, StreamingResponse]:
        """Handle /api/chat endpoint."""
        model_name = self.tokenizer_manager.served_model_name

        # Convert messages to SGLang format
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Apply chat template using tokenizer
        prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Convert options to sampling params
        sampling_params = self._convert_options_to_sampling_params(request.options)

        # Create SGLang request with input_ids
        gen_request = GenerateReqInput(
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            stream=request.stream,
        )

        if request.stream:
            return await self._stream_chat_response(
                gen_request, raw_request, model_name
            )
        else:
            return await self._generate_chat_response(
                gen_request, raw_request, model_name
            )
```

Ollama stream 当前无条件把内部 text 当作累积快照再切 delta，外层协议是 NDJSON。默认 `incremental_streaming_output=False` 时这与 TokenizerManager 匹配；若启用 `--incremental-streaming-output`，内部 `text` 已是分段 delta，这段 `previous_text` 切片就可能漏字或产出空片段，需要单独验证或修复 adapter。

```python
# 来源：python/sglang/srt/entrypoints/ollama/serving.py L145-L164
                # Calculate delta (new text since last chunk)
                delta = text[len(previous_text) :]
                previous_text = text

                if is_done:
                    # Final chunk
                    response = OllamaChatStreamResponse(
                        model=model_name,
                        created_at=self._get_timestamp(),
                        message=OllamaMessage(role="assistant", content=""),
                        done=True,
                        done_reason="stop",
                    )
                else:
                    response = OllamaChatStreamResponse(
                        model=model_name,
                        created_at=self._get_timestamp(),
                        message=OllamaMessage(role="assistant", content=delta),
                        done=False,
                    )
```

```python
# 来源：python/sglang/srt/entrypoints/ollama/serving.py L168-L171
        return StreamingResponse(
            generate_stream(),
            media_type="application/x-ndjson",
        )
```

这说明“上游 chunk 究竟是累积态还是分段态”才是共同问题；`SSE vs NDJSON` 只是外层包装差异。Chat 已显式适配两种模式，Completion 与 Ollama 当前没有相同分支。Ollama 还有两层协议损失：终态 chunk 固定发空 content，当前轮新 delta 若只在终态到达可能被丢弃；`done_reason` 也硬编码为 `stop`，不会保留内部 `length` 或 `abort` 等真实 finish reason。

## 运行验证

本专题的验证不要求启动完整 GPU serving 才能开始。静态检查可以先做三件事：

1. 在 `serving_chat.py` 中确认 `_convert_to_internal_request` 是否把目标字段写入 `GenerateReqInput`。
2. 在 `_generate_stream_content` 中确认目标字段属于普通 content、reasoning 还是 tool call。
3. 在 `TokenizerManager.generate_request` 中确认请求已经进入 tokenization 和 scheduler 边界。

如果能启动服务，再用 OpenAI SDK 做三组请求：普通 streaming chat、带 `tools` 的 streaming chat、带 `stream_options.include_usage` 的 streaming chat。观察首包 role-only chunk、正文 delta、usage chunk 和 `[DONE]` 顺序。

## 复盘迁移

读完这条主线后，其他 endpoint 都可以用同一个问题拆开：

| 问题 | 去哪里找 |
|------|----------|
| 外部字段在哪里定义 | `protocol.py` |
| 外部字段在哪里变成内部字段 | 对应 serving handler 的 `_convert_to_internal_request` |
| 生成什么时候真正开始 | `TokenizerManager.generate_request` |
| 流式形状在哪里生成 | handler 的 `_generate_*_stream` 或 `sse_utils.py` |
| 客户端断连如何回收 | `create_abort_task` 和 background task |
