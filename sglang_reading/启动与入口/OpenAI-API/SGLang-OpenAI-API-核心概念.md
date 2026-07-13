---
title: "OpenAI-API · 核心概念"
type: concept
framework: sglang
topic: "OpenAI-API"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# OpenAI-API · 核心概念

## 先回答为什么读

OpenAI 兼容层最容易被误读成“FastAPI wrapper”。实际它承担的是协议转换和流式状态维护：它把客户端字段、SGLang 扩展字段、模板处理、tool/reasoning 约束、usage 统计和连接中断处理都收束到内部请求边界。

读完本篇，你应该能判断一个 bug 属于哪一层：

| 症状 | 更可能的入口 |
|------|--------------|
| OpenAI SDK 字段校验失败 | `protocol.py` 和 `OpenAIServingBase._validate_request` |
| prompt 内容与预期不一致 | Chat `_process_messages` 或 Completion template |
| stream 文本重复或漏字 | `stream_offsets` 和 `incremental_streaming_output` |
| tool call 或 reasoning chunk 形状不对 | Chat stream parser 状态 |
| DP rank、LoRA、routing key 不生效 | `OpenAIServingBase` 到 `GenerateReqInput` 的转换字段 |

## 心理模型：协议海关

兼容层有三道关：

1. 外部契约：Pydantic model 接收 OpenAI/Ollama 形状。
2. 内部契约：handler 转成 `GenerateReqInput` 或 `EmbeddingReqInput`。
3. 输出契约：内部 chunk 被还原成 OpenAI SSE 或 Ollama NDJSON。

`OpenAIServingBase.handle_request` 是 OpenAI 路径的总模板。

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

这段的读法是：`_validate_request` 还在外部协议层，`_convert_to_internal_request` 是边界线，stream/non-stream 是输出协议层。真正执行生成不在这里。

## 外部契约不是内部行为

`CompletionRequest` 和 `ChatCompletionRequest` 的字段顺序对齐 OpenAI API 文档，但字段存在不等于 Scheduler 原生理解这些字段。handler 必须把它们重新解释。

```python
# 来源：python/sglang/srt/entrypoints/openai/protocol.py L316-L339
class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: int = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: Optional[str] = None
```

Chat 比 Completion 复杂，因为输入不是一个 prompt，而是一组消息和约束。

```python
# 来源：python/sglang/srt/entrypoints/openai/protocol.py L654-L701
class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    model: str = Field(
        default=DEFAULT_MODEL_NAME,
        description="Model name. Supports LoRA adapters via 'base-model:adapter-name' syntax.",
    )
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of completion tokens for a chat completion request, "
        "including visible output tokens and reasoning tokens. Input tokens are not included. ",
    )
    n: int = 1
    presence_penalty: float = 0.0
    response_format: Optional[Union[ResponseFormat, StructuralTagResponseFormat]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )  # noqa
    parallel_tool_calls: bool = True
    return_hidden_states: bool = False
    return_routed_experts: bool = False
    routed_experts_start_len: int = 0
    return_cached_tokens_details: bool = False
    return_prompt_token_ids: bool = False
    return_meta_info: bool = False
    reasoning_effort: Optional[Literal["none", "low", "medium", "high", "max"]] = Field(
        default=None,
        description="Constrains effort on reasoning for reasoning models. "
        "'none' disables reasoning entirely, 'low' is the least effort, 'high' is the most effort. "
        "Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning "
```

因此排查 Chat 时不要停在 `protocol.py`。`protocol.py` 只说明外部请求能被接收；实际 prompt、tool schema、reasoning constraint 要看 serving handler。

## 内部契约：`GenerateReqInput`

Completion 的转换比较直观：prompt 可能是 text 或 input ids，sampling 参数从 OpenAI 字段重建，LoRA 和 DP rank 等扩展也在这里进入内部请求。

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

Chat 的转换多了消息模板、tool constraint、multimodal payload 和 reasoning。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L553-L622
        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)
        # Build sampling parameters
        sampling_params = request.to_sampling_params(
            stop=processed_messages.stop,
            model_generation_config=self.default_sampling_params,
            tool_call_constraint=processed_messages.tool_call_constraint,
        )

        if request.input_ids is not None:
            prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
        elif is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        # Extract custom labels from raw request headers
        custom_labels = self.extract_custom_labels(raw_request)

        # Extract routed_dp_rank from header (has higher priority than body)
        effective_routed_dp_rank = self.extract_routed_dp_rank_from_header(
            raw_request, request.routed_dp_rank
        )

        # Resolve LoRA adapter from model parameter or explicit lora_path
        lora_path = self._resolve_lora_path(request.model, request.lora_path)
        img_max_dynamic_patch, vid_max_dynamic_patch = _extract_max_dynamic_patch(
            request
        )
        require_reasoning = self._get_reasoning_from_request(request)

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
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
            require_reasoning=require_reasoning,
            priority=request.priority,
            routing_key=self.extract_routing_key(raw_request),
            custom_labels=custom_labels,
            custom_logit_processor=request.custom_logit_processor,
            images_config=getattr(request, "images_config", None),
            image_max_dynamic_patch=img_max_dynamic_patch,
            video_max_dynamic_patch=vid_max_dynamic_patch,
            max_dynamic_patch=getattr(request, "max_dynamic_patch", None),
            use_audio_in_video=getattr(request, "use_audio_in_video", False),
            return_prompt_token_ids=request.return_prompt_token_ids,
        )
```

这里的关键不是字段多，而是所有外部协议差异最后都被压进一个内部请求对象。

但“生成约束已建立”不等于“响应一定会被还原成 OpenAI `tool_calls`”。required/named tool choice 在没有 parser-specific constraint 时可以退回 JSON schema；当前出站 tool-call 分支仍要求配置 `tool_call_parser`。完全没有 parser 时，受约束 JSON 可能作为普通 `content` 返回，客户端不能只凭 `tool_choice="required"` 推断一定拿到结构化 `tool_calls`。

## 流式输出：先识别上游 chunk 语义，再生成外部 delta

默认非增量模式下，TokenizerManager 的 chunk 是“当前累积状态快照”，OpenAI SDK 期待的是 delta，兼容层必须为每个 choice 维护 offset。打开 `incremental_streaming_output` 后，TokenizerManager 已直接给出互不重叠的分段；Chat handler 会显式切换到直接透传该分段，不能再套用累积 offset。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L356-L361
        offset = stream_offsets.get(index, 0)
        if self.tokenizer_manager.server_args.incremental_streaming_output:
            delta = content["text"]
        else:
            delta = content["text"][offset:]
            stream_offsets[index] = len(content["text"])
```

Completion 当前仍无条件使用累积文本切片，没有像 Chat 一样检查 `incremental_streaming_output`。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_completions.py L316-L318
                # Generate delta
                delta = text[offset:]
                stream_offsets[index] = len(content["text"])
```

所以重复或漏字问题要先同时检查两件事：TokenizerManager 当前产出的是累积快照还是分段 delta，以及对应 adapter 是否采用了匹配的 offset 策略。特别是 Completion 与 Ollama，在启用 `--incremental-streaming-output` 时不能直接假设兼容层已经正确适配。

## 复盘迁移

读完本专题后，把同一个模型迁移到其他 endpoint：

| Endpoint | 迁移读法 |
|----------|----------|
| `/v1/embeddings` | 外部 `input` 变成 `EmbeddingReqInput`，不走生成 stream |
| `/api/chat` | Ollama 独立协议，仍转成 `GenerateReqInput` |
| OpenAI Responses | 仍然先问外部事件如何映射到内部请求和输出事件 |

如果一个 endpoint 没有走 `OpenAIServingBase`，不要套用 OpenAI handler 的模板方法；先找它自己的 adapter。
