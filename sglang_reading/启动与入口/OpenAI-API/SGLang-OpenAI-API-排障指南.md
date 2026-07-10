---
title: "OpenAI-API · 排障指南"
type: troubleshooting
framework: sglang
topic: "OpenAI-API"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# OpenAI-API · 排障指南

## 怎么读这篇

这篇按症状排障，不按源码文件排序。每个问题都先判断它属于哪条边界：

| 边界 | 典型症状 |
|------|----------|
| 外部协议 | 请求字段校验失败、SDK 参数不被接受 |
| 内部转换 | LoRA、DP rank、sampling、tool constraint 不生效 |
| 生成边界 | 请求进入后卡住、pause、LoRA 校验、tokenize 失败 |
| 输出协议 | stream 重复、空 delta、usage 不对、tool call 断裂 |

## Q1：OpenAI API 与原生 `/generate` 的真正区别是什么？

不是“一个是兼容接口，一个是原生接口”这么简单。OpenAI API 会先把外部协议翻译成内部请求，再由 TokenizerManager 进入同一生成核心；原生 `/generate` 更接近内部契约。

| 项目 | OpenAI `/v1/*` | 原生 `/generate` |
|------|----------------|------------------|
| 输入契约 | OpenAI Pydantic model | SGLang native request |
| Prompt 处理 | Chat template、tool、reasoning 约束 | 调用方直接给 text 或 input ids |
| 流式格式 | OpenAI SSE chunk | SGLang 原生 stream |
| 扩展字段 | 兼容层选择性暴露 | 更贴近 `GenerateReqInput` |
| 排查入口 | serving handler | TokenizerManager 或 Scheduler |

验证：同一个 prompt 分别走 `/v1/chat/completions` 和 `/generate`，打印进入 TokenizerManager 前的内部请求，比较 `sampling_params`、`lora_path`、`return_logprob`、`extra_key`。

## Q2：stream 文本重复或漏字，先看哪里？

先看 offset。内部 chunk 可能是当前完整 text，外部 OpenAI chunk 需要 delta。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L356-L361
        offset = stream_offsets.get(index, 0)
        if self.tokenizer_manager.server_args.incremental_streaming_output:
            delta = content["text"]
        else:
            delta = content["text"][offset:]
            stream_offsets[index] = len(content["text"])
```

Completion 也有同样的切片。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_completions.py L316-L318
                # Generate delta
                delta = text[offset:]
                stream_offsets[index] = len(content["text"])
```

验证：打印每个 `index` 的 `len(content["text"])` 和旧 offset。如果 offset 没按 choice 隔离，多 choice stream 会互相污染。

## Q3：为什么 Chat stream 第一包 `delta.content` 是空？

这是正常的 role-only chunk。Chat stream 会先发 `role="assistant"` 和空 content，然后才进入正文、reasoning 或 tool call chunk。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L1099-L1110
                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    yield build_sse_content(
                        chunk_id=content["meta_info"]["id"],
                        created=int(time.time()),
                        model=request.model,
                        index=index,
                        role="assistant",
                        content="",
                    )
                    stream_started = True
```

验证：客户端不要把首包空 content 当作生成失败；应继续读取后续 SSE 事件。

## Q4：为什么 stream 请求有时返回普通 JSON error？

因为 handler 会先 kick-start generator。能在第一个 chunk 前发现的错误会尽量以 HTTP error 返回，而不是已经发出 stream 之后再塞错误 chunk。

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

验证：构造一个超长 context 请求。如果还没发出首包，应该看到普通 error response；如果 stream 已开始，中途 abort 会走 streaming error chunk。

## Q5：reasoning 内容为什么没有出现在 `delta.content`？

reasoning 不是普通 content。启用 parser 且请求要求分离时，delta 会先经过 reasoning parser，产出 `reasoning_content`，剩余文本才继续走 content 或 tool call。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L1594-L1615
    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: Dict[int, ReasoningParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[Optional[str], str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            is_force_reasoning = (
                self.template_manager.force_reasoning
                or self._get_reasoning_from_request(request)
            )
            reasoning_parser_dict[index] = ReasoningParser(
                self.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)
```

验证：看客户端是否读取 `delta.reasoning_content` 或额外字段，而不是只读 `delta.content`。

## Q6：tool call stream 断裂或 arguments 不完整，先看哪里？

先看 tool parser 的状态。Chat stream 会按 choice index 保存 parser，并把普通文本和 tool calls 分开产出。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L1831-L1877
    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: Dict[int, FunctionCallParser],
        content: Dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: Dict[int, bool],
        continuous_usage_stats: bool = False,
    ):
        """Process tool calls in streaming response"""
        if index not in parser_dict:
            is_required = request.tool_choice == "required" or isinstance(
                request.tool_choice, ToolChoice
            )
            # For required/named tool choice: use JsonArrayParser when the
            # constrained output is plain JSON (detector doesn't support
            # structural_tag or no parser configured). Use FunctionCallParser
            # only when the detector supports structural_tag and will produce
            # native format output.
            if is_required:
                use_native_parser = False
                if self.tool_call_parser:
                    probe = FunctionCallParser(
                        tools=request.tools,
                        tool_call_parser=self.tool_call_parser,
                    )
                    use_native_parser = probe.detector.supports_structural_tag()
                if use_native_parser:
                    parser_dict[index] = probe
                else:
                    parser_dict[index] = JsonArrayParser()
            else:
                parser_dict[index] = FunctionCallParser(
                    tools=request.tools,
                    tool_call_parser=self.tool_call_parser,
                )

        parser = parser_dict[index]

        # Handle both FunctionCallParser and JsonArrayParser
        if isinstance(parser, JsonArrayParser):
            result = parser.parse_streaming_increment(delta, request.tools)
            normal_text, calls = result.normal_text, result.calls
        else:
            normal_text, calls = parser.parse_stream_chunk(delta)
```

验证：打印 `index`、parser 类型、`normal_text` 和 `calls`。如果 required tool choice 被 plain JSON parser 接管，客户端不应期待 native structural tag 形态。

## Q7：usage 为什么在 `n>1` 时看起来少算 prompt tokens？

因为同一个 prompt 的多个 choices 共享 prompt tokens。streaming usage 只在 `index % n_choices == 0` 时计入 prompt tokens。

```python
# 来源：python/sglang/srt/entrypoints/openai/usage_processor.py L69-L88
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_reasoning_tokens = sum(reasoning_tokens.values())
        total_completion_tokens = sum(completion_tokens.values())

        cached_details = (
            UsageProcessor._details_if_cached(
                sum(tok for idx, tok in cached_tokens.items() if idx % n_choices == 0)
            )
            if enable_cache_report
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
            completion_tokens=total_completion_tokens,
            cached_tokens=cached_details,
```

验证：用 `n=2` 的请求比较每个 choice 的 `meta_info.prompt_tokens` 和最终 usage。不要把每个 choice 的 prompt tokens 简单求和。

## Q8：LoRA adapter 为什么不按 `lora_path` 走？

因为 `model` 字段中的 `base-model:adapter-name` 优先级更高。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_base.py L55-L71
    def _resolve_lora_path(
        self,
        request_model: str,
        explicit_lora_path: Optional[Union[str, List[Optional[str]]]],
    ) -> Optional[Union[str, List[Optional[str]]]]:
        """Resolve LoRA adapter with priority: model parameter > explicit lora_path.

        Returns adapter name or None. Supports both single values and lists (batches).
        """
        _, adapter_from_model = self._parse_model_parameter(request_model)

        # Model parameter adapter takes precedence
        if adapter_from_model is not None:
            return adapter_from_model

        # Fall back to explicit lora_path
        return explicit_lora_path
```

验证：同时传 `model="base:a"` 和 `lora_path="b"`，内部 `GenerateReqInput.lora_path` 应该是 `a`。

## Q9：Ollama 输出长度为什么和 SGLang native 默认不同？

Ollama adapter 会把 `num_predict` 映射到 `max_new_tokens`，如果用户没传，默认给 2048，而不是 SGLang native 常见默认值。

```python
# 来源：python/sglang/srt/entrypoints/ollama/serving.py L41-L66
    def _convert_options_to_sampling_params(self, options: dict = None) -> dict:
        """Convert Ollama options to SGLang sampling params."""
        sampling_params = {}

        if options:
            # Map Ollama options to SGLang params
            param_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "top_k": "top_k",
                "num_predict": "max_new_tokens",
                "stop": "stop",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
                "seed": "seed",
            }
            for ollama_param, sglang_param in param_mapping.items():
                if ollama_param in options:
                    sampling_params[sglang_param] = options[ollama_param]

        # Set a reasonable default for max_new_tokens if not specified
        # Ollama users typically expect longer responses than SGLang's default (128)
        if "max_new_tokens" not in sampling_params:
            sampling_params["max_new_tokens"] = 2048

        return sampling_params
```

验证：分别传和不传 `options.num_predict`，打印进入 TokenizerManager 的 `sampling_params.max_new_tokens`。

## Q10：Embedding 请求为什么和 Chat 排障路径不同？

Embedding 走 `EmbeddingReqInput`，输入校验也不同。空字符串、混合类型 list、负 token id 会在 embedding serving 层被拦截。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_embedding.py L42-L75
    def _validate_request(self, request: EmbeddingRequest) -> Optional[str]:
        """Validate that the input is not empty or whitespace only."""
        if not (input := request.input):
            return "Input cannot be empty"

        # Handle single string
        if isinstance(input, str):
            if not input.strip():
                return "Input cannot be empty or whitespace only"
            return None

        # Handle list inputs
        if isinstance(input, list):
            if len(input) == 0:
                return "Input cannot be empty"

            # Check first element to determine type
            first_item = input[0]

            if isinstance(first_item, str):
                # List of strings
                for i, item in enumerate(input):
                    if not isinstance(item, str):
                        return "All items in input list must be strings"
                    if not item.strip():
                        return f"Input at index {i} cannot be empty or whitespace only"
            elif isinstance(first_item, int):
                # List of integers (token IDs)
                for i, item in enumerate(input):
                    if not isinstance(item, int):
                        return "All items in input list must be integers"
                    if item < 0:
                        return f"Token ID at index {i} must be non-negative"
        return None
```

验证：排查 embedding 时不要套 Chat 的 `_process_messages`、tool parser 或 reasoning parser。先看 input 类型和 embedding handler。

---

## 运行验证

维护本文时，先用下面的命令确认十个问题仍有源码入口：

```powershell
rg -n "OpenAIServingBase|generate_request|ChatCompletionResponseStreamChoice|UsageProcessor|class OpenAIServingEmbedding|class OllamaServing|_convert_options_to_sampling_params|reasoning|tool_call|finish_reason" sglang/python/sglang/srt/entrypoints/openai sglang/python/sglang/srt/entrypoints/ollama sglang/python/sglang/srt/managers/tokenizer_manager.py
```

预期信号：

- stream chunk、finish reason、usage、reasoning 和 tool call 仍在 OpenAI serving 层可定位。
- LoRA / priority / request id 等请求级字段仍经 serving base 或 TokenizerManager 转换。
- Ollama 和 Embedding 仍有独立 handler，不应合并到 Chat 排障路径。

如果某类问题的入口消失，先检查协议实现是否被拆到新的 serving 文件，再更新对应 Q&A 的断点和验证建议。
