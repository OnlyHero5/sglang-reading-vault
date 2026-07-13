---
title: "OpenAI-API · 数据流"
type: dataflow
framework: sglang
topic: "OpenAI-API"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/dataflow
  - source-reading
updated: 2026-07-11
---
# OpenAI-API · 数据流

## 先回答为什么读

源码走读告诉你函数顺序，本篇补上对象在每一步的形状。OpenAI API 兼容层的关键数据流不是“HTTP 到模型”，而是下面这条转换链：

```text
ChatCompletionRequest / CompletionRequest
  -> GenerateReqInput
  -> TokenizerManager request state
  -> internal content chunk
  -> OpenAI SSE chunk
```

如果你能说清楚每个节点持有什么字段，就能更快定位“请求字段没生效”“stream 重复”“usage 不对”“连接断开后请求没停”等问题。

## 入站数据流：外部 JSON 到内部请求

### 1. 路由阶段只保留原始请求和 Pydantic 对象

FastAPI route 收到两个东西：已经解析好的 Pydantic request，以及 `raw_request`。前者给 handler 做协议转换，后者保留 headers、连接状态、请求日志上下文。

```text
raw HTTP body + headers
  -> CompletionRequest / ChatCompletionRequest
  -> handler.handle_request(request, raw_request)
```

`raw_request` 后面会被用于请求日志、custom labels、routing key、DP rank header，并传入 `TokenizerManager.generate_request` 做连接断开检测。StreamingResponse 的 background abort task 持有的是转换后的内部请求对象，不是 `raw_request`。

### 2. Header 可以覆盖 body 的 DP rank

兼容层不只是 body 字段映射。`X-Data-Parallel-Rank` 这类部署侧 header 会在 handler 中进入内部请求，而且优先级高于 body。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_base.py L277-L306
    def extract_routed_dp_rank_from_header(
        self, raw_request: Request, body_routed_dp_rank: Optional[int] = None
    ) -> Optional[int]:
        """Extract routed_dp_rank from HTTP header, with higher priority than routed_dp_rank in body.

        Header name: X-Data-Parallel-Rank (case-insensitive in HTTP/1.1/2)
        """
        if raw_request is None:
            return body_routed_dp_rank

        header_value = raw_request.headers.get("x-data-parallel-rank")
        if header_value is not None:
            try:
                header_dp_rank = int(header_value)
                if (
                    body_routed_dp_rank is not None
                    and header_dp_rank != body_routed_dp_rank
                ):
                    logger.debug(
                        f"X-Data-Parallel-Rank header ({header_dp_rank}) overrides "
                        f"body routed_dp_rank ({body_routed_dp_rank})"
                    )
                return header_dp_rank
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid X-Data-Parallel-Rank header: must be an integer, got '{header_value}'",
                )

        return body_routed_dp_rank
```

排查 DP routing 时，要同时看 body 和 header。只打印 JSON body 可能看不到最终 rank。

### 3. LoRA 可以藏在 model 字段里

SGLang 允许 `base-model:adapter-name` 这种兼容语法。它让 OpenAI 的 `model` 字段携带 LoRA adapter 信息。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_base.py L40-L71
    def _parse_model_parameter(self, model: str) -> Tuple[str, Optional[str]]:
        """Parse 'base-model:adapter-name' syntax to extract LoRA adapter.

        Returns (base_model, adapter_name) or (model, None) if no colon present.
        """
        if ":" not in model:
            return model, None

        # Split on first colon only to handle model paths with multiple colons
        parts = model.split(":", 1)
        base_model = parts[0].strip()
        adapter_name = parts[1].strip() or None

        return base_model, adapter_name

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

所以 LoRA 未生效时，要先确认 `model` 字段里有没有冒号，以及它是否覆盖了显式 `lora_path`。

## 生成边界：Serving 到 TokenizerManager

`GenerateReqInput` 进入 `TokenizerManager.generate_request` 后，才开始进入 SGLang serving 内核的公共路径。这里会 normalize batch、设置 priority、校验 DP rank、初始化请求状态、等待 pause、校验 LoRA、tokenize 并发送 scheduler。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L619-L636
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
```

这个边界很重要：OpenAI handler 负责“把请求说清楚”，TokenizerManager 负责“把请求送进系统”。

## 出站数据流：内部 chunk 到 OpenAI SSE

### 1. Chat stream 先发 role-only chunk

OpenAI Chat stream 通常先发一个 `role=assistant`、content 为空的 chunk。SGLang 在每个 choice 的第一次 chunk 上显式发出这个角色。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_chat.py L1099-L1125
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

                # Generate streaming content (override in subclass for custom behavior)
                async for chunk in self._generate_stream_content(
                    content=content,
                    index=index,
                    request=request,
                    stream_offsets=stream_offsets,
                    reasoning_parser_dict=reasoning_parser_dict,
                    parser_dict=parser_dict,
                    has_tool_calls=has_tool_calls,
                    choice_logprobs=choice_logprobs,
                    finish_reason_type=finish_reason_type,
                    continuous_usage_stats=continuous_usage_stats,
                    prompt_tokens=prompt_tokens,
                    reasoning_tokens=reasoning_tokens,
```

如果客户端以为第一包一定有正文，就会误判“空 delta”。这不是生成失败，而是 OpenAI Chat stream 的外层形状。

### 2. Usage 不能简单累加所有 prompt tokens

多 choice 下，一个 prompt 会产生多个 choice。usage 统计时 prompt tokens 只按每组 choice 的第一个 index 计入，completion 和 reasoning tokens 才按所有 choice 累加。

```python
# 来源：python/sglang/srt/entrypoints/openai/usage_processor.py L58-L92
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        reasoning_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
        image_tokens: int = 0,
        audio_tokens: int = 0,
        video_tokens: int = 0,
    ) -> UsageInfo:
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
            image_tokens=image_tokens,
            audio_tokens=audio_tokens,
            video_tokens=video_tokens,
        )
```

所以 `n>1` 时 usage 看起来“不等于所有 chunk prompt_tokens 相加”是符合设计的。

### 3. Abort 是连接生命周期问题

StreamingResponse 的 background task 会在响应 teardown 后延迟尝试 abort 请求；如果请求已经正常结束，abort 通常是 no-op，如果连接提前断开，它负责补充清理。等待响应期间的即时断连检测则依赖传入 `generate_request` 的 `raw_request.is_disconnected()`。这两个机制都在 TokenizerManager 边界，而不是 Scheduler。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L1808-L1819
    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
```

如果客户端断开后 GPU 仍继续跑，检查 StreamingResponse 是否挂了这个 background task，以及请求是否 batch 化。

## Embedding 数据流

Embedding 也走 `OpenAIServingBase`，但内部请求是 `EmbeddingReqInput`，不走 Chat 的 stream 状态机。当前 serving 层会严格检查单字符串、`list[str]` 和扁平 `list[int]` 的空值、混合类型与负 token id；嵌套 token-id batch 和 `MultimodalEmbeddingInput` 不由这段分支完整验证，而是在后续转换、normalize 和 tokenization 边界继续处理。

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

Embedding 问题不要套 Chat 的 template/tool/reasoning 逻辑；它有自己的 input 校验和 embedding request 转换。

## Ollama 数据流

Ollama 是另一套协议入口，但仍然复用 TokenizerManager。它的 options 会先转成 SGLang sampling params，并给 `max_new_tokens` 一个更长的默认值。它的 stream adapter 默认假设上游 text 是累积态；打开 incremental streaming 后必须额外检查 delta 是否被再次切片。

```python
# 来源：python/sglang/srt/entrypoints/ollama/serving.py L31-L66
class OllamaServing:
    """Handler for Ollama-compatible API endpoints."""

    def __init__(self, tokenizer_manager):
        self.tokenizer_manager = tokenizer_manager

    def _get_timestamp(self) -> str:
        """Get current timestamp in Ollama format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

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

排查 Ollama 输出长度时，先看 `num_predict` 是否映射到 `max_new_tokens`，以及默认值是否符合预期。排查尾字缺失或终止原因不准时，还要看 stream adapter：它在 `is_done` 分支丢弃本轮 `delta`，并把 `done_reason` 固定为 `stop`，没有透传内部 finish reason。

## 交互边界总结

| 边界 | 上游对象 | 下游对象 | 排查重点 |
|------|----------|----------|----------|
| Route → Handler | Pydantic request + `raw_request` | handler method call | endpoint 是否走对 handler |
| Handler → Internal | OpenAI/Ollama fields | `GenerateReqInput` / `EmbeddingReqInput` | 字段是否被翻译或丢弃 |
| Internal → TM | internal request | tokenized request / scheduler request | LoRA、priority、pause、DP rank |
| TM → Handler | content + `meta_info` | stream state | `index`、finish_reason、usage counters |
| Handler → Client | delta chunk | SSE / NDJSON | offset、reasoning、tool calls、usage、done |

这张表还有一个运行时开关边界：默认模式中 Handler 收到累积 text，需要自己切 delta；`incremental_streaming_output=True` 时 TokenizerManager 已给出分段。Chat handler 检查这个开关，Completion 与 Ollama 当前仍按累积态处理，后两条路径启用该开关时应列为兼容性风险。

---

## 运行验证

维护本文时，先用下面的命令确认 OpenAI / Ollama / Embedding 数据流边界仍在：

```powershell
rg -n "OpenAIServingBase|generate_request|ChatCompletionResponseStreamChoice|UsageProcessor|class OpenAIServingEmbedding|class OllamaServing|_convert_options_to_sampling_params" sglang/python/sglang/srt/entrypoints/openai sglang/python/sglang/srt/entrypoints/ollama sglang/python/sglang/srt/managers/tokenizer_manager.py
```

预期信号：

- OpenAI serving 基类、Chat / Completion stream choice 和 usage processor 仍在 OpenAI entrypoints 下。
- `TokenizerManager.generate_request` 仍是 OpenAI / Ollama / Embedding 进入后端推理的共同边界。
- Ollama options 转换和 embedding handler 仍有独立入口，不应套用 Chat 的 tool / reasoning 数据流。

如果 handler 被拆分到新协议层，应先更新本篇的边界表，再更新 OpenAI API 的关键问题页。
