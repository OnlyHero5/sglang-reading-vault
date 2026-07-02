---
type: batch-doc
module: 06-TokenizerManager
batch: "06"
doc_type: faq
title: "TokenizerManager：关键问题"
tags:
 - sglang/batch/06
 - sglang/module/tokenizer-manager
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# TokenizerManager：关键问题

## 1. TokenizerManager 与 Scheduler 的职责边界

**Q：谁做 continuous batching？谁做分词？**

| 职责 | 负责模块 | 原因 |
|------|----------|------|
| 文本 → token ids | **TokenizerManager** | 靠近 API，可用 CPU 并行、dynamic batch tokenizer |
| 请求排队 / batch 合并 | **Scheduler** | 需感知 KV cache、Radix 前缀、GPU 内存 |
| token ids → 文本 | **DetokenizerManager** | 与 Scheduler 同进程树，避免回传 Python 主进程 |

TokenizerManager **不做** GPU batching；它最多把多条请求打包为 `BatchTokenizedGenerateReqInput` 一次 ZMQ 发送，Scheduler 仍决定如何并入 `ScheduleBatch`。

---

## 2. `skip_tokenizer_init=True` 何时使用？

**Explain：** 嵌入纯 token-id 工作流（如外部系统已分词、或 benchmark 直喂 ids）时可跳过加载 HF tokenizer，节省内存与启动时间。

**易错写法 vs 正确写法：**

```python
# ❌ 易错：skip_tokenizer_init=True 仍传 text
obj = GenerateReqInput(text="Hello", sampling_params={"max_new_tokens": 10})
# → _tokenize_one_request 抛出 ValueError

# ✅ 正确：提供 input_ids
obj = GenerateReqInput(
 input_ids=[1, 2, 3, 4],
 sampling_params={"max_new_tokens": 10},
)
```

**Code（校验逻辑）：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L817-L822
# 提交版本：70df09b
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
```

**Comment：**

- `skip_tokenizer_init` 时 Detokenizer 可能被 bypass，Scheduler 直发 `BatchTokenIDOutput`。
- 需要 logprob 中的 **文本** 时，必须保留 tokenizer 或自行 detokenize。

---

## 3. 权重更新时请求为何「卡住」？

**Explain：** 权重热更新或显式 `pause_generation` 时，新到达的 `generate_request` 会在分词前被两道锁挡住：先 `await is_pause_cond` 等待全局 pause 解除，再 `async with model_update_lock.reader_lock` 与权重 writer 互斥。这保证 Scheduler 侧 swap 权重期间不会有新请求带着旧 LoRA id 进入 GPU batch。

**Code：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L619-L623
# 提交版本：70df09b
            async with self.is_pause_cond:
                await self.is_pause_cond.wait_for(lambda: not self.is_pause)

            async with self.model_update_lock.reader_lock:
                await self._validate_and_resolve_lora(obj)
```

**Comment：**

- 若 Admin API 触发 `update_weights_from_tensor(abort_all=True)`，会先 `abort_request(abort_all=True)` 再持 writer_lock。
- 客户端表现为长时间无响应 → 检查是否正在进行权重热更新或 `/pause_generation`。

---

## 4. 流式输出的三种模式

| 模式 | 条件 | 客户端看到的内容 |
|------|------|------------------|
| 非流式 | `stream=False` | 仅最终一条 `{text, meta_info}` |
| 流式 + 非 incremental | `stream=True`, incremental off | 中间 chunk `text=None`，末 chunk 全量 text |
| 流式 + incremental | `incremental_streaming_output=True` | 每 chunk 为 delta text/token |

**易错：** 假设每个 SSE chunk 都含完整累积文本 — 在 incremental 模式下只有 **delta**。

**Code：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L1983-L1996
# 提交版本：70df09b
                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "text": delta_text,
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
```

---

## 5. input_embeds 与 Radix Cache 冲突

**Q：为什么 `input_embeds` 要求 `--disable-radix-cache`？**

RadixAttention 按 **token id 序列** 做前缀共享。若输入是 arbitrary embedding 而非离散 token，prefix tree 无法正确匹配与复用。

**Code：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L806-L811
# 提交版本：70df09b
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
```

---

## 6. 多 HTTP Worker 下 pause 如何一致？

**Explain：** 任一 Worker 收到 `pause_generation` 时，`MultiTokenizerRouter` 向 **所有** 已注册 Worker IPC 广播 `PauseContinueBroadcastReq`，保证每个进程的 `is_pause` 同步。

**Code：**

```python
# 来源：python/sglang/srt/managers/multi_tokenizer_mixin.py L463-L477
# 提交版本：70df09b
            if isinstance(
                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
            ):
                # Broadcast to ALL workers so every worker's is_pause is set
                is_pause = isinstance(recv_obj, PauseGenerationReqInput)
                broadcast = PauseContinueBroadcastReq(is_pause=is_pause)
                for ipc_name in self.all_worker_ipcs:
                    self.socket_mapping.send_output(ipc_name, broadcast)
                # Forward to scheduler rank 0 (it broadcasts to all TP/PP/DP
                # ranks internally). Skip for abort mode which drains via polling.
                if not (
                    isinstance(recv_obj, PauseGenerationReqInput)
                    and recv_obj.mode == "abort"
                ):
                    await async_sock_send(self.send_to_scheduler, recv_obj)
```

**Comment：** 若只 pause 单个 Worker，其他 Worker 仍会向 Scheduler 送请求，导致「半暂停」不一致状态。

---

## 7. score API 与 generate 的关系

**Q：`/v1/score` 是否单独走 Scheduler 路径？**

否。`TokenizerManagerScoreMixin.score_request` 构造 `GenerateReqInput(max_new_tokens=0)` 或 `EmbeddingReqInput`，调用 **`generate_request`**，从 logprob 或 embedding 字段提取分数。

**Code：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager_score_mixin.py L80-L107
# 提交版本：70df09b
    def _build_multi_item_token_sequence(
        self, query: List[int], items: List[List[int]], delimiter_token_id: int
    ) -> Tuple[List[int], List[int]]:
        """
        Build a single token sequence for multi-item scoring.
        Format: query<delimiter>item1<delimiter>item2<delimiter>item3<delimiter>

        Args:
            query: Query token IDs
            items: List of item token ID sequences
            delimiter_token_id: Token ID to use as delimiter

        Returns:
            Tuple of (combined token sequence, delimiter indices)
        """
        combined_sequence = query[:]  # Start with query
        delimiter_indices = []

        for item in items:
            delimiter_indices.append(len(combined_sequence))
            combined_sequence.append(delimiter_token_id)  # Add delimiter
            combined_sequence.extend(item)  # Add item tokens

        # Add final delimiter after the last item for logprob extraction
        delimiter_indices.append(len(combined_sequence))
        combined_sequence.append(delimiter_token_id)

        return combined_sequence, delimiter_indices
```

**Comment：** `--enable-mis` 时用 `MIS_DELIMITER_TOKEN_ID` 分隔 query/items，一次 forward 得到多个 delimiter 位置的 logprob。

---

## 8. 与 vLLM API Server 的对比（简表）

| 维度 | SGLang TokenizerManager | vLLM（典型架构） |
|------|-------------------------|------------------|
| 进程模型 | 独立进程 + ZMQ 与 Scheduler 通信 | 常 AsyncLLMEngine 同进程 asyncio |
| 分词位置 | 专用 TokenizerManager 进程/线程 | Engine 内 tokenize |
| 控制面 | Mixin + FanOutCommunicator | 较分散在 engine / worker |
| 多 HTTP Worker | Router + TokenizerWorker | 通常 uvicorn workers 各载引擎 |

SGLang 选择 **进程隔离** 是为了：Python GIL 下 HTTP 与分词不阻塞 Scheduler；多 Worker 水平扩展分词 CPU 开销。

---

## 验证建议（零基础可试）

1. **操作：** 启动 `sglang serve --model-path meta-llama/Llama-3.2-1B`，另开终端执行 
 `curl -N http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"default","messages":[{"role":"user","content":"Say hi in 5 words"}],"stream":true,"stream_options":{"include_usage":true}}'` 
 **预期现象：** 终端逐行出现 `data: {"choices":[{"delta":{"content":"..."}}]}`，末行 `data: [DONE]`；服务端日志可见 TokenizerManager 收 Detokenizer 回包。 
 **对应文档节：** [[06-TokenizerManager-01-核心概念|01-核心概念 § 用户故事]]、§2 ReqState、§4 分词策略

2. **操作：** 同一服务加 `--skip-tokenizer-init`，再 curl 传 `"text":"Hello"`（不传 input_ids）。 
 **预期现象：** HTTP 400/422，`ValueError: cannot accept text prompts`；改传 `"input_ids":[1,2,3]` 后正常。 
 **对应文档节：** §2 `skip_tokenizer_init`、§5 input_embeds 冲突

3. **操作：** 流式请求时观察中间 chunk：默认 vs 加 `"stream_options"` 且 body 含 `"incremental_streaming_output": true`（若 API 支持）或查 server 对应 flag。 
 **预期现象：** 非 incremental 时中间包 `text` 可能为 `null`，末包才含全量；incremental 时每包为 delta。 
 **对应文档节：** §4 流式三种模式、[[06-TokenizerManager-04-关键问题|04-关键问题 §4]]

---

## 9. 常见报错速查

| 现象 | 可能原因 | 文档章节 |
|------|----------|----------|
| `cannot accept text prompts` | skip_tokenizer_init 但未传 input_ids | §2 |
| `longer than context length` | 输入+max_new_tokens 超限 | `_validate_one_request` |
| `Received output for rid but state deleted` | 重复 abort / 竞态 | `_handle_batch_output` |
| 流式无 text | 非 incremental 中间 chunk 设计如此 | §4 |
| LoRA not found | `lora_path` 未注册或已 LRU 卸载 | Control Mixin `load_lora_adapter` |
