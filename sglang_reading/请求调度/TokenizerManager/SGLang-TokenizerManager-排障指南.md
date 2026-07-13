---
title: "TokenizerManager · 排障指南"
type: troubleshooting
framework: sglang
topic: "TokenizerManager"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# TokenizerManager · 排障指南

本篇按症状组织，不按源码文件顺序组织。先定位现象，再找源码入口和验证方式。

## 症状速查

| 症状 | 可能原因 | 源码入口 | 验证 |
|------|----------|----------|------|
| text 请求报 `skip_tokenizer_init=True cannot accept text prompts` | 启动时跳过 tokenizer，却仍传 `text` | `_tokenize_one_request` | 改传 `input_ids`，或关闭 `skip_tokenizer_init` |
| 请求在权重更新期间迟迟不进入 Scheduler | 等待 pause 解除或 writer lock | `generate_request`、`update_weights_*` | 打日志看是否停在 `is_pause_cond` 或 `model_update_lock.reader_lock` |
| 流式中间包 `text=None` | 非 incremental streaming 延迟 materialize，避免 O(n^2) 拼接 | `_handle_batch_output`、`_wait_one_response` | 打开 `--incremental-streaming-output` 对比输出形态 |
| 流式输出一次吐多个 token | 前台消费慢或协程调度期间同一 rid 积压多个 delta | `_coalesce_streaming_chunks` | 观察 backlog warning和客户端消费；不要把 `batch_notify_size` 当成跨消息 token buffer |
| 多 worker 下回包找不到 state | `http_worker_ipc` 丢失或路由回错误 worker | `_dispatch_to_scheduler`、`_distribute_result_to_workers` | 检查请求对象和回包里的 `http_worker_ipc(s)` |
| `rid_to_state` 增长 | 请求到 Scheduler 前异常未清理、后端没有 finished 回包，或 `n>1` 留下规范化 placeholder state | `_discard_pending_req_states`、`_handle_batch_output`、`_handle_batch_request` | 分别对照失败请求、普通 `n=1`、固定 `B/N` 的 `n>1` 请求完成前后 state 数量 |
| 客户端断连后 GPU 仍在跑 | streaming background abort 未执行、非流式轮询未命中，或 `AbortReq` 未传播 | `create_abort_task`、`_wait_one_response`、`abort_request` | 区分 SSE 与非流式连接，沿同一 rid 检查 background task 和 Scheduler abort echo |

## `skip_tokenizer_init=True` 不是“只少一步分词”

`skip_tokenizer_init=True` 的本质是主进程没有 tokenizer。它改变输入和输出两侧：

| 方向 | 普通模式 | skip tokenizer |
|------|----------|----------------|
| 输入 | 可以传 `text`，TokenizerManager 分词 | 必须传 `input_ids` 或 `input_embeds` |
| 输出 | Scheduler 输出 token ids，经 Detokenizer 变成 text | TokenizerManager 直接处理 `BatchTokenIDOutput`，通常返回 `output_ids` |

源码入口：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L817-L822
            if self.tokenizer is None:
                raise ValueError(
                    "The engine initialized with skip_tokenizer_init=True cannot "
                    "accept text prompts. Please provide input_ids or re-initialize "
                    "the engine with skip_tokenizer_init=False."
                )
```

配置校验还会强制关掉 tokenizer/detokenizer worker 和 tokenizer batch encode：

```python
# 来源：sglang/python/sglang/srt/server_args.py L6146-L6175
        if self.skip_tokenizer_init:
            if self.tokenizer_worker_num != 1:
                logger.warning(
                    "skip_tokenizer_init=True disables tokenizer workers; forcing tokenizer_worker_num=1 "
                    f"(requested {self.tokenizer_worker_num})."
                )
                self.tokenizer_worker_num = 1
            if self.detokenizer_worker_num != 1:
                logger.warning(
                    "skip_tokenizer_init=True disables detokenizer workers; forcing detokenizer_worker_num=1 "
                    f"(requested {self.detokenizer_worker_num})."
                )
                self.detokenizer_worker_num = 1

            if self.enable_tokenizer_batch_encode:
                logger.warning(
                    "skip_tokenizer_init=True ignores --enable-tokenizer-batch-encode; disabling it."
                )
                self.enable_tokenizer_batch_encode = False

            if self.enable_dynamic_batch_tokenizer:
                logger.warning(
                    "skip_tokenizer_init=True ignores --enable-dynamic-batch-tokenizer; disabling it."
                )
                self.enable_dynamic_batch_tokenizer = False

            logger.info(
                "skip_tokenizer_init=True: string-based stop conditions (stop, stop_regex) "
                "and min_new_tokens are unavailable."
            )
```

验证方式：启动 skip tokenizer 后分别发送 text 和 input ids。text 应该在 TokenizerManager 前置校验失败；input ids 会进入 Scheduler。

## 为什么权重更新时请求会卡住

这是设计结果，不是死锁。`generate_request` 在分词和发送前先等 pause，再拿 `model_update_lock.reader_lock`：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L619-L631
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
```

权重更新侧如果没处在 paused 状态，会拿 writer lock：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_control_mixin.py L395-L418
    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_distributed_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_distributed_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
```

排查时不要只看 Scheduler 队列。如果请求还没进入 `_send_one_request`，它可能还在 TokenizerManager 的 pause/lock 门口。

## 为什么流式中间包可能没有完整文本

非 incremental streaming 下，中间包可以设置 `text=None`，把完整字符串拼接延迟到 `_wait_one_response`：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1997-L2011
                    elif state.finished:
                        out_dict = {
                            "text": state.get_text(),
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        # Non-incremental intermediate: pass reference (no
                        # copy) and defer text to _wait_one_response to avoid
                        # O(n) per-step cost that compounds to O(n^2).
                        out_dict = {
                            "text": None,
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
```

前台等待者醒来后，如果发现 `text is None`，再调用 `state.get_text()`：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1494-L1503
            # Resolve deferred text for non-incremental streaming.
            # _handle_batch_output sets "text": None on intermediate chunks
            # to avoid O(n) string rebuild per step (O(n^2) total).
            if (
                is_stream
                and not incremental_stream
                and "text" in out
                and out["text"] is None
            ):
                out["text"] = state.get_text()
```

验证方式：对同一请求分别打开/关闭 `--incremental-streaming-output`。incremental 模式下 chunk text 应该是 delta；非 incremental 模式下可能看到中间状态延迟 materialize。

## 为什么一次 SSE chunk 会包含多个 token

前台醒来前如果同一 `rid` 的 `out_list` 已积压多个 incremental chunk，就会合并。`batch_notify_size` 只是在处理同一个批回包时，每通知若干个 rid 后让出 event loop；批尾余数仍会立即通知，它不会跨多个后端消息主动攒 token。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1365-L1400
    def _coalesce_streaming_chunks(
        self,
        out_list: list,
        rid: str,
        customized_info_keys: Optional[Iterable[str]] = None,
    ) -> dict:
        """Coalesce multiple incremental streaming chunks into one.

        Both text and output_ids are incremental deltas, so we concatenate them;
        all other fields (meta_info, etc.) are taken from the last chunk.
        """
        if len(out_list) >= 20:
            logger.warning(
                "Streaming backlog: rid=%s, coalescing %d queued chunks into one. "
                "This may inflate P99 ITL for affected requests.",
                rid,
                len(out_list),
            )
        out = dict(out_list[-1])
        if "output_ids" in out:
            out["output_ids"] = [id for chunk in out_list for id in chunk["output_ids"]]
        if "text" in out:
            out["text"] = "".join(chunk["text"] for chunk in out_list)
        if "meta_info" in out:
            meta_info_list = [chunk["meta_info"] for chunk in out_list]
            meta_info = dict(meta_info_list[-1])
            incremental_streaming_keys = set(_INCREMENTAL_STREAMING_META_INFO_KEYS)
            if customized_info_keys is not None:
                incremental_streaming_keys.update(customized_info_keys)
            for key in incremental_streaming_keys:
                if any(key in m for m in meta_info_list):
                    meta_info[key] = [
                        item for m in meta_info_list for item in m.get(key, [])
                    ]
            out["meta_info"] = meta_info
        return out
```

这不是 token 丢失，而是队列积压后的合并。真正要关注的是 backlog warning、ITL、HTTP 客户端消费速度和前台协程调度；只有在单个回包 batch 很大时，`batch_notify_size` 才影响后台让出 event loop 的节奏。

## 客户端断连如何传播

非流式请求仍在等待时，`_wait_one_response` 每次等待 event 超时都会检查 request 是否断连；如果断连且不是 background，就发 abort：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1455-L1472
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue
```

HTTP streaming 还有另一条更重要的收尾：`StreamingResponse` 注册 `create_abort_task(obj)` 为 background task，响应结束或连接被取消后延迟两秒，再对 single/batch 的实际 rid 发送 abort。正常完成时 single-worker 的 state 已删除，`abort_request` 会成为 no-op；异常断连时它负责兜底。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1808-L1820
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
        return background_tasks
```

`abort_request` 只是向 Scheduler 发 `AbortReq`，不是本地强删所有状态：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1677-L1689
    def abort_request(self, rid: str = "", abort_all: bool = False):
        # Empty rid would startswith-match every request on the scheduler.
        if not abort_all and not rid:
            logger.warning("Ignore abort_request with empty rid and abort_all=False")
            return
        if (
            not abort_all
            and self.server_args.tokenizer_worker_num == 1
            and rid not in self.rid_to_state
        ):
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self._dispatch_to_scheduler(req)
```

排查断连后 GPU 仍跑的问题，要先区分 streaming background abort 与非流式轮询检测，再看 `AbortReq` 是否到 Scheduler、Scheduler 是否返回 abort finish reason。background 请求显式跳过 HTTP disconnect 轮询，因为它允许脱离原连接继续运行。

## 为什么不要同时传 `text`、`input_ids` 或 `input_embeds`

公开契约应当三选一。当前 `GenerateReqInput._validate_inputs` 直接拒绝的是三者全空或三者全有；若只同时传两个字段，后续 batch-size 判断和 `_tokenize_one_request` 的优先级可能产生难以预期的组合。不要把这种当前实现细节当成受支持能力，API adapter 应在边界处保持单一输入来源。

验证方式：在调用方构造请求前断言非空输入字段数量为 1；若准备收紧 upstream 校验，要同时覆盖 HTTP schema、Engine API、batch 与 multimodal 用例。

## `n > 1` 为什么会看到额外请求和 rid

parallel sampling 不是简单让 Scheduler 对一个 rid 返回 N 份结果。TokenizerManager 会先发送一次 `max_new_tokens=0` 的前缀预热请求，再复制 tokenized object，为每个实际 sample 重新生成 rid。结果用实际 sample rid 关联，并通过 `index` 恢复 batch 位置。

排查时若只按用户传入的 rid 搜日志，可能漏掉规范化 placeholder、预热和实际采样请求。应同时打印 `parallel_sample_num`、三组内部 rid、输出 `index`，并记录请求完成后的 `len(rid_to_state)`。

当前源码有一个可静态闭合的风险：原始 batch 为 `B`、`n=N>1` 时，normalization 与 `_init_req_state` 创建 `B×N` 个 placeholder state，而 `_handle_batch_request` 只把前 `B` 个当作 prompt，并只显式删除这 `B` 个；余下 `B×(N-1)` 个既没有发给 Scheduler，也没有正常 finished 回包。若线上表现为“只有 `n>1` 时 state 数单调增长”，先用固定 `B/N` 连续请求，检查每轮净增长是否接近 `B×(N-1)`。这是当前基线的生命周期缺口；内存、告警或最终故障强度仍需结合真实 workload 测量，不能直接外推固定阈值。

## 多 detokenizer 为什么不能只看 `MultiTokenizerRouter`

当 detokenizer worker 数大于 1 时，Scheduler 输出先由 `MultiDetokenizerRouter` 根据 `http_worker_ipc` 做稳定哈希，确保同一请求持续进入同一 detokenizer；解码完成后，detokenizer worker 再直接把单条结果发送给 owner tokenizer worker。若只检查 `MultiTokenizerRouter._distribute_result_to_workers`，会漏掉这一条回程。

## input embeds 为什么要求关闭 radix cache

`input_embeds` 绕过普通 token embedding 查表，prefix cache key 很难与普通 token ids 语义保持一致。因此源码要求关闭 radix cache：

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L805-L813
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
```

验证方式：构造 `input_embeds` 请求，分别在开启和关闭 radix cache 下运行。开启时应在 TokenizerManager 前置失败。

## 多 HTTP worker 下 pause 为什么要广播

单 worker 下 `is_pause` 是一个进程内状态；多 worker 下每个 TokenizerWorker 都有自己的 `is_pause`。如果只暂停发起 worker，其他 worker 仍会向 Scheduler 发新请求。

router forward path 在收到 pause/continue 时广播给所有 worker：

```python
# 来源：sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py L463-L480
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
                continue

            await async_sock_send(self.send_to_scheduler, recv_obj)
```

worker 收到广播后本地更新 `is_pause`，continue 时唤醒等待的请求：

```python
# 来源：sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py L662-L669
    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        """Apply pause/continue state under the condition lock."""
        async with self.is_pause_cond:
            if obj.is_pause:
                self.is_pause = True
            else:
                self.is_pause = False
                self.is_pause_cond.notify_all()
```

验证方式：多 worker 启动后从任意 worker 触发 pause，再向其他 worker 发 generate。正确现象是所有 worker 的新请求都等待。

## score API 为什么还会进入 `generate_request`

score 不是独立后端。generation 模型走 `GenerateReqInput(max_new_tokens=0, return_logprob=True)`，embedding 模型走 `EmbeddingReqInput`，最终都复用 `generate_request`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager_score_mixin.py L691-L713
        if is_generation:
            batch_request = GenerateReqInput(
                text=text_prompts,
                input_ids=input_ids,
                token_ids_logprob=label_token_ids,
                return_logprob=True,
                # Set logprob_start_len=0 for multi-item scoring since we want logprobs at all delimiter positions
                logprob_start_len=0 if use_multi_item_scoring else -1,
                stream=False,
                sampling_params={"max_new_tokens": 0},
                positional_embed_overrides=positional_embed_overrides,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )
        else:
            batch_request = EmbeddingReqInput(
                text=text_prompts,
                input_ids=input_ids,
                positional_embed_overrides=positional_embed_overrides,
                return_pooled_hidden_states=return_pooled_hidden_states,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )

        results = await self.generate_request(batch_request, request).__anext__()
```

排查 score 时，应先确认构造出来的是 generation score 还是 embedding score，再沿普通 TokenizerManager 数据面追。
