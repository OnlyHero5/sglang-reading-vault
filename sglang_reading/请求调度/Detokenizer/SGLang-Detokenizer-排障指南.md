---
title: "Detokenizer · 排障指南"
type: troubleshooting
framework: sglang
topic: "Detokenizer"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# Detokenizer · 排障指南

## 你为什么要读

这篇按症状排查。先判断输出回程走的是普通文本模式、skip token-id 模式，还是 embedding 透传模式。

## Q1：为什么看到了 token id，却没有文本？

先检查是否启用了 `skip_tokenizer_init`。该模式会禁用 tokenizer worker 扩展和 detokenizer worker 扩展，并让主 generate 回路输出 token ids。

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

排障判断：

| 现象 | 可能原因 | 看哪里 |
|------|----------|--------|
| 响应只有 `output_ids` | `skip_tokenizer_init=True` | `SchedulerIpcChannels.create` |
| 有 `BatchTokenIDOutput` 直接进 TokenizerManager | 正常 skip 回路 | `TokenizerManager._handle_batch_output` |
| 普通模式下没有文本 | Detokenizer 没收到或没发回 | `detokenizer_ipc_name`、`tokenizer_ipc_name` |

## Q2：流式输出为什么会重复或漏字？

重点看 `sent_offset`。它不是 token offset，而是字符串已发送边界。遇到不完整 UTF-8 时，Detokenizer 可以先发送可打印前缀，但不能推进 `surr_offset/read_offset`；下一轮补齐后要跳过 pending 前缀。

```python
# 来源：sglang/python/sglang/srt/managers/detokenizer_manager.py L350-L369
            if recv_obj.finished_reasons[i] is None:
                # Streaming. Invariant: sent_offset >= decoded_text_len. The
                # gap (`pending`) is "printable but uncommitted" text emitted
                # in a prior "�" recovery step; we skip it from this step's
                # emission so we don't double-send.
                pending = s.sent_offset - s.decoded_text_len
                if new_text and not new_text.endswith("�"):
                    # Clean text: commit to decoded_text and advance offsets.
                    s.append_decoded_text(new_text)
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    s.sent_offset = s.decoded_text_len
                    output_strs.append(new_text[pending:] if pending else new_text)
                else:
                    # Incomplete UTF-8: emit the printable prefix only; do not
                    # commit (token offsets stay so the next iteration retries
                    # with more tokens).
                    printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
                    output_strs.append(printable[pending:] if pending else printable)
```

正确理解：

| 误解 | 正确理解 |
|------|----------|
| 每个 token 都能独立 decode 成最终字符 | tokenizer 可能需要前后文 token |
| 出现 replacement char 就该立刻发出 | Detokenizer 只发安全可打印前缀 |
| `output_strs[i]` 是完整文本 | streaming 时它是本轮增量 |

还要区分下一层：`output_strs` 在 Detokenizer 内始终是 delta，但 TokenizerManager 可以按 `incremental_streaming_output` 对客户端暴露 delta 或累积语义。客户端“重复全文”未必是 Detokenizer 重复 decode，也可能只是非 incremental streaming 的预期接口。

## Q3：`Decode status not found` 怎么处理？

这通常说明 `decode_status` 状态表容量不足，旧 rid 被驱逐后又来了后续 chunk。

```python
# 来源：sglang/python/sglang/srt/managers/detokenizer_manager.py L56-L60
# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))
```

处理顺序：

| 步骤 | 操作 |
|------|------|
| 1 | 确认是否是高并发长连接 streaming |
| 2 | 检查是否存在请求 finish 包延迟或客户端断连未清理 |
| 3 | 调大 `SGLANG_DETOKENIZER_MAX_STATES` |
| 4 | 观察 Detokenizer 进程内存和状态缺失错误是否下降 |

注意：状态表是 FIFO 容量保护，不是严格 LRU。容量压力下，早插入但仍活跃的请求也可能被驱逐。

## Q4：`disable_tokenizer_batch_decode` 应该什么时候打开？

默认批量 decode 是为了吞吐。但某些 tokenizer 或模型配置下，`batch_decode` 和逐行 `decode` 结果可能不一致。源码注释提到 gpt-oss 这类 edge case。fast tokenizer 的正常批路径还会按 `(skip_special_tokens, spaces_between_special_tokens)` 分组；因此只要 batch 内配置不同，并不意味着配置会串用。

```python
# 来源：sglang/python/sglang/srt/managers/detokenizer_manager.py L311-L332
        else:
            # Do not use batch decode to prevent some detokenization edge cases (e.g., gpt-oss).
            surr_texts = [
                self.tokenizer.decode(
                    surr, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for surr, skip, space in zip(
                    surr_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
            read_texts = [
                self.tokenizer.decode(
                    read, skip_special_tokens=skip, spaces_between_special_tokens=space
                )
                for read, skip, space in zip(
                    read_ids,
                    recv_obj.skip_special_tokens,
                    recv_obj.spaces_between_special_tokens,
                )
            ]
```

建议：只有确认是 batch decode 语义差异导致错误文本时再打开。它会牺牲吞吐，因为每个请求逐行 decode。

## Q5：embedding 模型为什么也经过 Detokenizer？

消息会经过同一条 IPC 回程，但不做 detokenize。`BatchEmbeddingOutput` handler 直接透传。

```python
# 来源：sglang/python/sglang/srt/managers/detokenizer_manager.py L203-L205
    def handle_batch_embedding_out(self, recv_obj: BatchEmbeddingOutput):
        # If it is embedding model, no detokenization is needed.
        return recv_obj
```

所以 embedding 路径的延迟问题一般不在 `DecodeStatus` 或 UTF-8 边界上，而在 embedding 输出序列化、TokenizerManager 接收或 HTTP 返回。

## Q6：Detokenizer 会用 FanOutCommunicator 吗？

不会。`FanOutCommunicator` 是 TokenizerManager 控制面向多个 Scheduler rank fan-out 请求并收齐响应的工具。Detokenizer 数据面是 Scheduler 到 Detokenizer 到 TokenizerManager 的输出链。

源码里的 Detokenizer dispatcher 没有 communicator：

```python
# 来源：sglang/python/sglang/srt/managers/detokenizer_manager.py L151-L159
    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),
                (BatchTokenIDOutput, self.handle_batch_token_id_out),
                (FreezeGCReq, self.handle_freeze_gc_req),
                (ConfigureLoggingReq, self.handle_configure_logging_req),
            ]
        )
```

读源码时如果看到 `communicator.py`，把它放到控制面专题理解，不要把它接进 token id 到文本的主线。

## Q7：多 detokenizer worker 会不会破坏 DecodeStatus？

设计目标就是不破坏。Router 会按 `http_worker_ipc` 稳定选择 worker，且 batch 会被拆成 one-item batch 输出。

风险来自两类破坏：

| 风险 | 表现 |
|------|------|
| `http_worker_ipcs` 缺失或长度不等于 `rids` | router assert |
| 同一请求被路由到不同 worker | 后续 worker 找不到 `DecodeStatus` |

正常路径中，`MultiDetokenizerRouter` 的 `_pick` 是确定性的，能保持同一 key 回同一 worker。但 key 是 HTTP worker IPC，不是 rid：同一 HTTP worker 的所有请求共享 Detokenizer 亲和。如果只有某个 Detokenizer CPU 饱和而其他 worker 空闲，要检查上游 HTTP worker 流量倾斜，不能只看 detokenizer worker 数量。

## Q8：stop string 和 stop token 在哪里裁剪？

Detokenizer 同时处理两种 stop：

- token stop：在 decode 前对 list[int] 裁剪。
- string stop：在 finished 文本上查找 matched string 后裁剪。

```python
# 来源：sglang/python/sglang/srt/managers/detokenizer_manager.py L171-L201
    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            if no_stop_trim:
                return output
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]
        return output
```

如果你看到 stop 字符串泄漏到最终输出，先看 `finished_reason["matched"]` 是字符串还是 token id，再看 `no_stop_trim`。

## 最小验证

| 验证 | 操作 | 预期 |
|------|------|------|
| 文本回程 | 普通 streaming generate | `BatchStrOutput.output_strs` 有增量文本 |
| skip 回程 | `--skip-tokenizer-init` 并直传 ids | 响应主要返回 `output_ids`，不是文本 |
| UTF-8 边界 | 发送包含中文和 emoji 的 streaming prompt | 不应永久输出 replacement char 或重复可打印前缀 |
| 多 worker | 开启多个 tokenizer/detokenizer worker | 输出按 `http_worker_ipcs` 回到对应 HTTP worker |
| 多 worker 倾斜 | 按 `http_worker_ipc` 汇总请求数和 Detokenizer CPU | 同 key 固定落点；负载均衡质量取决于 HTTP worker 流量分布 |
| 状态容量 | 人为调小 `SGLANG_DETOKENIZER_MAX_STATES` 压测 | 可复现状态缺失错误，调大后缓解 |

## 运行验证

如果暂时不能启动服务，可以先用源码检索确认 detokenizer 的五个关键边界仍在：skip tokenizer、状态容量、按类型分发、stop 裁剪、多 HTTP worker 回程。

```powershell
rg -n 'skip_tokenizer_init|class DetokenizerManager|DecodeStatus|TypeBasedDispatcher|BatchTokenIDOutput|BatchStrOutput|http_worker_ipcs|trim_matched_stop|SGLANG_DETOKENIZER_MAX_STATES|init_request_dispatcher|handle_batch_token_id_out' sglang/python/sglang/srt/server_args.py sglang/python/sglang/srt/managers/detokenizer_manager.py
```

读输出时按请求末端顺序看：`server_args.py` 先处理 `skip_tokenizer_init` 的限制；`DetokenizerManager` 初始化 tokenizer 与 dispatcher；`handle_batch_token_id_out` 把 token id 转成 `BatchStrOutput`；`http_worker_ipcs` 决定结果回到哪个 HTTP worker；`trim_matched_stop` 才是 stop string / stop token 的最终裁剪点。
