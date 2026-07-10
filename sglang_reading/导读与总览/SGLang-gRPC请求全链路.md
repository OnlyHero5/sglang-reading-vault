---
title: "SGLang gRPC 请求全链路"
type: walkthrough
framework: sglang
topic: "导读与总览"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-10
---
# SGLang gRPC 请求全链路

> 与 [[SGLang-HTTP请求全链路]] 对称 · baseline：`TextGenerate` 流式 RPC · Git `70df09b`

本文追踪 **gRPC `TextGenerate` 流式请求** 从 Proto 到文本 chunk 的职责边界。HTTP 部署可跳过；Gateway / 多语言 client 场景必读。

---

## 长文读法

这篇按 gRPC 请求如何复用 HTTP 后端主链路读：Rust server 接收 `TextGenerate`，`PyBridge` 做跨语言提交和背压，`RuntimeHandle` 进入 Python async generator，后半段复用 `TokenizerManager -> Scheduler -> Detokenizer`，最后通过 callback / channel 把文本 chunk 流回 client。

| 你的任务 | 先读 | 抓住什么 |
|----------|------|----------|
| 建立 gRPC 总图 | 总览时序 | gRPC 前半段不同，推理核心仍复用 Python runtime |
| 排查启动模式 | 启动 gRPC 模式 | 先确认 server 是否进入 gRPC 模式 |
| 排查 Rust / Python 边界 | Rust 接收 → PyBridge 背压 → RuntimeHandle | Rust 接收 RPC，PyBridge 处理背压，RuntimeHandle 接入 Python 生成器 |
| 对照 HTTP 主链路 | TokenizerManager → Scheduler → Detokenizer | 与 HTTP 路径共享核心职责 |
| 排查流式回程 | Detokenizer 回程、运行验证 | chunk 回到 TokenizerManager，再经 callback / channel 返回 Rust |
| 决定是否继续深读 | 与 HTTP 路径对照、导航 | 不需要 gRPC 时可回到 HTTP 全链路；需要网关时继续看 proto / bridge |

## 总览时序

```mermaid
sequenceDiagram
 participant Client
 participant Rust as sglang-grpc server.rs
 participant Bridge as PyBridge
 participant RH as RuntimeHandle
 participant TM as TokenizerManager
 participant SCH as Scheduler
 participant DET as Detokenizer

 Client->>Rust: gRPC TextGenerate
 Rust->>Bridge: submit_request(dict)
 Bridge->>RH: generate_request async gen
 RH->>TM: generate_request (同 HTTP)
 TM->>SCH: ZMQ TokenizedGenerateReqInput
 loop event_loop
 SCH->>DET: BatchTokenIDOutput
 end
 DET->>TM: BatchStrOutput
 TM-->>RH: yield chunk
 RH-->>Bridge: ChunkCallback
 Bridge-->>Rust: mpsc chunk
 Rust-->>Client: TextGenerateResponse stream
```

---

## 启动 gRPC 模式

**读法：** `run_server` 在 `server_args.grpc_mode=True` 时走 legacy Python gRPC 路径，加载 Rust `sglang-grpc` 扩展并绑定 `RuntimeHandle`。与 HTTP 共用同一套 Engine 子进程（Scheduler/Detokenizer）。

**源码锚点：**

```python
## 来源：python/sglang/launch_server.py L41-L45
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

```

**要点：** 默认 HTTP 分支不加载 Tonic；`--enable-metrics` 在 gRPC 模式可能依赖 sidecar（见 gRPC/Proto FAQ）。

---

## Rust 接收 RPC

**读法：** `SglangServiceImpl` 实现 proto 定义的 `TextGenerate` 双向流 handler：解析 `TextGenerateRequest`，构造 Python dict，经 `PyBridge` 提交。

**源码锚点：**

```rust
// 来源：rust/sglang-grpc/src/server.rs（TextGenerate handler 职责）
// 1. validate sampling_params
// 2. build_text_generate_dict(req)
// 3. bridge.submit_request(dict) -> stream TextGenerateResponse
```

**要点：** `rid` 缺省则 Rust 侧生成 UUID；`trace_headers` 透传到 Python 供分布式追踪。

---

## PyBridge 背压

**读法：** 每个请求创建 mpsc channel；Python `RuntimeHandle` 在独立线程/async 任务中 yield chunk，Rust 侧 `ChunkCallback` 非阻塞发送。channel 满时返回背压状态，避免 Python 产出快于 gRPC 发送导致 OOM。

**源码锚点：**

```rust
// 来源：rust/sglang-grpc/src/bridge.rs（概念结构）
// PyBridge::submit_request -> per-request mpsc::Sender<Chunk>
// Python callback 写入 chunk；Rust stream poll 读出
```

**要点：** 背压是跨语言边界的关键设计；HTTP SSE 由 asyncio 自然反压，gRPC 需显式 channel 容量。

---

## RuntimeHandle 进入 Python 运行时

**读法：** `grpc_bridge.RuntimeHandle` 把 proto dict 转为内部 `GenerateReqInput` 等价结构，调用 `TokenizerManager.generate_request`，从这里开始与 HTTP 的 TokenizerManager 后端路径相同。

**源码锚点：**

```python
## 来源：python/sglang/srt/entrypoints/grpc_bridge.py（职责）
# submit_request(proto_dict) -> asyncio generator of response chunks
# 内部：tokenizer_manager.generate_request(obj, ...)
```

**要点：** 从本边界起，可对照 [[SGLang-HTTP请求全链路]] 中的 TokenizerManager、Scheduler 和回程部分。

---

## TokenizerManager → Scheduler

**读法：** tokenize 后 ZMQ 发送 `TokenizedGenerateReqInput`；gRPC 与 HTTP 共用 `PortArgs` 与 socket 拓扑。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/tokenizer_manager.py（generate_request 核心）
 async def generate_request(self, obj, request=None):
 obj.normalize_batch_and_arguments()
 tokenized_obj = await self._tokenize_one_request(obj)
 self._send_one_request(tokenized_obj)
 async for response in self._wait_one_response(obj, request):
 yield response
```

---

## Scheduler 连续批处理

**读法：** 与 HTTP 路径一致：`event_loop_overlap` → `get_next_batch_to_run` → `run_batch` → `BatchTokenIDOutput` 到 Detokenizer。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1574-L1595
            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()
                # Opportunistic flush at the disable_overlap sync boundary:
                # forward_stream is idle (prev forward drained, next not launched),
                # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                if self.server_args.enable_unified_memory:
                    try:
                        self.token_to_kv_pool_allocator.flush_opportunistic()
                    except Exception:
                        pass

            # Launch the current batch
            if batch:
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
```

---

## Detokenizer → 回程

**读法：** Detokenizer 子进程增量解码；TokenizerManager 把 chunk dict 交给 RuntimeHandle；PyBridge 回调 Rust；Tonic 封装为 `TextGenerateResponse { text, meta_info, finished }`。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/detokenizer_manager.py L406-L419
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)
        indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
```

**要点：** `finished=True` 的 chunk 关闭 gRPC stream；客户端应处理 `meta_info` 中的 `cached_tokens` 等指标字段。

---

## 运行验证

不用启动完整服务，也可以先用源码定位确认这条链路是否仍成立：

```powershell
rg -n "grpc_mode|serve_grpc|Default mode: HTTP mode" sglang/python/sglang/launch_server.py
rg -n "type TextGenerateStream|submit_request|TextGenerateResponse" sglang/rust/sglang-grpc/src/server.rs
rg -n "mpsc::channel|ChunkCallback|submit_request" sglang/rust/sglang-grpc/src/bridge.rs
rg -n "class RuntimeHandle|generate_request|run_coroutine_threadsafe" sglang/python/sglang/srt/entrypoints/grpc_bridge.py
rg -n "TokenizedGenerateReqInput|BatchStrOutput|rid_to_state" sglang/python/sglang/srt/managers/tokenizer_manager.py
rg -n "event_loop_overlap|BatchTokenIDOutput|process_batch_result" sglang/python/sglang/srt/managers/scheduler.py
rg -n "BatchTokenIDOutput|BatchStrOutput|handle_batch_token_id_out" sglang/python/sglang/srt/managers/detokenizer_manager.py
```

预期现象：

- `launch_server.py` 能看到 `grpc_mode` 分支和默认 HTTP 分支并列，说明 gRPC 是入口差异，不是另一套 Scheduler。
- Rust `server.rs` 与 `bridge.rs` 能看到 `TextGenerateResponse`、`submit_request` 和 per-request channel，说明 stream 背压在跨语言边界显式处理。
- Python `grpc_bridge.py` 继续调用 `TokenizerManager.generate_request`，后续 TokenizerManager、Scheduler、Detokenizer 命中和 HTTP 路径相同的对象。

---

## 与 HTTP 路径对照

| 边界 | HTTP | gRPC |
|------|------|------|
| 入口 | FastAPI `/generate` | Tonic `TextGenerate` |
| 协议转换 | JSON → GenerateReqInput | Proto → dict → GenerateReqInput |
| 运行时 | TokenizerManager 起 | 同左（RuntimeHandle 包装） |
| 调度/执行/解码 | ZMQ 三进程 | **完全相同** |
| 响应 | SSE `data: {...}` | gRPC stream message |

深读 [[SGLang-gRPC-Proto]]。

---

## 导航

- [[SGLang-HTTP请求全链路]]
- [[SGLang-导读与总览]]
- [[SGLang-学习路径]]
- [[knowledge_maps/AI-Infra联合学习路径|双库联合路径]]
