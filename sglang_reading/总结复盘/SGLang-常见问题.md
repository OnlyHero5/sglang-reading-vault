---
title: "SGLang 常见问题"
type: troubleshooting
framework: sglang
topic: "总结复盘"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# SGLang 常见问题

> 新读者常见问题：先定位责任边界，再进入对应概念、源码和验证。

## 你为什么要读

这页处理的是读完导读后最常见的“方向性故障”：从哪读、HTTP 与 gRPC 怎么分、缓存在哪生效、旧入口该换到哪里。它不会重复专题细节，而是像总服务台一样把问题送到正确楼层，避免你拿着 Scheduler 的问题去翻 HTTP 路由。

本文回答完成导读后复盘时最常遇到的定位问题。每问按问题、源码锚点和读法收束。

---

## Q1 · 我应该从哪读起？

**读法：** 先回到 [[SGLang-导读与总览]] 建立 monorepo 与三进程心智模型，再沿 HTTP 全链路串请求路径，最后按需查概念与文件地图。若时间有限，走“项目总览 → 架构分层 → 全链路追踪 → 入口、调度与模型执行”；若做 gRPC 部署，额外读 [[SGLang-gRPC请求全链路]] 与 gRPC/Proto 专题。

**源码锚点：**

```python
## 来源：python/sglang/cli/serve.py L121-L128
        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            server_args = prepare_server_args(dispatch_argv)

            run_server(server_args)
```

**要点：**

- 一切服务进程从 `sglang serve` 进入；`prepare_server_args` 是 CLI 与 Runtime 的边界。
- 读代码时以 `run_server` 分发为轴：HTTP（默认）/ gRPC / Ray / Encoder。
- 本页只负责分流；进入具体问题后，继续阅读对应概念、数据流、源码走读或排障指南。

---

## Q2 · HTTP 与 gRPC 启动路径有何不同？

**读法：** 默认路径走 FastAPI `http_server.launch_server`；`--grpc-mode` 或 native Rust gRPC（`SGLANG_ENABLE_GRPC`）走 `rust/sglang-grpc`，由 Tonic 接收 RPC 后经 PyBridge 调用 Python `RuntimeHandle`，再复用同一套 TokenizerManager → Scheduler 链路。HTTP 由 FastAPI 路由接收 JSON；gRPC 由 `SglangServiceImpl.text_generate` 接收 proto。二者在 TokenizerManager 之后完全汇合。

**源码锚点：**

```python
## 来源：python/sglang/launch_server.py L41-L51
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)
```

**要点：**

- `grpc_mode` 为 legacy SMG path；新部署倾向 native Rust gRPC + `RuntimeHandle`（见 gRPC/Proto）。
- Ray 模式仍用 HTTP 入口，仅 backend 换 Ray worker pool。
- `encoder_only` 分支用于 PD encoder 分离，与 LLM 主路径正交。

---

## Q3 · 主进程与子进程如何分工？

**读法：** `Engine._launch_subprocesses` 在主进程启动 TokenizerManager（async event loop），并 spawn Scheduler、Detokenizer 两个子进程。HTTP/gRPC 入口、tokenize、流式响应组装都在主进程；GPU batch 调度与 forward 在 Scheduler 子进程；token id → 文本在 Detokenizer 子进程。三者通过 ZMQ + `io_struct.py` 定义的消息类型通信——这是理解「为什么 Detokenizer 是回程第一站」的关键。

**源码锚点：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L2494-L2506
    # Launch subprocesses
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )
```

**要点：**

- 主进程不跑 ModelRunner；GPU 计算仅在 Scheduler 子进程的 TP Worker 内。
- `port_args` 分配 ZMQ socket 端口，Tokenizer ↔ Scheduler ↔ Detokenizer 各用不同 socket。
- gRPC 的 `RuntimeHandle` 仍绑定主进程 TokenizerManager，不改变三进程拓扑。

---

## Q4 · RadixAttention / 前缀缓存在哪生效？

**读法：** 前缀匹配发生在 Scheduler 收请求后、`get_next_batch_to_run` 之前：`Req` 构造时调用 `RadixCache.match_prefix`，命中则缩短 extend 长度、提高 `cached_tokens`。KV 写入在 ModelRunner forward 的 RadixAttention 层完成。因此“缓存命中”由 Scheduler 决策与 ModelRunner 执行共同完成，不是 HTTP 层行为。

**源码锚点：**

```python
## 来源：python/sglang/srt/mem_cache/radix_cache.py L355-L365
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
```

**要点：**

- `page_aligned` 与 `page_size` 决定 radix 树 key 粒度；EAGLE 投机用 bigram 视图。
- 命中后 `Req` 的 `prefix_indices` 指向已有 KV slot，extend 跳过对应 token。
- 专题深度见RadixAttention–KV Cache；概念摘要见 [[SGLang-关键概念]]。

---

## Q5 · Continuous Batching 的核心循环在哪？

**读法：** Scheduler 的 `event_loop_normal` 或 `event_loop_overlap` 是吞吐主轴：recv 请求 → 组 batch → `run_batch` → `process_batch_result` → 发 token 到 Detokenizer。overlap 版用 `result_queue` 让 GPU forward 与 CPU 后处理错开一拍。读 Scheduler 时优先找这三个函数，而非从 ModelRunner 反推。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L1521-L1540
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
```

**要点：**

- `get_next_batch_to_run` 合并 waiting prefill 与 running decode，是 continuous batching 的组 batch 入口。
- `process_batch_result` 打包 `BatchTokenIDOutput` ZMQ 发往 Detokenizer。
- 显存不足时 `schedule_policy` 触发 retract，踢出低优先级 decode 请求。

---

## Q6 · 如何在语义知识库中快速定位？

**读法：** 首次学习从 [[SGLang学习指南]] 或 [[SGLang-导读与总览]] 进入；按主题深读时使用 [[SGLang-源码地图]]；按故障症状进入 [[SGLang-生产排障]]；需要动态筛选时使用 [[SGLang内容.base]]。

**要点：**

- 文件名直接表达主题和职责，不需要记忆排序编号。
- Backlinks 与 Local Graph 用于查看当前专题的上下游。
- [[SGLang-综合学习检查]] 用能力和实验验收，而不是阅读篇数。

---

## Q7 · 读完索引层后如何深潜某一专题？

**读法：** 复盘层不重复各专题的深度内容。确定兴趣点后，直接进入 [[SGLang-KV-Cache]]、[[SGLang-Speculative]] 或 [[SGLang-model-gateway]]。[[SGLang-学习路径]] 的主题深读索引给出系统职责与专题入口的对应关系。

**源码锚点：**

```python
## 来源：python/sglang/srt/managers/scheduler.py L2586-L2590
    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        self.process_pending_chunked_abort()

        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
```

**要点：**

- 索引层教会「去哪找」；专题目录教会「怎么读」。
- [[SGLang-源码地图]] 按 architecture layer 列 file 节点，适合查具体文件名。
- [[SGLang-复杂度热点]] 标记 `complexity: complex` 节点，优先深读。
