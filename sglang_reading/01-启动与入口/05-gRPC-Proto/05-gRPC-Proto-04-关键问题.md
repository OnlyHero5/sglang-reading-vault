---
type: batch-doc
module: 05-gRPC-Proto
batch: "05"
doc_type: faq
title: "gRPC/Proto：关键问题"
tags:
 - sglang/batch/05
 - sglang/module/grpc-proto
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# gRPC/Proto：关键问题

---

## Q1：`--grpc-mode` 和 `SGLANG_ENABLE_GRPC` 有什么区别？

| | `--grpc-mode` | `SGLANG_ENABLE_GRPC` |
|---|---------------|----------------------|
| 配置方式 | CLI 标志 | 环境变量（暂无 CLI） |
| 与 HTTP 关系 | **互斥**，只开 gRPC | 设计为 **与 HTTP 并存** |
| 实现入口 | `grpc_server.serve_grpc` → smg-grpc-servicer | `sglang.srt.grpc._core.start_server` |
| 成熟度 | 可用（需安装 servicer） | 校验逻辑已有，默认 HTTP 分支未接线 |
| Sidecar | servicer ≥0.5.3 支持 metrics/profile | 待定 |

**易错：** 以为 `SGLANG_ENABLE_GRPC=1` 会自动启动 gRPC——当前仍需 `--grpc-mode` 或外部 servicer 集成。

**Comment：** 选型时先确认部署形态：纯 gRPC 用 `--grpc-mode`；未来 HTTP+gRPC 并存需等 `SGLANG_ENABLE_GRPC` 接线完成。

---

## Q2：为什么 gRPC 核心用 Rust 而不是纯 Python grpcio？

1. **Tonic + Tokio** 流式性能与 backpressure 控制成熟。
2. **Tokenize/Detokenize** 可走 Rust `tokenizers` crate，释放 GIL。
3. **与 model-gateway 同栈**：gateway 已是 Rust，Proto 类型可共享。
4. Python 保留 **TokenizerManager 异步语义**，通过 PyO3 薄桥接，避免重写调度逻辑。

**Comment：** Rust 侧负责高性能 stream 与 tokenizers crate；Python Scheduler 逻辑无需 fork 一份到 Rust。

---

## Q3：没有安装 `smg-grpc-servicer` 会怎样？

**Explain：** `--grpc-mode` 在 import 阶段即失败，错误信息指向 pip 安装命令。

**Code（正确做法）：**

```bash
pip install smg-grpc-servicer[sglang]
sglang serve --grpc-mode --model-path meta-llama/Llama-3.2-1B-Instruct
```

**Code（错误现象）：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L160-L166
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e
```

**Comment：** 未安装 servicer 时 import 即失败，错误信息含 pip 命令——CI 镜像需预装 `smg-grpc-servicer[sglang]`。

---

## Q4：`--enable-metrics` + 旧版 servicer 为何直接报错？

**Explain：** metrics 依赖 HTTP sidecar 的 `/metrics`；旧 servicer 不接受 `on_request_manager_ready`，sidecar 永远不会启动，若静默继续会导致「以为有 metrics 实际没有」。

**Code（会抛 RuntimeError 的情况）：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L235-L244
    elif server_args.enable_metrics:
        # User explicitly asked for metrics but the installed servicer can't
        # start the sidecar that serves them — fail loud rather than silently
        # produce a server with no /metrics endpoint.
        raise RuntimeError(
            "--enable-metrics requires smg-grpc-servicer ≥ 0.5.3 (the version "
            "that accepts 'on_request_manager_ready'); installed version "
            "lacks the hook so the HTTP sidecar would never start. Upgrade "
            "smg-grpc-servicer or remove --enable-metrics."
        )
```

**Code（无 metrics 时仅 warning）：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L246-L251
        logger.warning(
            "Installed smg-grpc-servicer does not accept "
            "'on_request_manager_ready'; HTTP sidecar disabled "
            "(no /metrics, /start_profile, /stop_profile). "
            "Upgrade smg-grpc-servicer to ≥ 0.5.3 to enable it."
        )
```

**Comment：** metrics 依赖 sidecar HTTP；旧 servicer 无 hook 时**必须** fail-fast，避免「配置了 metrics 但 scrape 空」的 silent 故障。

---

## Q5：Classify RPC 为什么走 `embed` 路径？

**Explain：** 在 SGLang 内部，分类与嵌入共用 `EmbeddingReqInput` 与同一 forward 路径；Proto 层区分语义，Rust 统一 `submit_request(..., "embed", ...)`。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/server.rs L446-L451
 let req_dict = build_classify_dict(&rid, &req);
 let mut receiver = self
 .bridge
 .submit_request(&rid, "embed", req_dict)
 ...
```

客户端若期望分类 label，需从 `meta_info` 或下游模型 head 解析，而非 Proto 单独字段。

**Comment：** Classify 在 SGLang 内部与 Embedding 共用 forward；Proto 语义区分由 Rust servicer 层完成，Python 侧无单独 classify head。

---

## Q6：gRPC 消息大小限制怎么调？

默认 **64 MiB**（非 Tonic 默认 4 MiB）。临时可通过环境变量：

```bash
export SGLANG_TONIC_PAYLOAD=134217728 # 128 MiB
```

**Code：**

```rust
// 来源：rust/sglang-grpc/src/server.rs L29-L31, L37-L57
pub const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

fn resolve_max_message_size() -> usize {
 match std::env::var("SGLANG_TONIC_PAYLOAD") {
 Ok(raw) => match raw.parse::<usize>() {
 Ok(n) if n > 0 => n,
 _ => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
 },
 Err(_) => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
 }
}
```

**Comment：** 多模态或长 context 请求可能超默认 4 MiB Tonic 限制；128 MiB 为常见生产配置起点。

---

## Q7：客户端断开连接后服务器会泄漏请求吗？

**Explain：** 不会（设计上）。`RequestAbortGuard` 在 gRPC stream drop 时调用 `abort`；channel 关闭也会记录 `TerminalError::ClientDisconnected`。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/bridge.rs L596-L604
 Err(TrySendError::Closed(_)) => {
 close_channel_with_error(
 py, rid, state, runtime_handle,
 TerminalError::ClientDisconnected { rid: rid.into() },
 );
 Ok(ChunkSendStatus::Closed)
 }
```

**Comment：** gRPC stream drop 必须 abort 上游 Req，否则 Scheduler KV 泄漏；与 HTTP Q7 disconnect 路径对称。

---

## Q8：Proto 修改后需要 rebuild 什么？

1. **Rust 扩展**：`cargo build` / `pip install -e .` 触发 `build.rs` 重新编译 proto。
2. **外部 client**：gateway、`smg_grpc_client` 等需同步 proto 版本。
3. **Python**：无 generated pb2（typed 路径在 Rust）；OpenAI pass-through 不受 message 字段变更影响。

**Comment：** 改 proto 后需同步 rebuild Rust wheel 与外部 gateway client，版本漂移会导致 decode 失败。

---

## Q9：gRPC 与 HTTP OpenAI API 选哪个？

| 场景 | 推荐 |
|------|------|
| curl / 浏览器 / 通用工具链 | HTTP OpenAI（OpenAI API） |
| 低延迟流式、二进制 payload、多语言强类型 client | gRPC native RPC |
| 已有 OpenAI SDK 代码 | HTTP；或 gRPC `ChatComplete` + JSON body |
| K8s + Prometheus | HTTP 或 gRPC+sidecar |
| 需要 API Key 认证 | **目前仅 HTTP**（gRPC auth TODO） |

**Comment：** 内网 gRPC 通常靠 mTLS/网络隔离；公网暴露仍需 Gateway 或 sidecar 做 auth。

---

## Q10：`srt/grpc/` 目录为什么几乎是空的？

**Explain：** 历史命名空间占位；实际二进制在 `sglang.srt.grpc._core`（Rust cdylib）。`__init__.py` 仅注释：

```python
# 来源：python/sglang/srt/grpc/__init__.py
# SGLang gRPC module
```

导入应使用 `from sglang.srt.grpc._core import start_server`，而非期望纯 Python 实现。

**Comment：** `srt/grpc/` 为命名空间占位；实际 binary 在 Rust cdylib `_core`，与 `rust/sglang-grpc` 同栈。

---

## 易错点速查

1. ❌ gRPC 端口与 HTTP `--port` 相同 → ✅ `SGLANG_GRPC_PORT` 默认 `port+10000`
2. ❌ 在 gRPC 主端口 scrape `/metrics` → ✅ 访问 sidecar `port+1`
3. ❌ 把 `meta_info` 值当纯 string → ✅ 按 JSON parse
4. ❌ Stream RPC 忽略 `finished` 字段 → ✅ 末包 `finished=true` 才结束
5. ❌ 未消费 stream 导致 channel full → ✅ 客户端持续 recv 或 abort

---

## 验证建议（零基础可试）

以下 **4 条不要求 GPU**，可在读完本模块 FAQ 后立刻动手，确认 gRPC 入口、依赖与配置常量与文档一致。

1. **操作：** `sglang serve --help 2>&1 | rg -i "grpc-mode|grpc"`（Windows 可用 `findstr /i "grpc"`）。 
 **预期现象：** 出现 `--grpc-mode` 说明——与 HTTP 分支互斥的 gRPC 专用启动路径；**不会**因设置 `SGLANG_ENABLE_GRPC=1` alone 而自动起 gRPC。 
 **对应文档节：** Q1 `--grpc-mode` vs `SGLANG_ENABLE_GRPC`

2. **操作：** `pip show smg-grpc-servicer`（未安装则 `pip install smg-grpc-servicer[sglang]` 前先记当前报错）。 
 **预期现象：** 已安装时显示 `Version:`；若 `<0.5.3` 且计划 `--enable-metrics`，应对照 Q4 的 fail-fast / sidecar 说明。 
 **对应文档节：** Q3 未安装 servicer、Q4 metrics 与 sidecar 版本

3. **操作：** 在 sglang 源码树执行 
 `rg "DEFAULT_GRPC_MAX_MESSAGE_SIZE|SGLANG_TONIC_PAYLOAD|SGLANG_GRPC_PORT" rust/sglang-grpc python/sglang -n` 
 **预期现象：** Rust 侧默认 64 MiB、`SGLANG_TONIC_PAYLOAD` 解析逻辑，以及 gRPC 端口相对 HTTP `port+10000` 的约定。 
 **对应文档节：** Q6 消息大小、易错点速查 #1–#2

4. **操作：** `rg "\.proto" sglang --glob "*.proto" -l | head -5`（或 `find . -name "*.proto" | head`）。 
 **预期现象：** 列出 Proto 源文件路径；改 proto 后需 rebuild Rust 扩展（Q8），Python 无独立 pb2 生成物。 
 **对应文档节：** Q8 Proto 修改 rebuild 清单、Q10 `srt/grpc/` 占位 vs `_core` cdylib
