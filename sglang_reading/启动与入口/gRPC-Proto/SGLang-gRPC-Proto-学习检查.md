---
title: "gRPC-Proto · 学习检查"
type: exercise
framework: sglang
topic: "gRPC-Proto"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# gRPC-Proto · 学习检查

## 读者能做什么

- [ ] 能画出 `TextGenerateRequest -> build_text_generate_dict -> PyBridge -> RuntimeHandle -> GenerateReqInput -> TokenizerManager -> ResponseChunk -> TextGenerateResponse`。
- [ ] 能解释 `--grpc-mode`、`SGLANG_ENABLE_GRPC`、`encoder_only + grpc_mode` 三条路径的当前边界。
- [ ] 能说出 Proto 三类 RPC：typed native、OpenAI JSON pass-through、Admin/Ops。
- [ ] 能沿 `rid` 说明 channel、callback、abort 和 Python 请求对象如何关联。
- [ ] 能解释 `ChunkSendStatus.Ready/Pending/Closed` 对 Python producer 的影响。
- [ ] 能指出 `meta_info` 为什么是 JSON 编码后的 string map。
- [ ] 能用症状定位源码入口：缺 servicer、无 sidecar metrics、大消息失败、断连未 abort、tokenize fallback。

## 源码入口验收

| 问题 | 应能定位到 |
|------|------------|
| CLI 为什么不走 HTTP | `python/sglang/launch_server.py::run_server` |
| Proto 契约分层 | `proto/sglang/runtime/v1/sglang.proto::SglangService` |
| TextGenerate 字段映射 | `rust/sglang-grpc/src/utils/request_utils.rs::build_text_generate_dict` |
| Rust 到 Python 提交 | `rust/sglang-grpc/src/bridge.rs::PyBridge::submit_request` |
| Python 内部请求构造 | `python/sglang/srt/entrypoints/grpc_bridge.py::RuntimeHandle.submit_request` |
| stream 回包映射 | `rust/sglang-grpc/src/server.rs::text_generate` |
| 背压 | `try_send_chunk` 与 `_send_with_backpressure` |
| 断连取消 | `RequestAbortGuard` 与 `RuntimeHandle.abort` |
| legacy sidecar | `python/sglang/srt/entrypoints/grpc_server.py::serve_grpc` |

## 可执行检查

在本地源码树可运行：

```powershell
python -m py_compile `
  "sglang/python/sglang/launch_server.py" `
  "sglang/python/sglang/srt/server_args.py" `
  "sglang/python/sglang/srt/entrypoints/grpc_bridge.py" `
  "sglang/python/sglang/srt/entrypoints/grpc_server.py"
```

如果 Rust toolchain 可用，可补充：

```powershell
cargo test --manifest-path "sglang/rust/sglang-grpc/Cargo.toml"
```

读者不需要把这个专题理解成“能独立启动完整模型”的 smoke test。它的验收重点是协议入口、对象变形和失败边界。

## 场景复述

选择任意一个场景，能在 3 分钟内讲清楚：

| 场景 | 必须讲到 |
|------|----------|
| `TextGenerate(stream=true)` | `rid`、dict 映射、`GenerateReqInput`、`finish_reason`、`ResponseChunk::Finished` |
| 慢客户端 | bounded channel、`Pending`、on-ready、300 秒 backpressure timeout |
| 客户端断开 | stream drop、`RequestAbortGuard`、`bridge.abort`、`TokenizerManager.abort_request` |
| `--enable-metrics` 失败 | sidecar hook、servicer 版本、fail-fast |
| 大 JSON 请求 | `DEFAULT_GRPC_MAX_MESSAGE_SIZE`、`SGLANG_TONIC_PAYLOAD` |

## 迁移复盘

本专题不再以“贴了多少源码”验收。真正的通过标准是：

- 能把 gRPC 看成入口协议层，而不是独立推理后端。
- 能把当前 `--grpc-mode` 和 Native Rust gRPC 预留路径分开。
- 能用源码证明每个对象在哪里改变形态。
- 能把故障症状映射到具体边界，而不是只说“gRPC 有问题”。
