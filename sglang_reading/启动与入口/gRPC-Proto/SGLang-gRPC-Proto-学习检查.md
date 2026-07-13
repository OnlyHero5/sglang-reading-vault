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
updated: 2026-07-11
---
# gRPC-Proto · 学习检查

## 读者能做什么

- [ ] 能画出 `TextGenerateRequest -> build_text_generate_dict -> PyBridge -> RuntimeHandle -> GenerateReqInput -> TokenizerManager -> ResponseChunk -> TextGenerateResponse`。
- [ ] 能解释 `--grpc-mode`、`SGLANG_ENABLE_GRPC`、`encoder_only + grpc_mode` 三条路径的当前边界。
- [ ] 能说出 Proto 三类 RPC：typed native、OpenAI JSON pass-through、Admin/Ops。
- [ ] 能沿 `rid` 说明 channel、callback、abort 和 Python 请求对象如何关联。
- [ ] 能解释 `ChunkSendStatus.Ready/Pending/Closed` 对 Python producer 的影响。
- [ ] 能说明一个 parked chunk、第二次违规发送和 `ChannelFull` 之间的状态关系。
- [ ] 能指出 `meta_info` 为什么是 JSON 编码后的 string map。
- [ ] 能区分正常 terminal、client drop、receiver closed、timeout 和显式 Abort RPC。
- [ ] 能说明未认证结论只针对 Native Tonic listener，legacy servicer 需要独立审计。
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

先做不导入 SGLang 包的语法检查：

```powershell
python -m py_compile `
  "sglang/python/sglang/launch_server.py" `
  "sglang/python/sglang/srt/server_args.py" `
  "sglang/python/sglang/srt/entrypoints/grpc_bridge.py" `
  "sglang/python/sglang/srt/entrypoints/grpc_server.py"
```

预期：四个文件均通过语法编译。这个检查不证明 SGLang 可以在当前 Windows 环境启动；完整 import 仍可能受 POSIX `resource`、FastAPI 或其他依赖限制。

再验证三条启动路径没有被概念混用：

```powershell
rg -n "encoder_only|grpc_mode|serve_grpc|launch_server" "sglang/python/sglang/launch_server.py"
rg -n "SGLANG_ENABLE_GRPC|SGLANG_GRPC_PORT|enable_grpc|grpc_port" "sglang/python/sglang/srt/server_args.py"
rg -n "start_server" "sglang/rust/sglang-grpc/src/lib.rs" "sglang/python/sglang"
```

预期：第一条显示 encoder、legacy gRPC、默认 HTTP 的互斥分发；第二条显示 Native 配置已解析和校验；第三条显示扩展定义存在，但默认 Python HTTP 启动链没有生产调用点。只有同时看到“配置”和“调用”才能声称 listener 已接线。

如果 Rust toolchain、`protoc` 和依赖可用，可补充：

```powershell
cargo test --manifest-path "sglang/rust/sglang-grpc/Cargo.toml"
```

预期：crate 完成编译并运行测试；这仍是 bridge/server 构建验证，不是完整模型 listener smoke test。失败时记录缺失的 toolchain、`protoc`、网络依赖或平台限制，不能把未收集到测试误报为通过。

读者不需要把这个专题理解成“能独立启动完整模型”的 smoke test。它的验收重点是协议入口、对象变形和失败边界。

## 场景复述

选择任意一个场景，能在 3 分钟内讲清楚：

| 场景 | 必须讲到 |
|------|----------|
| `TextGenerate(stream=true)` | `rid`、dict 映射、`GenerateReqInput`、`finish_reason`、`ResponseChunk::Finished` |
| 慢客户端 | bounded channel、一个 parked chunk、`Pending`、on-ready、第二次发送触发 `ChannelFull`、300 秒 timeout |
| 客户端断开 | stream drop 或 receiver closed、`RequestAbortGuard`、`bridge.abort`、5 秒 Python abort wait、`TokenizerManager.abort_request` |
| `--enable-metrics` 失败 | sidecar hook、servicer 版本、fail-fast |
| 大 JSON 请求 | `DEFAULT_GRPC_MAX_MESSAGE_SIZE`、`SGLANG_TONIC_PAYLOAD` |

## 迁移复盘

本专题不再以“贴了多少源码”验收。真正的通过标准是：

- 能把 gRPC 看成入口协议层，而不是独立推理后端。
- 能把当前 `--grpc-mode` 和 Native Rust gRPC 预留路径分开。
- 能用源码证明每个对象在哪里改变形态。
- 能把故障症状映射到具体边界，而不是只说“gRPC 有问题”。
