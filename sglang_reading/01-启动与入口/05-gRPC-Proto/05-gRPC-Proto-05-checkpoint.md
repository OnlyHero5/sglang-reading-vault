---
type: batch-doc
module: 05-gRPC-Proto
batch: "05"
doc_type: checkpoint
title: "gRPC/Proto 验收清单"
tags:
 - sglang/batch/05
 - sglang/module/grpc-proto
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# gRPC/Proto 验收清单

## 读者自测（不打开 sglang/）

- [x] 仅读本模块 sglang_reading，能口头说明 gRPC 模块职责：Proto 契约 + Rust Tonic 服务 + Python RuntimeHandle 桥接
- [x] 能画出 gRPC 客户端 → Tonic → PyBridge → TokenizerManager → Scheduler 的位置图
- [x] 能说出 3 个核心组件及其职责：
 - `SglangServiceImpl`（Rust Tonic handler）
 - `PyBridge` / `ChunkCallback`（跨语言 channel 与背压）
 - `RuntimeHandle`（Python 侧 submit_request / generate_request）
- [x] 能追踪一条 `TextGenerate(stream=true)` 从 RPC 到 token chunk 再回 gRPC stream 的路径
- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [x] 内嵌实码 + ETC 讲解（2026-07-02）

- [x] 对照 knowledge-graph 已补充 batch-05 节点（grpc_server、proto、sglang-grpc 等）
- [x] 来源注释路径/行号与当前 git 一致
- [x] 已更新 [[progress]] gRPC/Proto → ✅

## 核心结论（3 句话）

1. **`proto/sglang/runtime/v1/sglang.proto`** 是 SGLang gRPC 的单一契约，涵盖 typed Generate/Embed、OpenAI JSON pass-through 与 Admin RPC。
2. **`rust/sglang-grpc`** 用 Tonic 实现 servicer，通过 **PyBridge + mpsc** 把 Python TokenizerManager 的异步流暴露为 gRPC stream，并可选 Rust 原生 Tokenize。
3. **`--grpc-mode`** 当前走 **smg-grpc-servicer** 独立部署 + HTTP sidecar；**`SGLANG_ENABLE_GRPC`** 指向未来的 HTTP+gRPC 双栈，核心 Rust 扩展已就绪。

## 遗留问题

- `launch_server` 默认 HTTP 分支何时接线 `SGLANG_ENABLE_GRPC`？
- gRPC API Key 认证何时与 HTTP 对齐（`server.rs` TODO grpc-auth）？
- `SGLANG_TONIC_PAYLOAD` 何时提升为正式 CLI `--grpc-max-message-size`？

## 代码块统计（维护者）

| 文件 | 代码块数 | 约行数 |
|------|----------|--------|
| README.md | 1 | 35 |
| 01-核心概念.md | 4 | 75 |
| 02-源码走读.md | 12 | 280 |
| 03-数据流与交互.md | 6 | 120 |
| 04-关键问题.md | 8 | 130 |
| **合计** | **31** | **~640** |

满足 PLAN 要求：≥ 15 代码块、≥ 200 行。
