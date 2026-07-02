---
type: batch-doc
module: 27-model-gateway
batch: "27"
doc_type: checkpoint
title: "model-gateway 验收清单"
tags:
 - sglang/batch/27
 - sglang/module/model-gateway
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# model-gateway 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能说明 gateway 是 Rust Axum 代理层，不做 GPU forward
- [ ] 能画出 Client → smg → WorkerRegistry/Policy → srt worker 架构图
- [ ] 能说出 `AppState`、`RouterTrait::route_chat`、`select_worker_for_model` 的职责
- [ ] 能追踪 `/v1/chat/completions` 逐步路径（03-数据流与交互.md）
- [ ] 能解释 PD readiness 的双 worker 要求

## 维护者检查

- [x] 覆盖关键 file：`server.rs`, `routers/factory.rs`, `routers/http/router.rs`, `routers/router_manager.rs`, `core/worker_registry.rs`, `core/worker.rs`
- [x] 来源路径与 git `70df09b` 一致
- [x] **03-数据流与交互.md 已从薄稿扩写为完整 ETC + Mermaid**
- [ ] 已更新 [[progress]]（由 P8 整合）

## 验证统计（2026-07-02 人工复核）

| 文件 | ETC 段数 | 内嵌代码行数 |
|------|----------|-------------|
| README.md | 1 | 8 |
| 01-核心概念.md | 5 | 58 |
| 02-源码走读.md | 12 | 185 |
| 03-数据流与交互.md | 8 | 78 |
| 04-关键问题.md | 7 | 52 |
| **合计** | **33** | **~381** |

- ETC 段数 ≥ 15：✅（33）
- 代码行数 ≥ 200：✅（~381）
- 03 完整：✅（原缺失/薄稿已替换）

## 核心结论（3 句话）

1. smg 用 Axum 暴露 OpenAI 兼容 API，handler 薄层委托 `RouterTrait` 实现。
2. `WorkerRegistry` + Policy + HashRing 选 healthy worker，reqwest/tonic 反向代理到 srt。
3. Regular/PD/OpenAI × HTTP/gRPC 由 RouterFactory/RouterManager 组合，readiness 反映 worker 池状态。

## 遗留问题

- gRPC pipeline stages 细节可另开子批（本模块以 HTTP Regular 主路径为主）。
- mesh/rate-limit WASM 模块仅架构提及，未逐文件走读。
