---
type: batch-doc
module: 02-启动链路
batch: "02"
doc_type: checkpoint
title: "启动链路 验收清单"
tags:
 - sglang/batch/02
 - sglang/module/launch
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# 启动链路 验收清单

> Git：`70df09b` | 图谱：`sglang/.understand-anything/knowledge-graph.json`（batch-01-initial，含 launch_server/cli 节点）

## 读者自测（不打开 sglang/）

- [x] 仅读本模块 sglang_reading，能口头说明本模块职责：CLI 解析 argv → 加载插件 → 构造 ServerArgs → run_server 四路分发
- [x] 能画出本模块在全局架构中的位置：用户 shell → cli/ → server_args → launch_server → Runtime 入口
- [x] 能说出 3 个核心类/函数及其职责（文档中均有内嵌代码）：
 - `prepare_server_args` — argv → ServerArgs 工厂
 - `run_server` — HTTP/gRPC/Ray/Encoder 四路分发
 - `load_plugins` — entry_points 发现 + HookRegistry.apply_hooks
- [x] 能追踪 `sglang serve --model-path M --tp-size 2` 的完整启动路径（见 03-数据流与交互.md §4）
- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [x] 内嵌实码 + ETC 讲解（2026-07-02）

- [x] 对照 knowledge-graph 无遗漏 batch-02 关键 file 节点（launch_server、cli/main、cli/serve、server_args、plugins）
- [x] 内嵌代码来源注释路径/行号与 commit 70df09b 一致
- [x] 已更新 [[progress]]

## 内嵌源码统计

| 文件 | 代码块数 | 约行数 |
|------|----------|--------|
| README.md | 1 | 37 |
| 01-核心概念.md | 4 | 95 |
| 02-源码走读.md | 12 | 285 |
| 03-数据流与交互.md | 6 | 95 |
| 04-关键问题.md | 5 | 75 |
| **合计** | **28** | **~587** |

## 核心结论（3 句话）

1. `sglang serve` 经 `cli/main → cli/serve → prepare_server_args → run_server` 完成启动；默认走 HTTP 路径。
2. `ServerArgs` 是 Annotated dataclass，CLI 参数自动生成；`__post_init__` 做交叉校验与模型特定默认值推导。
3. `load_plugins()` 在参数解析前执行，通过 HookRegistry 对 Runtime 代码 monkey-patch，支持硬件平台与通用扩展。

## 遗留问题

- `http_server.launch_server` 内部如何 fork Scheduler/Worker 进程树 → **HTTP Server**
- `grpc_server.serve_grpc` 与 Rust gRPC sidecar 演进 → **gRPC/Proto**
- `ServerArgs.__post_init__` 中模型特定 backend 推导全貌 → **ModelRunner–Quantization** 按需展开
- `PortArgs.init_new` 如何分配 ZMQ/NCCL 端口 → **TokenizerManager–Scheduler**
- diffusion 路径（`multimodal_gen` CLI）→ **multimodal_gen**
