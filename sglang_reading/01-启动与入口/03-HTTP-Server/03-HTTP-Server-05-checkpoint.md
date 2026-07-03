---
type: batch-doc
module: 03-HTTP-Server
batch: "03"
doc_type: checkpoint
title: "HTTP Server 验收清单"
tags:
 - sglang/batch/03
 - sglang/module/http-server
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# HTTP Server 验收清单

> Git：`70df09b` | 图谱：`sglang/.understand-anything/knowledge-graph.json`（batch-01-initial，http_server/engine 待后续图谱更新 增量）

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明：HTTP 层是 FastAPI 薄路由，引擎逻辑在 `Engine._launch_subprocesses` + `TokenizerManager`
- [ ] 能画出本模块在全局架构中的位置：入口层（启动链路）→ HTTP/Engine（本模块）→ TokenizerManager（TokenizerManager）→ Scheduler（Scheduler）
- [ ] 能说出 3 个核心类/函数及其职责：
 - `launch_server` — 拉起子进程并启动 uvicorn/Granian
 - `Engine._launch_subprocesses` — Scheduler/Detokenizer/Tokenizer 启动总控
 - `generate_request`（HTTP）— Native API 入口，委托 `tokenizer_manager.generate_request`
- [ ] 能追踪 `POST /generate` 从 FastAPI 到 TokenizerManager 的路径（见 03-HTTP-Server-03-数据流与交互.md §5）
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 内嵌源码统计

| 文件 | 代码块数 | 约行数 |
|------|----------|--------|
| 03-HTTP-Server-00-MOC.md | 1 | 35 |
| 03-HTTP-Server-01-核心概念.md | 4 | 75 |
| 03-HTTP-Server-02-源码走读.md | 12 | 220 |
| 03-HTTP-Server-03-数据流与交互.md | 7 | 95 |
| 03-HTTP-Server-04-关键问题.md | 6 | 70 |
| **合计** | **30** | **~495** |

## 核心结论（3 句话）

1. `launch_server` = `Engine._launch_subprocesses`（子进程引擎）+ `_setup_and_run_http_server`（FastAPI 监听）；HTTP 与 Python `Engine()` 共享同一套子进程启动逻辑。
2. 所有推理请求（Native `/generate` 或 OpenAI `/v1/*`）最终汇聚到主进程 `TokenizerManager.generate_request`，再经 ZMQ 与 Scheduler/Detokenizer 交互。
3. `_GlobalState` + FastAPI `lifespan` 负责 HTTP 层运行时单例与 Serving handler 延迟初始化；warmup 线程在监听开始后探测直至 `ServerStatus.Up`。

## 遗留问题

- OpenAI Serving 协议转换细节 → **OpenAI API**
- `TokenizerManager.generate_request` 内部 ZMQ 消息格式 → **TokenizerManager、09**
- 全量图谱补全 http_server/engine 节点 → **gRPC/Proto** `/understand` 增量
