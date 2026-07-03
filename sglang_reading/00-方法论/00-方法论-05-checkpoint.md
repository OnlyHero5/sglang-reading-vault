---
type: batch-doc
module: 00-方法论
batch: "01"
doc_type: checkpoint
title: "阅读方法论 验收清单"
tags:
 - sglang/batch/01
 - sglang/module/methodology
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# 阅读方法论 验收清单

> Git：`70df09b` | 图谱：`sglang/.understand-anything/knowledge-graph.json`（batch-01-initial）

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 SGLang 是 LLM/VLM 高性能推理框架，Runtime 在 `srt`
- [ ] 能画出文档中的三层架构：文档/配置 → 入口 → 运行时核心
- [ ] 能说出 3 个核心入口及其职责：
 - `cli/main.py` — 子命令路由
 - `cli/serve.py` — LLM/diffusion 分发 + 调 launch_server
 - `launch_server.run_server` — HTTP/gRPC/Ray/Encoder 四选一
- [ ] 能追踪 `sglang serve --model-path M` 的代码路径（见 00-方法论-03-数据流与交互.md §4）
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 内嵌源码统计

| 文件 | 代码块数 | 约行数 |
|------|----------|--------|
| 00-方法论-00-MOC.md | 1 | 8 |
| 00-方法论-01-核心概念.md | 7 | 90 |
| 00-方法论-02-源码走读.md | 12 | 180 |
| 00-方法论-03-数据流与交互.md | 6 | 55 |
| 00-方法论-04-关键问题.md | 5 | 45 |
| **合计** | **31** | **~378** |

## 核心结论（3 句话）

1. SGLang monorepo 的核心是 `python/sglang/srt`（Runtime），`lang` 是可选 Frontend。
2. 推荐入口 `sglang serve` → `prepare_server_args` → `run_server`，默认启动 HTTP OpenAI 兼容服务。
3. 本 sglang_reading 项目采用 ETC 内嵌源码格式，读者无需打开 `sglang/` 目录。

## 遗留问题

- `prepare_server_args` / `ServerArgs` 字段全貌 → **启动链路**
- `http_server.launch_server` 内部进程树 → **HTTP Server**
- 全量 knowledge-graph（3600+ 文件）→ **gRPC/Proto** 运行 `/understand --language zh`
