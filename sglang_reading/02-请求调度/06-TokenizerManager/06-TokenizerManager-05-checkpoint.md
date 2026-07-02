---
type: batch-doc
module: 06-TokenizerManager
batch: "06"
doc_type: checkpoint
title: "TokenizerManager 验收清单"
tags:
 - sglang/batch/06
 - sglang/module/tokenizer-manager
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# TokenizerManager 验收清单

## 读者自测（不打开 sglang/）

- [x] 仅读本模块 sglang_reading，能口头说明 TokenizerManager 职责（分词、ZMQ 转发、流式聚合、控制面 Mixin）
- [x] 能画出 TokenizerManager 在 HTTP → Scheduler → Detokenizer 闭环中的位置
- [x] 能说出 3 个核心类/函数及其职责：
 - `TokenizerManager.generate_request` — 数据面入口
 - `ReqState` + `handle_loop` — 异步收发包
 - `TokenizerControlMixin` — 控制面 FanOut
- [x] 能追踪一条流式生成请求：API → 分词 → ZMQ → Detokenizer 回传 → `_wait_one_response` yield
- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [x] 内嵌实码 + ETC 讲解（2026-07-02）

- [x] 对照 knowledge-graph 无遗漏 `tokenizer_manager.py` 及 mixin 节点
- [x] 来源注释路径/行号与 git `70df09b` 一致
- [x] 已更新 [[progress]] TokenizerManager → ✅

## 核心结论（3 句话）

1. **TokenizerManager 是 API 与 Scheduler 之间的「前台」**：负责分词/校验/LoRA 解析，经 ZMQ 发送 `Tokenized*ReqInput`，并通过 `handle_loop` 接收 Detokenizer 输出唤醒 `generate_request`。
2. **数据面（rid 多路复用）与控制面（FanOutCommunicator 广播）分离**，分别由主类与 `TokenizerControlMixin` 实现，权重更新通过 RWLock 与 pause 机制与推理互斥。
3. **多 HTTP Worker 模式下** `MultiTokenizerRouter` 聚合请求并按 `http_worker_ipc` 将结果路由回对应 `TokenizerWorker`，pause/continue 必须广播以保证全局一致。

## 遗留问题

- `io_struct.py` 全量字段需结合ScheduleBatch-IO 阅读
- Detokenizer 内部 decode 状态机留待后续图谱更新
- Scheduler 如何将 `TokenizedGenerateReqInput` 并入 `ScheduleBatch` 留待后续图谱更新

## 内嵌源码统计（维护者）

| 文件 | 代码块数 | 约行数 |
|------|----------|--------|
| README.md | 1 | 35 |
| 01-核心概念.md | 5 | 95 |
| 02-源码走读.md | 12 | 210 |
| 03-数据流与交互.md | 6 | 85 |
| 04-关键问题.md | 6 | 75 |
| **合计** | **30** | **~500** |

满足 PLAN 要求：≥ 15 段、≥ 200 行。
