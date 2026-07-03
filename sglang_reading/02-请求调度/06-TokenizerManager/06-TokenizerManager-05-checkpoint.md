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
updated: 2026-07-02
---
# TokenizerManager 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 TokenizerManager 职责（分词、ZMQ 转发、流式聚合、控制面 Mixin）
- [ ] 能画出 TokenizerManager 在 HTTP → Scheduler → Detokenizer 闭环中的位置
- [ ] 能说出 3 个核心类/函数及其职责：
 - `TokenizerManager.generate_request` — 数据面入口
 - `ReqState` + `handle_loop` — 异步收发包
 - `TokenizerControlMixin` — 控制面 FanOut
- [ ] 能追踪一条流式生成请求：API → 分词 → ZMQ → Detokenizer 回传 → `_wait_one_response` yield
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. **TokenizerManager 是 API 与 Scheduler 之间的「前台」**：负责分词/校验/LoRA 解析，经 ZMQ 发送 `Tokenized*ReqInput`，并通过 `handle_loop` 接收 Detokenizer 输出唤醒 `generate_request`。
2. **数据面（rid 多路复用）与控制面（FanOutCommunicator 广播）分离**，分别由主类与 `TokenizerControlMixin` 实现，权重更新通过 RWLock 与 pause 机制与推理互斥。
3. **多 HTTP Worker 模式下** `MultiTokenizerRouter` 聚合请求并按 `http_worker_ipc` 将结果路由回对应 `TokenizerWorker`，pause/continue 必须广播以保证全局一致。

## 遗留问题

- `io_struct.py` 全量字段需结合ScheduleBatch-IO 阅读
- Detokenizer 内部 decode 状态机留待后续图谱更新
- Scheduler 如何将 `TokenizedGenerateReqInput` 并入 `ScheduleBatch` 留待后续图谱更新
