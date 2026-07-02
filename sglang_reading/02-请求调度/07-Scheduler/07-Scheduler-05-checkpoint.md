---
type: batch-doc
module: 07-Scheduler
batch: "07"
doc_type: checkpoint
title: "Scheduler 验收清单"
tags:
 - sglang/batch/07
 - sglang/module/scheduler
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# Scheduler 验收清单

## 读者自测（不打开 sglang/）

- [x] 仅读本模块 sglang_reading，能口头说明 Scheduler 职责：GPU 子进程、Continuous Batching、驱动 TpWorker
- [x] 能画出 Scheduler 在 TokenizerManager ↔ TpWorker ↔ Detokenizer 之间的位置
- [x] 能说出 3 个核心函数/组件及其职责：
 - `get_next_batch_to_run` — prefill 优先、merge、decode 更新
 - `event_loop_overlap` — CPU/GPU 流水线
 - `SchedulerRequestReceiver.recv_requests` — ZMQ 收包 + broadcast
- [x] 能追踪一条 generate 请求：`handle_generate_request` → `waiting_queue` → prefill → `running_batch` → decode → `stream_output`
- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [x] 内嵌实码 + ETC 讲解（2026-07-02）

- [x] 源码范围覆盖 `scheduler.py`、`scheduler_pp_mixin.py`、`scheduler_components/` 关键路径
- [x] 来源注释路径/行号与当前源码一致（2026-07-02 走读）
- [x] 已更新 [[progress]] Scheduler 状态

## 核心结论（3 句话）

1. **Scheduler 是独立 GPU 进程内的调度中枢**，通过 ZMQ 收 Tokenized 请求，维护 `waiting_queue` / `running_batch`，每轮选出 `ScheduleBatch` 调用 TpWorker forward。
2. **默认 `event_loop_overlap` 将上一轮结果处理与当前 GPU forward 重叠**；PP / Disaggregation / MLX 走 `dispatch_event_loop` 分派的不同循环。
3. **Prefill 优先于 decode**；KV 不足时 `update_running_batch` 触发 retract，请求退回队列重调度。

## 遗留问题（后续专题）

- `PrefillAdder` / `SchedulePolicy` 内部算法 → 调度策略
- `Req` / `ScheduleBatch` 字段与 tensor 布局 → ScheduleBatch-IO
- Disaggregation mixin 队列与 KV 传输 → PD 分离
- `process_batch_result` 全部分支（spec、grammar、logprob）→ 与Sampling/21 交叉

## 内嵌源码统计（维护者）

| 文档 | 代码块数（约） | 说明 |
|------|----------------|------|
| README.md | 2 | 入口 + dispatch |
| 01-核心概念.md | 4 | 类定义、状态、ParallelState、prefill 优先 |
| 02-源码走读.md | 14 | 主走读 |
| 03-数据流与交互.md | 6 | IPC、数据流、rank0 收包 |
| 04-关键问题.md | 6 | FAQ 对比 |
| **合计** | **≥32 段** | **远超 15 段下限** |

合计内嵌源码行数：**约 450+ 行**（满足大模块 400+ 要求）。
