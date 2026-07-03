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
updated: 2026-07-02
---
# Scheduler 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 Scheduler 职责：GPU 子进程、Continuous Batching、驱动 TpWorker
- [ ] 能画出 Scheduler 在 TokenizerManager ↔ TpWorker ↔ Detokenizer 之间的位置
- [ ] 能说出 3 个核心函数/组件及其职责：
 - `get_next_batch_to_run` — prefill 优先、merge、decode 更新
 - `event_loop_overlap` — CPU/GPU 流水线
 - `SchedulerRequestReceiver.recv_requests` — ZMQ 收包 + broadcast
- [ ] 能追踪一条 generate 请求：`handle_generate_request` → `waiting_queue` → prefill → `running_batch` → decode → `stream_output`
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. **Scheduler 是独立 GPU 进程内的调度中枢**，通过 ZMQ 收 Tokenized 请求，维护 `waiting_queue` / `running_batch`，每轮选出 `ScheduleBatch` 调用 TpWorker forward。
2. **默认 `event_loop_overlap` 将上一轮结果处理与当前 GPU forward 重叠**；PP / Disaggregation / MLX 走 `dispatch_event_loop` 分派的不同循环。
3. **Prefill 优先于 decode**；KV 不足时 `update_running_batch` 触发 retract，请求退回队列重调度。

## 遗留问题（后续专题）

- `PrefillAdder` / `SchedulePolicy` 内部算法 → 调度策略
- `Req` / `ScheduleBatch` 字段与 tensor 布局 → ScheduleBatch-IO
- Disaggregation mixin 队列与 KV 传输 → PD 分离
- `process_batch_result` 全部分支（spec、grammar、logprob）→ 与Sampling/21 交叉
