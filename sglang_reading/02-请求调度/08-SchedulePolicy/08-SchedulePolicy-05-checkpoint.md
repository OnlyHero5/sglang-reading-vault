---
type: batch-doc
module: 08-SchedulePolicy
batch: "08"
doc_type: checkpoint
title: "调度策略 验收清单"
tags:
 - sglang/batch/08
 - sglang/module/schedule-policy
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# 调度策略 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明本模块职责
- [ ] 能画出本模块在全局架构中的位置
- [ ] 能说出 3 个核心类/函数及其职责（文档中均有内嵌代码）
- [ ] 能追踪一条典型请求经过本模块的路径（文档中有逐步讲解）
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. **SchedulePolicy** 负责对 `waiting_queue` 排序（LPM/DFS/FCFS 等），并在批内前缀场景 deprioritize 重复请求以提高 Radix 命中率。
2. **PrefillAdder** 在 KV/SWA/Mamba 多层预算内逐个准入 prefill 请求，支持分块 prefill、host load back 与优先级抢占。
3. **PrefillDelayer**（跨 rank）与 **MinFreeSlotsDelayer**（单 rank）是可选延迟层，分别优化 overlap decode batch 利用率与高成本准入的批量化。

## 遗留问题

- `add_one_req_ignore_eos` 的 min-free-token 模拟逻辑较复杂，是否与 hybrid SWA 路径完全对齐需结合KV Cache KV 文档交叉验证。
- DSA prefill context parallel 强制 `can_run_list >= 1` 限制是否为临时 workaround，需跟踪上游 issue。

## 内嵌源码统计

| 文件 | 代码块数（约） |
|------|----------------|
| 08-SchedulePolicy-00-MOC.md | 1 |
| 08-SchedulePolicy-01-核心概念.md | 3 |
| 08-SchedulePolicy-02-源码走读.md | 16 |
| 08-SchedulePolicy-03-数据流与交互.md | 6 |
| 08-SchedulePolicy-04-关键问题.md | 5 |
| **合计** | **≥ 31 段，> 400 行** |
