---
type: batch-doc
module: 11-ModelRunner
batch: "11"
doc_type: checkpoint
title: "ModelRunner 验收清单"
tags:
 - sglang/batch/11
 - sglang/module/model-runner
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# ModelRunner 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 ModelRunner / TpModelWorker 职责
- [ ] 能画出 Scheduler → TpModelWorker → ModelRunner → Model 的位置
- [ ] 能说出 3 个核心类：`ForwardBatch`、`ModelRunner`、`TpModelWorker` 及其职责
- [ ] 能追踪 decode step：`forward_batch_generation` → `init_new` → `forward` → `sample`
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. **TpModelWorker** 是 Scheduler 在 TP 进程内的唯一执行入口，负责把 `ScheduleBatch` 转为 `ForwardBatch` 并调用 **ModelRunner.forward**。
2. **ModelRunner** 统一管理权重加载、KV 内存池、Attention 后端与 CUDA Graph，是 GPU forward 的 orchestrator。
3. **ForwardMode** 决定 prefill/decode/verify 等路径，decode 固定 shape 时优先 CUDA Graph replay 以降低 launch 开销。

## 代码块统计

| 文件 | 代码块数 | 约行数 |
|------|---------|--------|
| 11-ModelRunner-00-MOC.md | 1 | 28 |
| 11-ModelRunner-01-核心概念.md | 4 | 95 |
| 11-ModelRunner-02-源码走读.md | 10 | 210 |
| 11-ModelRunner-03-数据流与交互.md | 5 | 85 |
| 11-ModelRunner-04-关键问题.md | 7 | 95 |
| **合计** | **27** | **~513** |

## 遗留问题

- `_forward_raw` 内 Graph/Eager 分支细节见 `runner/` 子目录，可与Attention Attention 后端对照阅读。
- MindSpore / NPU 专用 `mindspore_runner.py` 未在本模块展开。
