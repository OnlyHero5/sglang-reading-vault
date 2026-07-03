---
type: batch-doc
module: 09-ScheduleBatch-IO
batch: "09"
doc_type: checkpoint
title: "ScheduleBatch-IO 验收清单"
tags:
 - sglang/batch/09
 - sglang/module/schedule-batch-io
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# ScheduleBatch-IO 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 ScheduleBatch 与 ForwardBatch 的分工
- [ ] 能画出 TokenizerManager → Scheduler → DetokenizerManager 的 IPC 消息类型
- [ ] 能说出 Req、ScheduleBatch、TokenizedGenerateReqInput、BatchTokenIDOutput 四个核心类型的职责
- [ ] 能追踪一条生成请求从 GenerateReqInput 到 BatchStrOutput 的完整路径
- [ ] 能解释 PickleWrapper 存在的理由及 wrap/unwrap 时机
- [ ] 能说明 prepare_for_extend 与 prepare_for_decode 分别在何时调用、设置什么 forward_mode
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. **ScheduleBatch 是 Scheduler 的 CPU 侧批次工作台**（含 Req 列表、调度标志、GPU 张量镜像），每次 forward 前由 `ForwardBatch.init_new(batch)` 转为 ModelRunner 的 GPU 执行批次。
2. **io_struct.py 定义三进程 IPC 消息**——输入侧 `TokenizedGenerateReqInput`、输出侧 `BatchTokenIDOutput`/`BatchStrOutput`，默认 msgpack 序列化，opaque 字段通过 PickleWrapper 兜底。
3. **Req 是单请求的完整生命周期容器**——从 origin_input_ids 到 output_ids，从 prefix cache 命中到 KV 池索引，从 extend_range 到 finished_reason，贯穿调度、执行、输出全链路。

## 遗留问题

- `ScheduleBatch.merge_batch` 与 chunked prefill + overlap schedule 的交互细节，需在Scheduler（Scheduler event loop）中结合阅读
- `ForwardBatch.init_new` 的字段映射表，留给ModelRunner（ModelRunner）
- Embedding 路径的 `BatchEmbeddingOutput.pooled_hidden_states` 两种 IPC 格式（stacked vs non-stacked）的实际触发条件，需结合 tokenizer_manager 阅读

## 代码块统计

| 文件 | 代码块数 |
|------|---------|
| 09-ScheduleBatch-IO-00-MOC.md | 1 |
| 09-ScheduleBatch-IO-01-核心概念.md | 4 |
| 09-ScheduleBatch-IO-02-源码走读.md | 14 |
| 09-ScheduleBatch-IO-03-数据流与交互.md | 4 |
| 09-ScheduleBatch-IO-04-关键问题.md | 8 |
| **合计** | **31**（≥ 15 ✅） |
