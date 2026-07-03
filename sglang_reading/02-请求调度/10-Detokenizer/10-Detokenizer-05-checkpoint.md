---
type: batch-doc
module: 10-Detokenizer
batch: "10"
doc_type: checkpoint
title: "Detokenizer 验收清单"
tags:
 - sglang/batch/10
 - sglang/module/detokenizer
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# Detokenizer 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 Detokenizer 进程职责
- [ ] 能画出 Scheduler → Detokenizer → TokenizerManager 的 ZMQ 数据流
- [ ] 能说出 3 个核心类/函数：`DetokenizerManager`、`DecodeStatus`、`_decode_batch_token_id_output`（或 `FanOutCommunicator` 与控制面关系）
- [ ] 能追踪一条 streaming 请求在 Detokenizer 内的 offset 更新与 `�` 处理
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. Detokenizer 是独立进程，把 Scheduler 的 `BatchTokenIDOutput` 转为 `BatchStrOutput`，专职 CPU 侧 detokenize 与输出编码。
2. 流式增量依赖 per-rid 的 `DecodeStatus` 与 `surr/read/sent` 三组 offset，并处理 UTF-8 边界 `�` 避免重复发送。
3. `communicator.py` 的 `FanOutCommunicator` 属于 TokenizerManager **控制面**，与 Detokenizer **数据面** ZMQ 链路无关。

## 遗留问题

- ~~多 Detokenizer Worker + `MultiDetokenizerRouter` 拓扑~~ → 已补全于 [[10-Detokenizer-01-核心概念#6-multidetokenizerrouter-与多-worker-拓扑]]
- `skip_tokenizer_init` bypass 决策树见 [[10-Detokenizer-01-核心概念#7---skip-tokenizer-init-决策树]]

## 内嵌代码统计（维护者）

| 文件 | 代码块数（约） |
|------|----------------|
| 10-Detokenizer-00-MOC.md | 1 |
| 10-Detokenizer-01-核心概念.md | 4 |
| 10-Detokenizer-02-源码走读.md | 11 |
| 10-Detokenizer-03-数据流与交互.md | 6 |
| 10-Detokenizer-04-关键问题.md | 6 |
| **合计** | **≥ 28 段** |
