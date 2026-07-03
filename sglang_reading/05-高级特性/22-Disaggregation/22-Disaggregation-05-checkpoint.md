---
type: batch-doc
module: 22-Disaggregation
batch: "22"
doc_type: checkpoint
title: "PD 分离 验收清单"
tags:
 - sglang/batch/22
 - sglang/module/disaggregation
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# PD 分离 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能说明 Prefill 三队列与 Decode 四队列各自职责
- [ ] 能画出 KV 从 Prefill forward 到 Decode PrebuiltExtend 的时序
- [ ] 能说出 `DisaggregationMode`、`poll_and_all_reduce`、`_apply_metadata_gate` 的作用
- [ ] 能解释 DecodeReqToTokenPool 为何能 unblock Prefill bootstrap
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论

1. PD 分离通过 `--disaggregation-mode` 将 Prefill 算力与 Decode 算力解耦，KV 经 Transfer Backend 跨节点搬运。
2. Prefill 侧 Bootstrap→Waiting→Inflight，Decode 侧 Prealloc→Transfer→Waiting→Running，metadata gate 保证 TP 一致性与 metadata 就绪。
3. `DecodeReqToTokenPool` 利用空闲 KV 预分配握手，是 PD 流水线不被 running 槽位阻塞的关键。

## 遗留问题

- Mooncake/NIXL 底层连接细节见各 backend conn.py，可按部署环境专项阅读。
