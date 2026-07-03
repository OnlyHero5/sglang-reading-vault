---
type: batch-doc
module: 23-Distributed
batch: "23"
doc_type: checkpoint
title: "分布式并行 验收清单"
tags:
 - sglang/batch/23
 - sglang/module/distributed
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# 分布式并行 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能说明 TP/PP/EP/DP/CP 各维度的用途
- [ ] 能画出 initialize_model_parallel 在启动链中的位置
- [ ] 能说出 GroupCoordinator、communication_op、DataParallelController 的职责
- [ ] 能解释为何层代码应调用 tensor_model_parallel_all_reduce 而非裸 dist
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论

1. `parallel_state.py` 集中管理所有 ProcessGroup，`GroupCoordinator` 按场景路由通信实现。
2. `communication_op.py` 是模型层的 collective 唯一推荐入口，与 CUDA Graph custom op 对齐。
3. `DataParallelController` 用 ZMQ 将请求分发到多 Scheduler，PD 场景用 FOLLOW_BOOTSTRAP_ROOM 保 locality。

## 遗留问题

- Elastic EP 与 EPLB 联动细节见 MoE MoE 与 elastic_ep 源码。
