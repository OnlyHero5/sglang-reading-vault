---
type: batch-doc
module: 18-MoE
batch: "18"
doc_type: checkpoint
title: "MoE 验收清单"
tags:
 - sglang/batch/18
 - sglang/module/moe
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# MoE 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能用 prose 描述 MoE 五阶段（Router→TopK→Dispatch→GEMM→Combine）资源特征
- [ ] 能画出 dispatch → GEMM → combine 数据流
- [ ] 能说出 Router kernel、FusedMoE.forward_impl、EPLBManager 职责
- [ ] 能解释 EP 下 A2A 通信瓶颈与 DeepEP 作用
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论（3 句话）

1. **Router Triton kernel 融合 gate+topk+softmax 权重**，每 token 独立计算，无跨 rank 通信，是 compute-bound 轻量阶段。
2. **FusedMoE.forward_impl 严格 dispatch → run_moe_core → combine**，EP 下 dispatch/combine 是通信-bound 瓶颈，DeepEP 等 A2A backend 优化此路径。
3. **EPLB 周期性根据 expert_distribution 重排 logical→physical 映射**，通过 pre-dispatch hook 改写 topk_ids 平衡负载。
