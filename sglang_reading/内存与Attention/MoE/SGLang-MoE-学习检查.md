---
title: "MoE · 学习检查"
type: exercise
framework: sglang
topic: "MoE"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# MoE · 学习检查

## 读者能做什么

- [ ] 能画出 `hidden_states → router_logits → topk_ids/topk_weights → dispatch → expert GEMM → combine`。
- [ ] 能说清 `topk_ids` 是路由决策，dispatcher 只是搬运协议。
- [ ] 能解释 logical expert id 与 physical expert id 的差异。
- [ ] 能定位 DeepEP 的 dispatch A/B、combine A/B 四个阶段。
- [ ] 能说明 EPLB 根据统计周期性更新 expert location，而不是每个 token 动态调度。

## 场景推演

1. profiler 显示 MoE 层慢，但 router kernel 很短。你应该先拆 `FusedMoE.forward_impl` 的 dispatch、core、combine 三段。
2. top-k 输出 expert 3，但实际 dispatch 到 rank 上的 physical id 不是 3。你应该检查 `ExpertLocationDispatchInfo` 和 `topk_ids_logical_to_physical`。
3. DeepEP 报 stage assert。你应该检查 `_stage` 是否卡在 `AFTER_DISPATCH_A` 或 `AFTER_COMBINE_A`。
4. latency 周期性尖峰。你应该对齐 EPLB rebalance 日志和 `eplb_rebalance_num_iterations`。
5. piecewise CUDA Graph fallback。你应该先看 `topk_output` 是 standard、bypassed 还是其他格式。

## 排障入口

| 症状 | 优先入口 | 预期观察 |
|------|----------|----------|
| expert 分布不均 | `select_experts`、expert recorder | `recorder_topk_ids` 是否反映 routed experts |
| EP all-to-all 慢 | `DeepEPDispatcher.dispatch_a/b`、`combine_a/b` | 慢在搬运阶段，不在 GEMM |
| physical expert 不对 | `topk_ids_logical_to_physical` | logical id 被映射成 physical id |
| local expert 数不对 | `FusedMoE.__init__` | `num_experts - shared_slots` 是否能被 `moe_ep_size` 整除 |
| 量化 backend 不对 | `run_moe_core` | `self.quant_method.apply` 的具体类型 |
| graph break | `FusedMoE.forward` | `TopKOutputChecker` 是否接受当前格式 |
| padded token 污染 | `_post_process_topk_ids` | padded 区域 id/weight 是否被 mask |

## 验证实验

- [ ] 在模型 MoE block 断点，记录 `router_logits.shape` 和 `topk_ids.shape`。
- [ ] 在 `FusedMoE.forward_impl` 给 dispatch、core、combine 分别计时。
- [ ] 在 DeepEP 路径打印 `_stage`，确认一次 forward 后回到 `INITIAL`。
- [ ] 开启/关闭 EPLB，对比 rebalance 日志和 expert distribution。
- [ ] 切换 MoE runner backend，确认只改变 `run_moe_core` 内部实现，不改变主生命周期。

## 能力标准

达到通过标准时，你应该能用一句话说清本专题：

> MoE 层先用 gate/top-k 为每个 token 生成专家路由，再由 dispatcher 把 token 搬到 physical experts，`quant_method.apply` 执行本地专家 GEMM，最后 combine 按 `topk_weights` 回到原 token 顺序；EPLB 和 DeepEP 只改变放置与搬运，不改变这条基本生命周期。

## 后续阅读

- [[SGLang-Attention]]：对照 attention backend 如何消费 batch metadata。
- [[SGLang-Quantization]]：继续看 MoE GEMM runner 和量化权重格式。
- [[SGLang-专用模型]]：回到 DeepSeek、Qwen、Bailing 等模型如何选择 MoE 实现。
