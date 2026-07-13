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
updated: 2026-07-12
---
# MoE · 学习检查

## 读者能做什么

- [ ] 能画出 `hidden_states → router_logits → topk_ids/topk_weights → dispatch → expert GEMM → combine`。
- [ ] 能区分 logical selection、physical dispatch ids 与 recorder ids，知道三者并非总相同。
- [ ] 能解释 standard、bypassed、Triton-kernel、packed top-k 格式分别由谁消费。
- [ ] 能解释 logical expert id 与 physical expert id 的差异。
- [ ] 能定位 DeepEP 的 dispatch A/B、combine A/B 四个阶段。
- [ ] 能区分 EPLB placement rebalance 与 redundant-expert dispatch：前者按窗口更新，后者可逐 token 选 replica。
- [ ] 能指出量化/runner 还会影响 top-k materialization、通信 dtype 与 routed scaling ownership，而非只换 GEMM kernel。

## 场景推演

1. profiler 显示 MoE 层慢，但 router kernel 很短。你应该先拆 `FusedMoE.forward_impl` 的 dispatch、core、combine 三段。
2. top-k 输出 expert 3，但实际 dispatch 到 rank 上的 physical id 不是 3。你应该检查 `ExpertLocationDispatchInfo` 和 `topk_ids_logical_to_physical`。
3. DeepEP 报 stage assert。你应该检查 `_stage` 是否卡在 `AFTER_DISPATCH_A` 或 `AFTER_COMBINE_A`。
4. EPLB 日志显示一次 rebalance 很长。你应该先确认是否启用 layer chunks；`start/end` 可能跨多个 forward，而不是一次同步尖峰。
5. piecewise CUDA Graph 路径变化。你应该先看 `topk_output` 格式与 context，不能仅因调用 `forward_impl` 就断言 eager fallback。
6. expert 分布突然变成均匀或轮询。你应该检查 `SGLANG_SIMULATE_UNIFORM_EXPERTS` 与 `SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS` 是否覆盖真实 gate。
7. 空 token rank 卡死。若启用 LP dispatch，应检查 `empty_topk_output(layer_id=...)` 是否仍参与必要的 collective。

## 排障入口

| 症状 | 优先入口 | 预期观察 |
|------|----------|----------|
| expert 分布不均 | `select_experts`、post-process、expert recorder | logical capture、recorder ids、最终 dispatch ids 分别是什么 |
| EP all-to-all 慢 | DeepEP normal/LL dispatch 与 combine | 区分布局、A2A、expert core、回收与同步等待 |
| physical expert 不对 | `topk_ids_logical_to_physical` | logical id 被映射成 physical id |
| local expert 数不对 | `FusedMoE.__init__` | `num_experts - shared_slots` 是否能被 `moe_ep_size` 整除 |
| 量化 backend 不对 | `TopK.forward_cuda`、dispatcher quant config、`run_moe_core` | format、通信 dtype、scaling 与 method 是否配套 |
| graph 路径异常 | `FusedMoE.forward` | format、piecewise context 与注册 op，而非先假设 eager break |
| padded token 污染 | `_post_process_topk_ids` | HIP id mask 与受 `SGLANG_MORI_NO_PAD_MASK` 控制的 weight mask |
| dispatch 模式异常 | `ExpertLocationDispatchInfo`、ServerArgs | 实际值是 `static/dynamic/fake/lp`，警惕文件注解仍写 `random` |

## 验证实验

- [ ] 在模型 MoE block 断点，记录 `router_logits.shape` 和 `topk_ids.shape`。
- [ ] 在 `FusedMoE.forward_impl` 给 dispatch、core、combine 分别计时。
- [ ] 在 DeepEP 路径打印 `_stage`，确认一次 forward 后回到 `INITIAL`。
- [ ] 开启/关闭 EPLB，对比分 chunk 的 rebalance timeline、logical distribution 与 physical dispatch distribution。
- [ ] 切换 MoE runner backend，同时记录 `TopKOutput.format`、dispatcher output dtype、routed scaling 位置和 `quant_method`。
- [ ] 在 HIP 环境记录 `SGLANG_MORI_NO_PAD_MASK`，验证 padded weight 是否确实被清零。

## 能力标准

达到通过标准时，你应该能用一句话说清本专题：

> MoE 的外层生命周期是 gate/scoring → top-k materialization → logical/physical post-process → dispatch → expert core → combine；但 top-k ABI、replica 选择、通信量化和 routed scaling 的所有权会随 runner、DeepEP 与 EPLB 配置改变，所以排障必须同时追踪 ids、weights、format 和 stage state。

## 后续阅读

- [[SGLang-Attention]]：对照 attention backend 如何消费 batch metadata。
- [[SGLang-Quantization]]：继续看 MoE GEMM runner 和量化权重格式。
- [[SGLang-专用模型]]：回到 DeepSeek、Qwen、Bailing 等模型如何选择 MoE 实现。
