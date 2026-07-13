---
title: "Speculative · 学习检查"
type: exercise
framework: sglang
topic: "Speculative"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Speculative · 学习检查

这个 checkpoint 检查你能不能把投机解码当成一条可运行的 decode 流水线来理解，而不是检查是否看过多少段代码。

## 读者能做什么

- [ ] 能画出 `ServerArgs → worker 工厂 → 候选 → target verify → 算法专用验收/提交 → GenerationBatchResult` 的主线。
- [ ] 能说明控制账、阶段账、KV 账、验收账分别解决什么问题。
- [ ] 能解释为什么 `NGRAMWorker.draft_worker` 是 `None`，但 NGRAM 仍然要 target verify 和 KV 写回。
- [ ] 能说清 EAGLE/NGRAM 的 `accept_lens` 为什么包含 bonus token，而通用 KV mover 接收 drafts-only count。
- [ ] 能指出 topk 大于 1 时为什么要按 `accept_index.shape[1]` 而不是 `num_draft_tokens` 计算 KV mover 宽度。
- [ ] 能解释 `SpecInputType` 如何帮助 attention backend 区分 draft、draft extend 和 verify。
- [ ] 能描述 `eagle_sample` 只有 stochastic 分支为什么、何时在 TP rank 间 broadcast 三个结果，并指出 greedy/HIP/NPU 例外。
- [ ] 能定位自定义 speculative algorithm 的注册入口和 overlap 校验入口。
- [ ] 能说明 EAGLE adaptive 何时构建 runtime state、CPU result processor 何时反馈 verify 结果，以及 NGRAM 当前为何不在此能力路径。
- [ ] 能解释 DFLASH 为什么不经过 `eagle_sample`、tree custom mask 和通用 KV mover。
- [ ] 能列出 speculative verify 与普通 Sampling 的差异：relaxed penalty、无 min-p、无 custom logit processor、不同 seed/backend 语义。

## 口头复述

不打开 upstream 源码，尝试复述：

1. 启动时，`SpeculativeAlgorithm.from_string` 先把算法名变成内置 enum 或插件算法，再由 `handle_server_args` 修正算法专用参数，最后由 `create_worker` 返回 V2 worker class。
2. EAGLE decode step 中，draft worker 产生候选树，worker 把树拓扑放进 `EagleVerifyInput`，target worker 以 verify 模式 forward，`eagle_sample` 产出接受路径。
3. NGRAM 没有 draft model，也没有 draft KV；它从 corpus match 构造 `NgramVerifyInput`，后续复用 target verify、`eagle_sample` 和通用 KV mover。
4. EAGLE `topk > 1` 的 verify 临时树需要按 `accept_index` 搬移并 compact；`topk == 1` 的 chain 已在前部，因此跳过 mover。
5. DFLASH 用固定 block 与专用 accept/bonus 逻辑形成推进长度，最终结果契约相同，但内部没有 EAGLE/NGRAM 的树验收对象。

## 排障演练

| 场景 | 你应能先查哪里 | 预期判断 |
|------|----------------|----------|
| 开启 EAGLE 后吞吐下降 | `accept_lens`、draft steps、topk、adaptive 反馈 | 可能是接受率不足以覆盖 draft/verify 成本 |
| 多 TP rank 输出偶发不一致 | `eagle_sample` 实际分支 | stochastic 分支三个结果必须同源；greedy/HIP/NPU 不走 broadcast |
| topk > 1 时输出或 KV 错位 | `_finalize_accept_tree_path` 与 KV mover | accepted path 需要 compact，宽度来自 `accept_index.shape[1]` |
| NGRAM 多请求交错后命中怪异 | `_prev_decode_rids` 与 `erase_match_state` | match state 应随离批请求清理 |
| 自定义算法启动失败 | `CustomSpecAlgo.create_worker` | overlap 能力声明和调度模式不一致 |
| plan stream 下偶发 MTP acceptance 错 | `prepare_for_draft_extend` caller | dtype cast 必须在进入 plan stream 前完成 |
| DFLASH 非 greedy 配置却像 greedy | sampling verify kernel 可用性与 warning | kernel 不可用时会回退 argmax；不是 EAGLE rejection sampling |

## 最小验证实验

如果有匹配 checkpoint 和可运行环境，至少对比普通 decode 与一种投机算法；具备对应模型时再覆盖 EAGLE、NGRAM、DFLASH，并观察：

- EAGLE 路径应出现 draft worker，verify 后有 `accept_lens` 和 `next_draft_input`。
- NGRAM 路径不应加载 draft model，但 verify 分支仍调用 target worker。
- topk 大于 1 时，非 idle 请求会进入 accepted path compact。
- EAGLE adaptive 打开时，accepted drafts 应在 CPU result processing 后反馈给 controller。
- DFLASH 路径不应命中 `eagle_sample` 或通用 KV mover；`return_logprob=True` 当前应被明确拒绝。

没有可运行环境时，也要能按算法选择断点并说明预期现象：共同入口 `SpeculativeAlgorithm.create_worker`；EAGLE/NGRAM 的 `eagle_sample`；EAGLE 树的 `_finalize_accept_tree_path`；NGRAM 的 KV mover；DFLASH 的 `compute_dflash_*_correct_drafts_and_bonus`；EAGLE adaptive 的 `_resolve_spec_v2_tokens` 与 `AdaptiveController.on_verify_complete`。

## 继续阅读

- 采样约束和 logits 后处理：[[SGLang-Sampling]]
- batch 对象与调度边界：[[SGLang-ScheduleBatch数据结构]]
- KV cache 主链与 slot：[[SGLang-KV-Cache]]
- attention backend 如何消费 `spec_info`：[[SGLang-Attention]]
