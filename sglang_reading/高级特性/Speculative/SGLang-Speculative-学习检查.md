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

- [ ] 能画出 `ServerArgs → worker 工厂 → draft/corpus → VerifyInput → target verify → sampling → KV mover → GenerationBatchResult` 的主线。
- [ ] 能说明控制账、阶段账、KV 账、验收账分别解决什么问题。
- [ ] 能解释为什么 `NGRAMWorker.draft_worker` 是 `None`，但 NGRAM 仍然要 target verify 和 KV 写回。
- [ ] 能说清 `accept_lens` 为什么包含 bonus token，而 `move_accept_tokens_to_target_kvcache` 接收的是 drafts-only count。
- [ ] 能指出 topk 大于 1 时为什么要按 `accept_index.shape[1]` 而不是 `num_draft_tokens` 计算 KV mover 宽度。
- [ ] 能解释 `SpecInputType` 如何帮助 attention backend 区分 draft、draft extend 和 verify。
- [ ] 能描述 `eagle_sample` 为什么要在 TP rank 间 broadcast `predict`、`accept_index`、`num_correct_drafts`。
- [ ] 能定位自定义 speculative algorithm 的注册入口和 overlap 校验入口。
- [ ] 能说明 adaptive spec 何时构建 runtime state，何时根据 verify 结果切换 step。

## 口头复述

不打开 upstream 源码，尝试复述：

1. 启动时，`SpeculativeAlgorithm.from_string` 先把算法名变成内置 enum 或插件算法，再由 `handle_server_args` 修正算法专用参数，最后由 `create_worker` 返回 V2 worker class。
2. EAGLE decode step 中，draft worker 产生候选树，worker 把树拓扑放进 `EagleVerifyInput`，target worker 以 verify 模式 forward，`eagle_sample` 产出接受路径。
3. NGRAM 没有 draft model，也没有 draft KV；它从 corpus match 构造 `NgramVerifyInput`，但后续 target verify、sampling、KV 写回仍复用同一套验收账。
4. verify 临时写入的 KV 不能自动成为主链；只有 `accept_index` 指向的 accepted token 会被 mover 迁移到 target KV cache。

## 排障演练

| 场景 | 你应能先查哪里 | 预期判断 |
|------|----------------|----------|
| 开启 EAGLE 后吞吐下降 | `accept_lens`、draft steps、topk、adaptive 反馈 | 可能是接受率不足以覆盖 draft/verify 成本 |
| 多 TP rank 输出偶发不一致 | `eagle_sample` broadcast | 三个采样结果必须同源 |
| topk > 1 时输出或 KV 错位 | `_finalize_accept_tree_path` 与 KV mover | accepted path 需要 compact，宽度来自 `accept_index.shape[1]` |
| NGRAM 多请求交错后命中怪异 | `_prev_decode_rids` 与 `erase_match_state` | match state 应随离批请求清理 |
| 自定义算法启动失败 | `CustomSpecAlgo.create_worker` | overlap 能力声明和调度模式不一致 |
| plan stream 下偶发 MTP acceptance 错 | `prepare_for_draft_extend` caller | dtype cast 必须在进入 plan stream 前完成 |

## 最小验证实验

如果有可运行环境，选一个小模型分别跑普通 decode、EAGLE 或 NGRAM，并观察：

- EAGLE 路径应出现 draft worker，verify 后有 `accept_lens` 和 `next_draft_input`。
- NGRAM 路径不应加载 draft model，但 verify 分支仍调用 target worker。
- topk 大于 1 时，非 idle 请求会进入 accepted path compact。
- adaptive 打开时，verify 完成后 accepted drafts 会反馈给 controller。

没有可运行环境时，也要能用断点位置说明预期现象：`SpeculativeAlgorithm.create_worker`、`EAGLEWorkerV2.verify`、`eagle_sample`、`move_accept_tokens_to_target_kvcache`、`NGRAMWorker.forward_batch_generation`、`AdaptiveController.on_verify_complete`。

## 继续阅读

- 采样约束和 logits 后处理：[[SGLang-Sampling]]
- batch 对象与调度边界：[[SGLang-ScheduleBatch数据结构]]
- KV cache 主链与 slot：[[SGLang-KV-Cache]]
- attention backend 如何消费 `spec_info`：[[SGLang-Attention]]
