---
title: "Advantage计算 · 学习检查"
type: exercise
framework: slime
topic: "Advantage计算"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Advantage计算 · 学习检查

## 读者能做什么

- [ ] 能画出 `rewards/log_probs/ref_log_probs/values/loss_masks -> kl -> estimator -> OPD -> whitening -> advantages/returns` 的主线。
- [ ] 能说明为什么 `advantages` 只在 pipeline last stage 写入。
- [ ] 能解释 `kl_coef == 0` 时零 KL 张量的 shape/device 作用。
- [ ] 能区分 GRPO/GSPO/CISPO 在 advantage 阶段相同、policy loss 阶段不同。
- [ ] 能复述 PPO 分支如何把 token KL 与环境 reward 合成 GAE 输入。
- [ ] 能说明 OPD 为什么是 estimator 后处理，而不是新 estimator。
- [ ] 能定位 CP 下 advantage/mask shape mismatch 的源码入口。

## 源码入口自测

- [ ] actor 调用链：`slime/backends/megatron_utils/actor.py` L430-L509。
- [ ] forward-only 聚合：`slime/backends/megatron_utils/model.py` L344-L506。
- [ ] logprob/value 提取：`slime/backends/megatron_utils/loss.py` L470-L617。
- [ ] advantage 主函数：`slime/backends/megatron_utils/loss.py` L661-L828。
- [ ] estimator helper：`slime/utils/ppo_utils.py` L361-L639。
- [ ] DP whitening：`slime/utils/distributed_utils.py` L94-L154。

## 可执行验证

- [ ] 运行 `python -m pytest slime/tests/test_chunked_gae.py`，确认 chunked GAE 与 vanilla GAE 等价。
- [ ] 运行 `python -m pytest slime/tests/test_loss_cp_invariance.py`，确认 CP 切分不改变相关 loss 行为。
- [ ] 运行 `node maintenance/audit_source_evidence.mjs --note slime_reading/训练后端/Advantage计算/Slime-Advantage计算-源码走读.md`，确认本篇引用的 upstream 文件存在。
- [ ] 运行 `node maintenance/audit_wikilinks.mjs`，确认本专题双链无断链。

## 排障演练

- [ ] 构造 `--advantage-estimator ppo` 的阅读路径，能说出 critic values 从哪里来。
- [ ] 构造 `--use-opd` 的阅读路径，能说出 `teacher_log_probs` 缺失时查哪两条来源。
- [ ] 构造 `--normalize-advantages` 且 CP 大于 1 的路径，能说出完整 mask 如何切成本地 mask。
- [ ] 构造 `--use-rollout-logprobs` 的路径，能说出哪些 forward 仍可能发生。

## 迁移结论

这组文档读懂后，下一步读 [[Slime-Policy-Loss]]。本专题回答“每个 token 的训练权重从哪来”，下一专题回答“这些权重如何进入 policy gradient、clip、entropy、GSPO/CISPO 和 metrics”。
