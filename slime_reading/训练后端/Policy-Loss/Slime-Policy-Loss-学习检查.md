---
title: "Policy-Loss · 学习检查"
type: exercise
framework: slime
topic: "Policy-Loss"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Policy-Loss · 学习检查

## 读者能做什么

- [ ] 能画出 `logits -> current log_probs -> ppo_kl -> surrogate -> TIS/OPSM -> reducer -> Megatron scaled loss`。
- [ ] 能说明 policy loss 为什么只消费 `advantages`，不重新计算 reward 或 returns。
- [ ] 能区分 `ppo_kl` 与 reference KL penalty。
- [ ] 能解释 GSPO 为什么需要 full response logprob，再 expand 到 token 形状。
- [ ] 能说明 CISPO 的 stop-gradient ratio 对梯度路径的影响。
- [ ] 能解释 TIS 与 ICEPOP 都是同一个 hook 点的不同实现。
- [ ] 能说明 `loss_function` 三元组如何对接 Megatron。

## 源码入口自测

- [ ] training forward callback：`slime/backends/megatron_utils/model.py` L560-L638。
- [ ] policy loss 主线：`slime/backends/megatron_utils/loss.py` L881-L1110。
- [ ] value/SFT/dispatch：`slime/backends/megatron_utils/loss.py` L1113-L1320。
- [ ] PPO/CISPO/GSPO/OPSM helper：`slime/utils/ppo_utils.py` L54-L171。
- [ ] TIS/OPSM 参数：`slime/utils/arguments.py` L1038-L1103。
- [ ] 配置互斥：`slime/utils/arguments.py` L1796-L1835。

## 可执行验证

- [ ] 运行 `python -m pytest slime/tests/test_cispo_loss.py`，确认 CISPO 数值和梯度路径。
- [ ] 运行 `python -m pytest slime/tests/test_ppo_logprob_entropy.py`，确认 logprob/entropy 基础路径。
- [ ] 运行 `python -m pytest slime/tests/test_loss_cp_invariance.py`，确认 CP 相关路径不变性。
- [ ] 运行 `node maintenance/audit_source_evidence.mjs --note slime_reading/训练后端/Policy-Loss/Slime-Policy-Loss-源码走读.md`，确认源码引用可追踪。
- [ ] 运行 `node maintenance/audit_wikilinks.mjs`，确认双链无断链。

## 排障演练

- [ ] `--advantage-estimator gspo`：能指出 full logprob gather 和 `compute_gspo_kl`。
- [ ] `--advantage-estimator cispo`：能指出 `ratio_truncated.detach()` 和对应测试。
- [ ] `--use-tis`：能指出必须有 `rollout_log_probs`，且不能同时 `--use-rollout-logprobs`。
- [ ] allgather-CP 空 token：能指出 `0 * logits.sum()` 的作用。

## 迁移结论

这组文档读懂后，再读 [[Slime-上下文并行与路由重放]]。本专题把 policy loss 的算法和 Megatron 适配讲清楚；下一个专题专门处理 CP/routing replay 下的执行一致性。
