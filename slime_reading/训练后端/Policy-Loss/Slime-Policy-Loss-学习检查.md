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
updated: 2026-07-13
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
- [ ] 能说明 old logprob 缺失时 ratio 为何从 1 起步，以及这不等于 Advantage 阶段已有 shape template。
- [ ] 能解释 custom TIS 的 modified mask 为什么不会自动重算 `rollout_mask_sums`。
- [ ] 能说明 custom PG reducer 不会接管 entropy、clipfrac、`ppo_kl` 与 reference KL。
- [ ] 能区分 entropy“被计算用于日志”和“保存 backward 状态参与梯度”。

## 源码入口自测

- [ ] training forward callback：`slime/backends/megatron_utils/model.py` L560-L638。
- [ ] policy loss 主线：`slime/backends/megatron_utils/loss.py` L881-L1110。
- [ ] value/SFT/dispatch：`slime/backends/megatron_utils/loss.py` L1113-L1320。
- [ ] PPO/CISPO/GSPO/OPSM helper：`slime/utils/ppo_utils.py` L54-L171。
- [ ] TIS/OPSM 参数：`slime/utils/arguments.py` L1038-L1103。
- [ ] 配置互斥：`slime/utils/arguments.py` L1796-L1835。

## 可执行验证

- [ ] 从知识库根目录执行 `Set-Location slime`，再运行 `python -m pytest tests/test_cispo_loss.py`，确认 CISPO 数值和梯度路径。
- [ ] 运行 `python -m pytest tests/test_ppo_logprob_entropy.py`，确认 logprob/entropy 基础路径。
- [ ] 运行 `python -m pytest tests/test_loss_cp_invariance.py`，确认 CP 相关路径不变性。
- [ ] 用 `rg -n 'compute_policy_loss|compute_cispo_loss|compute_gspo|use_tis|icepop' slime/slime/backends/megatron_utils slime/slime/utils` 定位算法分支与 hook。

Slime 测试必须从仓库目录 `slime/` 执行；预期同时检查数值和梯度 case 被收集。若只运行成功但显示 `0 tests collected`，不算通过。Windows 若在 collection 阶段报 `torch.compile` 不受支持，应把它记录为环境限制，不能伪装成算法测试失败或通过。

## 排障演练

- [ ] `--advantage-estimator gspo`：能指出 full logprob gather 和 `compute_gspo_kl`。
- [ ] `--advantage-estimator cispo`：能指出 `ratio_truncated.detach()` 和对应测试。
- [ ] `--use-tis`：能指出必须有 `rollout_log_probs`，且不能同时 `--use-rollout-logprobs`。
- [ ] allgather-CP 空 token：能指出 `0 * logits.sum()` 的作用。
- [ ] 人为让 full/current/old/mask 列表长度不等：能解释 `zip(strict=False)` 为什么可能静默少处理样本。
- [ ] custom TIS 返回更小 mask：能分别核对 numerator、原 `rollout_mask_sums` denominator 和 pre-RS mismatch metrics。

## 迁移结论

这组文档读懂后，再读 [[Slime-上下文并行与路由重放]]。本专题把 policy loss 的算法和 Megatron 适配讲清楚；下一个专题专门处理 CP/routing replay 下的执行一致性。
