---
title: "Slime 闭环实验"
type: exercise
framework: slime
topic: "RL 后训练"
learning_role: practice
difficulty: intermediate
estimated_time: "90 到 180 分钟"
prerequisites:
  - "[[RL训练闭环主线]]"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# Slime 闭环实验

## 学习目标

把 rollout、训练和权重同步拆开验证，避免在完整多机任务中同时排查所有变量。

## 静态模式

```powershell
rg -n 'debug_rollout_only|debug_train_only|save_debug_rollout_data|load_debug_rollout_data|weight_version' slime/train.py slime/slime
```

预期：能找到 debug 分支、Sample 保存/加载和权重版本透传位置。

## Rollout-only

在项目提供的可运行测试或示例参数上开启 rollout-only 和 debug dump。记录 Sample 的：

- tokens / response length
- loss mask
- reward
- rollout logprob
- rollout_id
- weight version

预期：产生可检查的 Sample，但不进入 optimizer step。

## Train-only 重放

使用上一步保存的数据开启 train-only。固定随机种子和配置，记录 loss、KL、advantage 统计、gradient norm。

预期：不调用 SGLang 生成也能重现训练侧问题；Sample 缺少必要字段时应在明确边界失败。

## DP split 检查

打印或断点检查每个 DP rank 获得的 rollout ids、micro-batch indices 和 global batch size。

预期：需要 group baseline 的 response 保持正确分组；rank-local 数据总和能还原全局 batch。

## Weight version 检查

启用权重相等检查或记录 update 前后的版本。故意把某个 engine 配置为不可更新，只在隔离环境做故障演练。

预期：系统应明确报告 engine 被跳过或版本不一致，而不是静默继续产生旧版本样本。

## 通过标准

- [ ] Rollout-only 与 train-only 可独立解释。
- [ ] 能从 Sample 追到 rank-local RolloutBatch。
- [ ] 能手工检查一组 reward 到 advantage 的方向。
- [ ] 能证明下一轮 rollout 使用了更新后的权重版本。

