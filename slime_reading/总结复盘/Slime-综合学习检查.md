---
title: "Slime 综合学习检查"
type: exercise
framework: slime
topic: "总结复盘"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# Slime 综合学习检查

## 你为什么要做这组检查

这组检查用来确认你能把资源编排、样本生产、训练和权重同步连成一个可验证的 RL 闭环，而不是只记住单个类和函数。

## 闭环能力

- [ ] 能解释 Training、Rollout、Data Buffer 各自持有什么状态。
- [ ] 能复述 `generate → train → update_weights` 的输入、输出和同步屏障。
- [ ] 能画出 PlacementGroup、RolloutManager、SGLangEngine、RayTrainGroup、Megatron Actor 的关系。
- [ ] 能说明 `rollout_id`、`Sample`、`rollout_data_ref` 和 `weight_version` 的生命周期。
- [ ] 能解释 colocate 为什么需要 offload，以及资源复用失败时应看哪些日志。
- [ ] 能比较 NCCL、disk、delta、tensor 权重同步路径的适用边界。
- [ ] 能说明 Slime Rollout 如何复用 [[SGLang-HTTP请求全链路]]。

## 最小验证

操作：

```powershell
rg -n "create_placement_groups|create_rollout_manager|create_training_models|generate\.remote|async_train|update_weights" slime/train.py slime/train_async.py
```

预期：能在同步或异步主循环中找到资源创建、rollout、训练和权重更新，并解释它们的先后关系。若某个环节缺失，先确认当前是 debug、critic-only、异步预取还是正常训练分支。

## 深入验证

- [ ] 运行或阅读 `tests/test_qwen3_4B_ppo.py`，写出它验证的闭环不变量。
- [ ] 从 [[实验与检查.base]] 选择一个实验，记录环境、操作、预期和实际结果。
- [ ] 选择一次权重陈旧、样本数不齐或 actor 卡住的症状，从 [[Slime-可观测性与CI]] 进入源码定位。

## 复盘入口

回看 [[Slime-RL训练全链路]]，再用 [[Slime-总结复盘]] 检查跨专题理解。与推理栈对照见 [[Slime与SGLang-阅读对照]]。
