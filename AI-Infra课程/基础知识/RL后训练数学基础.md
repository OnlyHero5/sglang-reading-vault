---
title: "RL 后训练数学基础"
type: concept
framework: cross-framework
topic: "RL 后训练"
learning_role: core
difficulty: beginner
estimated_time: "75 分钟"
prerequisites:
  - "[[LLM推理与Token]]"
tags:
  - framework/cross-framework
  - content/concept
  - source-reading
updated: 2026-07-10
---

# RL 后训练数学基础

## 学习目标

读完后，你应该能解释一条 response 如何从 reward 变成 token-level loss，并能区分 rollout policy、old policy、current policy 和 reference policy。

## 一条样本需要什么

| 字段 | 作用 |
|------|------|
| prompt / response tokens | 模型实际输入与输出 |
| loss mask | 哪些 token 参与训练 |
| reward | response 的外部质量信号 |
| rollout logprob | 生成时策略对 token 的概率 |
| current logprob | 当前训练策略重新计算的概率，带梯度 |
| reference logprob | KL 约束的参考分布 |
| advantage | 每个 token 应被增强或压低的权重 |

## 从 reward 到 advantage

GRPO 可以先理解为同一个 prompt 的多条 response 相互比较。假设四条回答 reward 是：

```text
[1.0, 0.6, 0.2, 0.2]
mean = 0.5
中心化后 = [0.5, 0.1, -0.3, -0.3]
```

真实实现还会处理标准化、mask、KL 和并行统计，但这个例子已经说明 advantage 的方向：高于组平均的回答被增强，低于组平均的回答被压低。

PPO + Critic 则使用 value 估计和 GAE，把延迟 reward 分配到 token/time step。

## Policy ratio

```text
ratio = exp(current_logprob - old_logprob)
```

- `ratio > 1` 表示当前策略提高了该 token 的概率。
- `ratio < 1` 表示当前策略降低了该 token 的概率。
- PPO clip 限制一次更新偏离 old policy 太远。

如果 advantage 为正，希望增加概率；如果 advantage 为负，希望降低概率。clip 和 KL 用于控制更新步幅。

## 为什么要重新算 logprob

Rollout logprob 可能来自推理引擎，而训练 loss 必须通过当前 actor logits 建立梯度。两者还可能因精度、kernel、temperature、模型版本不同产生差异。

因此必须记录 `weight_version` 和采样配置，避免把旧版本样本误当成当前策略样本。

## 运行验证

阅读 [[Slime-Policy-Loss-核心概念]]，手算一个 `old_logprob=-1.2`、`current_logprob=-1.0`、`advantage=0.5` 的 ratio 和未裁剪 surrogate。

预期：ratio 约为 `exp(0.2)`，大于 1；正 advantage 会推动当前 token 概率继续上升。

## 复盘

- Reward 通常是 response 级，policy loss 最终是 token 级。
- Current logprob 是带梯度路径，old/rollout logprob 是比较基线。
- KL、clip 和 importance sampling 都在控制分布偏移。
- 下一篇读 [[性能指标与实验方法]]。

