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
updated: 2026-07-13
---
# RL 后训练数学基础

## 你为什么要读

RL infra 的核心不是“有一个 reward 就 backward”，而是把 response 级评价、安全地映射为 token 级梯度，同时说明样本由哪个策略产生、用哪个基线比较、哪些 token 可训练。只要 policy 身份或 mask 错一层，loss 仍可能有数值，却不再代表目标算法。

## 一条训练样本的七本账

| 字段/状态 | 回答的问题 |
|-----------|------------|
| prompt / response tokens | 模型真正看见并生成了什么？ |
| response length / loss mask | 哪些 token 属于 policy action、参与目标？ |
| reward | 整条 response 的外部质量信号是什么？ |
| value（PPO） | critic 对各 time step 回报的估计是什么？ |
| rollout logprob | 生成引擎当时报告的 token 概率是什么？ |
| old/current/reference logprob | surrogate 基线、带梯度策略、KL 参考分别是谁？ |
| group / rollout / version identity | 哪些样本可比较，它们何时、由什么权重产生？ |

observation、tool result、padding 或被截断的无效部分通常不应像 policy 生成 token 一样参与 loss；具体由 loss mask 和数据契约决定。

## 从 response reward 到 token advantage

### GRPO 的入门模型

同一个 prompt 产生四条 response，reward 为：

```text
[1.0, 0.6, 0.2, 0.2]
mean = 0.5
centered = [0.5, 0.1, -0.3, -0.3]
```

真实 group-relative 方法常再除以组内标准差并加 epsilon，然后把每条 response 的标量 return/advantage 广播到它的有效 response token。要保留三个边界：

- 只有同一比较组的样本才能共享 mean/std；fan-out 或错误 group id 会改变算法。
- 组内 reward 全相同时，中心化结果接近 0；epsilon 只能防除零，不能创造学习信号。
- mask 决定标量 advantage 最终作用到哪些 token。

Slime 中 GRPO、GSPO、CISPO 可在 advantage 层共享 group-relative returns，但它们的 policy objective 并不相同。详见 [[Slime-Advantage计算]]。

### PPO + Critic

PPO 用 critic value 和 reward 构造 temporal-difference residual，再通过 GAE 把末端/延迟信号分配到 time step：

```text
delta_t = r_t + gamma * V_{t+1} - V_t
A_t = delta_t + gamma * lambda * A_{t+1}
return_t = A_t + V_t
```

LLM 场景的 reward 常集中在 response 末尾，但 KL penalty、mask、截断和 value layout 会影响实际 token 序列。不能只把最终 reward 复制到每个 token 就称为 PPO GAE。

## 四种 policy/logprob 身份

| 名称 | 主要用途 | 是否带当前 backward 梯度 |
|------|----------|--------------------------|
| rollout policy/logprob | 描述样本生成分布、训练—推理偏差或 importance correction | 否 |
| old policy/logprob | PPO/GSPO surrogate 的比较基线 | 否 |
| current policy/logprob | 当前 actor logits 重新计算，形成优化目标 | 是 |
| reference policy/logprob | KL 约束的冻结参考 | 否 |

old policy 不必永远等于 rollout policy。异步生成、更新间隔、训练/推理 kernel 和采样变换都会让二者分开；具体框架还可能允许用 rollout logprob 替代某个 baseline，必须看配置与源码。

## Policy ratio 与 clipping

```text
ratio_t = exp(current_logprob_t - old_logprob_t)
surrogate_t = ratio_t * advantage_t
```

- 正 advantage 表示希望提高该 action/token 的相对概率。
- 负 advantage 表示希望降低它。
- PPO clipped objective 限制 ratio 超出区间后继续带来目标收益。

重要边界：clip 是 surrogate objective 的局部保护，不保证真实 KL、所有 token 的概率或最终策略距离一定小。KL loss、early stop、学习率和数据分布仍会影响实际更新幅度。

## 为什么 current logprob 必须重算

推理引擎返回的 rollout logprob 不连接当前训练图；backward 必须从当前 actor logits 重建 current logprob/entropy。重算还让训练端使用自己的 dtype、并行布局和 loss mask。

这也带来 mismatch：模型版本、temperature/top-p、精度、kernel、token 对齐或 routing replay 不一致，都可能使 rollout 与训练 logprob 分叉。实验应能重建样本的生成版本与采样配置；如果框架没有把 weight version 写进 Sample，就要用 rollout 时序、日志或额外字段补足证据，而不是假设它存在。

## 手算验证

给定：

```text
old_logprob = -1.2
current_logprob = -1.0
advantage = 0.5
epsilon = 0.2
```

计算：

```text
ratio = exp(0.2) ≈ 1.2214
unclipped surrogate ≈ 1.2214 × 0.5 = 0.6107
clipped ratio = 1.2
clipped surrogate = 0.6
```

对正 advantage，PPO 取 unclipped/clipped 中较保守的一项，因此是 `0.6`（loss 实现通常再取负号做最小化）。再把 advantage 改为 `-0.5`，解释 min/max 与负号为何容易写错。

对应源码见 [[Slime-Policy-Loss-核心概念]]。

## 复盘

- response reward 要经过 group/value/KL/mask 才成为 token advantage。
- rollout、old、current、reference 是四种身份，不是四个随意同义字段。
- current logprob 带梯度；其他 logprob 提供数据来源、基线或约束。
- clip 控制目标激励，不是全局“策略绝不会走远”的证明。

下一篇：[[性能指标与实验方法]]。
