---
title: "Policy-Loss · 核心概念"
type: concept
framework: slime
topic: "Policy-Loss"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-13
---
# Policy-Loss · 核心概念

## 你为什么要读

本篇建立 policy loss 的心智模型。它的输入不是 raw reward，而是上游已经算好的 `advantages`；它的任务是决定当前策略相对 old policy 应该如何更新。

## 四个对象

| 对象 | 含义 | 在源码中的形态 |
|------|------|----------------|
| current logprob | 当前 actor logits 对 response token 的 logprob，带梯度 | `log_probs` |
| old logprob | rollout 或旧 actor 的 logprob，作为比较分布 | `old_log_probs` |
| advantage | 每个 token 的训练权重 | `torch.cat(batch["advantages"])` |
| reducer | 把 token loss 变成 step 标量的规约函数 | `sum_of_sample_mean` |

公式层可以简化为：

```text
ratio = exp(current_logprob - old_logprob)
pg_loss = surrogate(ratio, advantage, current_logprob)
loss = reduce(pg_loss) - entropy_coef * reduce(entropy) + optional_kl_loss
```

## PPO、GSPO、CISPO 的分叉点

| 分支 | `ppo_kl` 如何定义 | surrogate 如何定义 |
|------|------------------|--------------------|
| PPO / GRPO / REINFORCE++ | token 级 `old_log_probs - log_probs` | `compute_policy_loss` |
| GSPO | sequence-level KL，再 expand 到本地 token 形状 | `compute_policy_loss` |
| CISPO | token 级 `old_log_probs - log_probs` | `compute_cispo_loss`，ratio stop-gradient |

GSPO 的 advantage 与 GRPO 在 [[Slime-Advantage计算]] 中相同，区别在这里才出现。

```python
# 定位骨架（基于 slime/backends/megatron_utils/loss.py L963-L981；省略参数与重复拼接）
if args.advantage_estimator == "gspo":
    ppo_kl = compute_gspo_kl(...)
    old_log_probs = torch.cat(old_log_probs, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
else:
    old_log_probs = torch.cat(old_log_probs, dim=0)
    log_probs = torch.cat(log_probs, dim=0)
    ppo_kl = old_log_probs - log_probs

if args.advantage_estimator == "cispo":
    pg_loss, pg_clipfrac = compute_cispo_loss(...)
else:
    pg_loss, pg_clipfrac = compute_policy_loss(...)
```

## PPO surrogate

`compute_policy_loss` 把 `ppo_kl = old - current` 转回 `ratio = exp(current - old)`，再做 PPO clip。

```python
# 定位骨架（基于 slime/utils/ppo_utils.py L124-L148；省略 dual-clip 分支）
ratio = (-ppo_kl).exp()
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
clipfrac = torch.gt(pg_losses2, pg_losses1).float()
...
return pg_losses, clipfrac
```

这里的 `clipfrac` 仍是逐 token 诊断张量，最终标量要经过 reducer。

## CISPO surrogate

CISPO 的核心不是多一个 clip，而是 clip 后的 ratio 停止梯度，梯度只通过 `log_probs` 走。

```python
# 来源：slime/utils/ppo_utils.py L167-L171
ratio = (-ppo_kl).exp()
ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs
clipfrac = (ratio_truncated != ratio).float()
return pg_losses, clipfrac
```

这解释了为什么 `tests/test_cispo_loss.py` 不只测数值，还测 `log_ratios.grad` 是否为空或为 0。

## GSPO sequence KL

GSPO 要看整条 response 的平均 KL，但工程接口仍然是 token loss。因此源码先算每条样本的 sequence KL，再 expand 到 local token 形状。

```python
# 来源：slime/utils/ppo_utils.py L114-L119
ppo_kl = [
    ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
    for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
]
ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
ppo_kl = torch.cat(ppo_kl, dim=0)
```

CP 下如果只看本地 chunk，就无法得到 sequence-level KL，所以 `policy_loss_function` 会在 GSPO 或 OPSM 时 all-gather full logprob。

这里有一个必须由调用边界承担的契约：`compute_gspo_kl` 和 gather 循环都使用 `zip(..., strict=False)`。它们不会主动证明 full/current/old/mask/local 五组列表等长；缺项会被静默截短，shape 不兼容则到乘法或拼接时才暴露。

## TIS、ICEPOP、OPSM 是后修正

| 功能 | 位置 | 改什么 |
|------|------|--------|
| OPSM | surrogate 后 | 用 sequence KL 与 negative advantage 生成 mask，再乘到 `pg_loss` |
| TIS | surrogate 后 | 用 train/rollout logprob 差构造 importance weight，乘到 `pg_loss` |
| ICEPOP | TIS hook 的一种实现 | 越界 token 权重置零，而不是 clamp |
| custom PG reducer | TIS 后 | 只替换 `pg_loss` 的规约口径 |

默认 TIS 与 ICEPOP 都走同一个 hook 签名：

```python
# 定位骨架（基于 slime/backends/megatron_utils/loss.py L831-L878；拼接 vanilla TIS 与 ICEPOP 的核心差异）
tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
pg_loss = pg_loss * tis_weights
...
ice_weight = torch.where(
    (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
)
pg_loss = pg_loss * ice_weight
```

TIS 还有两个容易被“modified mask”这个名字带偏的细节：

- 默认 vanilla TIS 和内置 ICEPOP 都原样返回 `loss_masks`；只有 custom hook 才可能真的返回修改后的 mask。
- 即便 custom hook 修改了 mask，重建 reducer 时仍传入基于原始 mask 预计算的 `rollout_mask_sums`。因此 rejected token 的 numerator 可以归零，但 per-rollout denominator 不会随 modified mask 重算。

## Reducer 是语义边界

同一个 token loss 可以按 per-token 或 per-rollout mean 规约。`loss_function` 用 `get_sum_of_sample_mean` 构造 reducer，再注入具体 loss 函数。

```python
# 定位骨架（基于 slime/backends/megatron_utils/loss.py L1254-L1279；省略 dispatch 分支参数）
num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
sum_of_sample_mean = get_sum_of_sample_mean(...)
...
if args.recompute_loss_function:
    loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
else:
    loss, log = func(args, batch, logits, sum_of_sample_mean)
```

读源码时要把“生成逐 token loss”和“把 token loss 规约成训练标量”分开，否则很容易误判 TIS、OPSM、metrics 的统计口径。

custom PG reducer 也不是全局 reducer：它只接管 `pg_loss`，调用签名只收到 `total_lengths`、`response_lengths`、所选 mask 和 `calculate_per_token_loss`，不会自动收到 `rollout_mask_sums`。`pg_clipfrac`、`ppo_kl`、entropy 与 reference KL 仍走当前 `sum_of_sample_mean`。

## Entropy 与 reference KL 的梯度边界

`policy_loss_function` 总是请求 entropy 结果用于日志，但 `get_log_probs_and_entropy` 只有在 `entropy_coef != 0` 时才保存 entropy backward 所需状态。因此“计算 entropy 指标”和“entropy 参与梯度”是两回事。rollout top-p replay 只约束所选 token 的 logprob 路径，不会把 entropy 也裁成同一 top-p 分布。

reference KL 则是独立的可选 penalty：`ppo_kl` 比较 old 与 current，`use_kl_loss` 比较 current 与 reference；`use_unbiased_kl` 还会传入 `exp(current-old)` importance ratio。`low_var_kl` 最后被 clamp 到 `[-10, 10]`，这是数值稳定边界，不是 PPO ratio clip。
