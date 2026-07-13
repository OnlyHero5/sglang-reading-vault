---
title: "Advantage计算 · 排障指南"
type: troubleshooting
framework: slime
topic: "Advantage计算"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# Advantage计算 · 排障指南

本篇按症状排障。先定位字段缺失、并行边界或配置互斥，再决定是否需要改 estimator。

## 症状速查

| 症状 | 最可能原因 | 源码入口 | 验证方法 |
|------|------------|----------|----------|
| 非 last PP rank 看不到 `advantages` | 预期行为，只在 last stage 写 | `loss.py` L696-L698 | 在 last PP rank 打印 `rollout_data.keys()` |
| `kl_coef == 0` 仍看到 `kl` | 零 KL 用于 shape/device 对齐 | `loss.py` L700-L713 | 检查 `kl[0].abs().max()` 应为 0 |
| PPO 报 `values` 相关错误 | critic values 未注入 actor | `actor.py` L497-L503 | 确认 critic actor 返回 `external_data["values"]` |
| CP 下 whitening shape mismatch | mask 没切成本地 response chunk | `loss.py` L775-L825 | 对比 `torch.cat(advantages).shape` 与重建后 `all_masks.shape` |
| OPD 报 teacher 缺失 | `teacher_log_probs` 没进入 batch | `loss.py` L644-L646 | 检查 teacher forward 或 rollout teacher 字段 |
| `use_rollout_logprobs` 与 TIS 冲突 | 参数互斥 | `arguments.py` L1804-L1805 | 启动前去掉其中一个配置 |
| `reinforce_plus_plus` 启动失败 | 必须 normalize advantages | `arguments.py` L1798-L1802 | 加 `--normalize-advantages` |
| `rollout_top_p != 1.0` 缺字段 | top-p replay token ids/offsets 未记录 | `loss.py` L40-L51 | 检查 batch 是否含 `rollout_top_p_token_ids` |
| 跳过 old-actor forward 后 advantage 阶段报 `NoneType` | 零 KL 没有 shape 模板 | `loss.py` L700-L704 | 检查 `rollout_log_probs/log_probs/values` 是否至少一个非空 |
| baseline + OPD 后 returns 也变化 | `returns = advantages` list 别名 | `loss.py` L748-L765 | 比较 `advantages is returns`，再看 OPD 元素替换 |
| mask 检查正常但 baseline 无效位也有 advantage | baseline helper 未使用 `loss_masks` | `ppo_utils.py` L441-L472 | 确认下游 reducer/whitening 才应用 mask |
| whitening 偶发 NaN | `E[x²]-E[x]²` 舍入成负方差 | `distributed_utils.py` L132-L151 | 打印 global variance 与 finite 状态 |

## 为什么 advantage 要在 backward 前整批算完？

因为本模块需要三种 micro-batch 内部拿不到的信息：

- PPO 的 GAE 要看完整 response 的 reward/value 时间轴。
- `normalize_advantages` 要跨整个 rollout batch，并在 DP group 上做 masked whitening。
- OPD 要把 student 与 teacher 的 response logprob 对齐后统一改 advantage。

源码中 `train_actor` 在 `train()` 前调用 `compute_advantages_and_returns`，注释也说明这是因为可能需要 normalize whole rollout。

源码入口：来源：slime/backends/megatron_utils/actor.py L507-L509

## `use_rollout_logprobs` 到底省了什么？

它让 KL 与 old logprob 可以直接用 rollout engine 返回的 `rollout_log_probs`，从而避免一次训练侧 old actor logprob forward。但它不保证所有 forward 都消失：

- ref KL 仍需要 `ref_log_probs`，除非 `kl_coef == 0`。
- OPD teacher 仍需要 `teacher_log_probs`。
- mismatch metrics 可能强制训练侧重算 logprob。

配置互斥由 arguments 校验：

```python
# 定位骨架（基于 `slime/utils/arguments.py` L1804-L1815；省略相邻校验）
if args.use_rollout_logprobs:
    assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."

if args.get_mismatch_metrics:
    ...
    if args.use_rollout_logprobs:
        logger.info(
            "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
        )
```

还要拆开两种“复用”：`use_rollout_logprobs` 明确选择 rollout logprob；`can_reuse_log_probs_in_loss` 则跳过 aux forward，准备在 policy loss 当前 forward 内建立 old logprob。后者发生得太晚，advantage 已经先执行。零 KL 分支虽然会回退到 `rollout_log_probs`，但自定义 rollout 若没提供该字段且无 critic values，`xs` 为 `None`，会在创建零 tensor 时失败。复用条件本身没有检查这个前提。

## GRPO、GSPO、CISPO 为什么 advantage 一样？

因为本专题只负责信用分配，三者在这里都把序列 reward 广播到 token。GSPO/CISPO 的差异在 policy loss 阶段，例如序列级 KL 或 clip 策略。

源码入口：来源：slime/backends/megatron_utils/loss.py L720-L724

排查建议：如果你在改 GSPO/CISPO 的 ratio 或 clip，不要改 `get_grpo_returns`，应去 [[Slime-Policy-Loss]]。

## PPO 为什么只在 `cp_rank == 0` 加环境 reward？

CP 会把同一条 response 拆给多个 rank。KL 是 token 级项，每个本地 token 都有；环境 reward 是整条 response 的标量，只能落一次。当前实现把它加到 `cp_rank == 0` 的本地末 token。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/loss.py` L726-L735；省略 estimator 外层）
old_rewards = rewards
rewards = []
kl_coef = -args.kl_coef
cp_rank = mpu.get_context_parallel_rank()
for reward, k in zip(old_rewards, kl, strict=False):
    k *= kl_coef
    if cp_rank == 0:
        k[-1] += reward
    rewards.append(k)
```

如果你怀疑 reward 被重复加，检查所有 CP rank 上 `rewards` 的末 token；只有 rank 0 应该包含环境 reward。

## OPD teacher 缺失怎么查？

先判断 teacher 来源：

| 路径 | 应检查 |
|------|--------|
| Megatron teacher | `weights_backuper.backup_tags` 是否含 `"teacher"`，actor 是否跑 `store_prefix="teacher_"` |
| rollout teacher | rollout 后处理是否把 teacher logprob 写入 sample，再进入 train_data |

失败路径很直接：

```python
# 定位骨架（基于 `slime/backends/megatron_utils/loss.py` L641-L646；省略函数上下文）
if student_log_probs is None:
    return
teacher_log_probs = rollout_data.get("teacher_log_probs")
if teacher_log_probs is None:
    raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")
```

若字段存在但数值异常，再检查每条 `teacher_log_probs[i]` 的长度是否与 `log_probs[i]` 和本地 response chunk 对齐。

## `normalize_advantages` 什么时候会出错？

常见错误不是 whitening 公式，而是 mask 空间错了：

- `advantages` 是 CP 本地 response chunk。
- `loss_masks` 原始是完整 response mask。
- CP 大于 1 时必须用 `get_logits_and_tokens_offset_with_cp` 把完整 mask 切成本地 mask chunk。

源码入口：来源：slime/backends/megatron_utils/loss.py L775-L825

验证方法：在 assert 前打印：

```python
torch.cat(advantages).shape, all_masks.shape, [a.shape for a in advantages]
```

如果 `all_masks.numel() == 0`，代码会跳过 whitening；如果全局 mask sum 为 0，`distributed_masked_whiten` 会抛错。

```python
# 定位骨架（基于 `slime/utils/distributed_utils.py` L119-L154；省略 docstring 与返回前上下文）
local_sum = (values * mask).sum()
local_sum_sq = ((values**2) * mask).sum()
local_mask_sum = mask.sum()
...
dist.all_reduce(stats_tensor, group=process_group)
...
if global_mask_sum.item() == 0:
    raise ValueError("The global mask sum across all participating GPUs is zero.")
```

## 自定义 advantage hook 的契约是什么？

自定义函数在 KL 已写入之后、OPD 和 normalization 之前运行。它必须自己填：

- `rollout_data["advantages"]`
- `rollout_data["returns"]`

源码入口：来源：slime/backends/megatron_utils/loss.py L715-L718

这意味着：

- 自定义函数可以复用 `rollout_data["kl"]`。
- OPD 仍会继续修改你写入的 `advantages`。
- normalization 仍会继续处理你写入的 `advantages`。
- 如果输出不是 `list[Tensor response_chunk]`，下游 policy loss 和 whitening 会出错。
- 当前主函数和多个 helper 使用 `zip(strict=False)`；自定义 hook 应显式断言所有 sample 列等长、tensor shape 相等、没有 `None`，不要继承静默截断行为。

## REINFORCE++ baseline 为什么会同时改 returns

baseline 分支先令 `returns = advantages`，二者引用同一 list。OPD 后处理不是 tensor 原地减法，而是 `advantages[i] = adv - coef * reverse_kl`；这会替换共享 list 的元素，所以 returns 也改变。之后 whitening 创建新的 advantages list，returns 才与 advantages 分离。

操作与预期：

- baseline + OPD：预期写回的 returns 含 OPD penalty。
- baseline + OPD + normalize：预期 advantages 是 whitened 新 list，returns 是 OPD 后未 whiten 的旧共享 list。
- 其他 estimator：预期 OPD 不改变 returns。

源码入口：来源：slime/backends/megatron_utils/loss.py L741-L790

## whitening 为何 mask sum 非零仍可能 NaN

`distributed_masked_whiten` 用全局 `sum/sum_sq/count` 计算 `global_var = E[x²]-E[x]²`，再做 Bessel 修正和 `rsqrt(global_var + epsilon)`。源码只拒绝 count=0，没有 clamp 负方差，也没有 finite 断言。浮点抵消导致轻微负值时仍可能得到 NaN。

验证应记录 `global_mean/global_var/global_mask_sum`，并断言 whitened output finite；不能把“不抛 zero-mask”当作数值正确。

## top-p replay 缺字段怎么判断？

当 `args.rollout_top_p != 1.0`，训练侧需要 top-p nucleus 的 token ids 和 offsets，否则无法重放 rollout 分布。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/loss.py` L40-L51；省略函数签名）
if args.rollout_top_p == 1.0:
    return {}
top_p_token_ids = batch.get("rollout_top_p_token_ids")
top_p_token_offsets = batch.get("rollout_top_p_token_offsets")
if top_p_token_ids is None or top_p_token_offsets is None:
    raise ValueError("rollout_top_p != 1.0 requires rollout_top_p_token_ids and rollout_top_p_token_offsets.")
```

排查顺序：

1. rollout 侧是否记录 top-p token 集合。
2. [[Slime-训练数据]] 是否把字段带入 rank-local batch。
3. `forward_only(..., use_rollout_top_p_replay=True)` 是否把字段传给 callback。

## 改 estimator 前必须跑哪些检查？

至少跑：

```powershell
python -m pytest slime/tests/test_chunked_gae.py
python -m pytest slime/tests/test_loss_cp_invariance.py
```

文档侧再跑：

```powershell
node maintenance/audit_source_evidence.mjs
node maintenance/audit_wikilinks.mjs
```

预期现象：源码引用无 missing/bad range，wikilink 无 broken。若只改本专题，旧三段式关键词扫描也不应再命中。
