---
title: "Policy-Loss · 排障指南"
type: troubleshooting
framework: slime
topic: "Policy-Loss"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# Policy-Loss · 排障指南

本篇按症状排障。先判断问题属于公式分支、字段来源、mask/reducer，还是 Megatron 适配。

## 症状速查

| 症状 | 最可能原因 | 源码入口 | 验证方法 |
|------|------------|----------|----------|
| `advantages` 缺失 | 上游未跑 advantage 计算 | `loss.py` L911 | 回查 [[Slime-Advantage计算]] |
| GSPO 与 GRPO advantage 一样 | 预期行为，差异在 policy loss | `loss.py` L963-L970 | 检查 `advantage_estimator == "gspo"` 是否进入 `compute_gspo_kl` |
| CISPO 梯度异常 | ratio 没 stop-gradient | `ppo_utils.py` L151-L171 | 跑 `tests/test_cispo_loss.py` |
| TIS 报缺 rollout logprob | TIS 必须有 `rollout_log_probs` | `loss.py` L987-L1005 | 检查 batch 字段 |
| `use_rollout_logprobs` 与 TIS 冲突 | 参数互斥 | `arguments.py` L1804-L1805 | 去掉其中一个 |
| TIS/ICEPOP 后 loss 分母异常 | modified mask 与原 rollout denominator 混淆 | `loss.py` L983-L1043 | 同时打印 modified numerator 与原 `rollout_mask_sums` |
| OPSM 少处理样本但不报错 | 多组 list 长度不等，`strict=False` 静默截短 | `ppo_utils.py` L54-L121 | 比较 full/current/old/advantage/mask 列表长度与逐项 shape |
| custom reducer 后 entropy/clip 指标没同步变化 | custom reducer 只接管 PG 项 | `loss.py` L1032-L1067 | 分别记录 PG reducer 与默认 reducer 输出 |
| CP 训练 hang | 空 shard 没保留反向图 | `loss.py` L1281-L1287 | 检查 allgather-CP 与 `0 * logits.sum()` |
| `kl_coef` 与 `kl_loss_coef` 冲突 | reward shaping KL 与 policy KL penalty 不能同时开 | `arguments.py` L1796 | 只保留一个 |

## PPO、GRPO、GSPO、CISPO 到底在哪里分开？

分两层看：

- [[Slime-Advantage计算]]：GRPO/GSPO/CISPO 的 advantage 分支相同。
- 本专题：GSPO 改 `ppo_kl` 的定义；CISPO 改 surrogate 公式。

源码入口：来源：slime/backends/megatron_utils/loss.py L963-L981

如果你要改 group baseline 或 reward 标准化，回到 Advantage 专题；如果你要改 ratio、clip、sequence KL，留在本专题。

## `ppo_kl` 是 reference KL 吗？

不是。`policy_loss_function` 中的 `ppo_kl` 默认是 old-current logprob 差：

```python
# 来源：slime/backends/megatron_utils/loss.py L974-L976
old_log_probs = torch.cat(old_log_probs, dim=0)
log_probs = torch.cat(log_probs, dim=0)
ppo_kl = old_log_probs - log_probs
```

reference KL penalty 是另一个可选项，在 `args.use_kl_loss` 下用 `ref_log_probs` 计算：

源码入口：来源：slime/backends/megatron_utils/loss.py L1053-L1067

配置上也禁止 `kl_coef` 和 `kl_loss_coef` 同时非零：

源码入口：来源：slime/utils/arguments.py L1796-L1796

## 为什么 `use_rollout_logprobs` 不能和 TIS 一起开？

`use_rollout_logprobs` 把 rollout logprob 直接当 old logprob；TIS 又需要 train logprob 与 rollout logprob 的差来做 off-policy correction。两者同时开会让比较分布语义混乱，所以参数校验直接禁止。

```python
# 定位骨架（基于 slime/utils/arguments.py L1804-L1815；省略相邻校验）
if args.use_rollout_logprobs:
    assert not args.use_tis, "use_rollout_logprobs and use_tis cannot be set at the same time."
...
if args.use_rollout_logprobs:
    logger.info(
        "get_mismatch_metrics is set; For metrics calculation, the log probs will still be recomputed by training engine. One more forward pass will be applied."
    )
```

如果只是想看 mismatch metrics，允许额外重算 train logprob；如果要用 TIS 修正 loss，不要把 rollout logprob直接设成 old logprob。

## vanilla TIS 与 ICEPOP 怎么接入？

Slime 没有单独的 `--icepop` 开关。默认 `--use-tis` 且没有 custom path 时用 `vanilla_tis_function`；ICEPOP 是同一个 hook 签名下的另一种 custom 实现。

CLI 入口：

```python
# 定位骨架（基于 slime/utils/arguments.py L1038-L1077；仅列本专题 CLI 项）
parser.add_argument("--use-rollout-logprobs", action="store_true", default=False, ...)
parser.add_argument("--use-tis", action="store_true", default=False, ...)
parser.add_argument("--tis-clip", type=float, default=2.0, ...)
parser.add_argument("--tis-clip-low", type=float, default=0, ...)
parser.add_argument("--custom-tis-function-path", type=str, default=None, ...)
parser.add_argument("--custom-pg-loss-reducer-function-path", type=str, default=None, ...)
```

运行时选择：

```python
# 来源：slime/backends/megatron_utils/loss.py L1011-L1015
if args.custom_tis_function_path is not None:
    tis_func = load_function(args.custom_tis_function_path)
else:
    tis_func = vanilla_tis_function
pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
```

## OPSM 为什么要 full response logprob？

OPSM 用整条 response 的平均 KL 判断是否屏蔽 negative-advantage token。CP 本地 chunk 不足以代表整条 response，因此要 all-gather current/old logprob。

```python
# 定位骨架（基于 slime/utils/ppo_utils.py L54-L92；省略循环与指标累加）
seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
opsm_mask_list.append(1 - mask)
```

OPSM 的输出再乘到 `pg_loss`：

源码入口：来源：slime/backends/megatron_utils/loss.py L953-L984

`seq_kl` 是样本级标量，`advantage < 0` 是逐 token 条件，所以 OPSM 可以只屏蔽同一条高 KL response 中 advantage 为负的 token。`opsm_clipfrac` 在 helper 中逐样本累加“被屏蔽 token 数 / 该样本有效 token 数”，不是再经过统一 batch reducer 的 token 比例。

## 为什么 custom TIS 修改了 mask，denominator 仍没变？

这是当前源码的刻意口径：custom hook 可让 rejected token 的 `pg_loss` numerator 归零，外层也会用 modified mask 重建 reducer；但传入的 `rollout_mask_sums` 仍来自原始 loss mask。因此 per-rollout denominator 保持 rollout 生成时的总有效 token 数。若插件作者预期“删 token 后重新平均”，必须显式审查或自定义 PG reducer，不能只返回 modified mask。

## allgather-CP 空 shard 为什么还要加零 loss？

某些 CP rank 可能没有有效 loss token。如果它们不经过相同的 autograd 图，CP gather 的 backward reduce-scatter 可能在其他 rank 等不到对应调用。`0 * logits.sum()` 不改变梯度数值，但强制图连通。

源码入口：来源：slime/backends/megatron_utils/loss.py L1281-L1287

同类保护还出现在 policy/value/SFT 内部的空张量路径。

“空 shard”不等于“没有 loss 对象”：这些分支的有效 numerator 可以是 0，但必须让 `logits` 或 predicted values 留在 autograd 图中。排障时既要看数值，也要确认每个 CP rank 是否执行同一组 collective backward。

## entropy 指标正常，为什么 entropy 没有梯度？

源码总会请求 entropy 供日志规约，但只有 `entropy_coef != 0` 时才保存 entropy backward 所需状态。先检查系数，再检查 `with_entropy_grad`，不要仅凭日志中出现 `entropy_loss` 就断言正则项参与了更新。top-p rollout replay 也只 mask 所选 token 的 logprob，不会把 entropy 改成同一个截断分布。

## `pg_clipfrac`、`ppo_kl`、`loss` 指标为什么和预期不一致？

先确认规约口径：

- `pg_loss` 可被 custom reducer 替换。
- `pg_clipfrac`、`ppo_kl`、entropy 默认仍用 `sum_of_sample_mean`。
- TIS/mismatch 指标用 pre-RS reducer，防止 modified mask 改变诊断分母。

源码入口：来源：slime/backends/megatron_utils/loss.py L1031-L1103

排查时分别打印 reducer 前的逐 token张量和 reducer 后的标量，别只看最终日志。

## 改 policy loss 前跑什么？

最少跑：

```powershell
Set-Location slime
python -m pytest tests/test_cispo_loss.py
python -m pytest tests/test_ppo_logprob_entropy.py
```

涉及 CP、GSPO、OPSM 时再跑：

```powershell
python -m pytest tests/test_loss_cp_invariance.py
```

文档侧检查：

```powershell
node maintenance/audit_source_evidence.mjs
node maintenance/audit_wikilinks.mjs
```
