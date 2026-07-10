---
title: "训练数据 · 排障指南"
type: troubleshooting
framework: slime
topic: "训练数据"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 训练数据 · 排障指南

本页是 Train Data 排障入口。读完后，你应该能把 global batch、static/dynamic batch、DP rank micro-batch 数、`micro_batch_indices`、loss mask shape、VPP iterator 和日志口径问题分别落到 schedule 或 `get_batch` 源码。

## 排障总表

| 症状 | 优先看哪里 | 常见原因 |
|------|------------|----------|
| `num_rollouts < global_batch_size` | `build_dp_schedule` step split | distinct rollout 数不够形成一个 step |
| static batch 断言 | K 对齐 DP/VPP 失败 | `step_size % (dp_size * micro_batch_size * mb_group) != 0` |
| dynamic batch 仍 OOM | `_pack_step_into_mbs` | 单条样本超过 cap 或 `balance_by_flops` 不 enforce token cap |
| DP rank mbs 数不一致 | `align_to` 和 `rank_mbs_idx` | schedule 被手改或配置不满足 |
| `micro_batch_indices` 索引错 | 下标空间混淆 | 把 rank-local 下标当全局 sample 下标 |
| `full_loss_masks.shape != tokens.shape` | `get_batch` mask padding | `response_lengths/total_lengths/loss_masks` 不一致 |
| VPP stage 读错 batch | `get_data_iterator` | iterator offset 没 reset 或复用方式错误 |
| 日志与 loss 口径不同 | `rollout_mask_sums` | per-sample mean 和 per-rollout mean 混用 |

## 1. 为什么先 pack 再 distribute

Pipeline parallel 要求所有 DP rank 在同一 step 执行相同数量的 micro-batch。如果先按 rank 切样本，再让每个 rank 自己 pack，很容易得到不同 mbs 数。

源码入口：来源：slime/utils/dp_schedule.py L1-L38

验证方法：

- 看 `tests/test_dp_schedule.py` 的 `assert_invariants`，它要求每个 rank 的 mbs 数等于 `sum(num_microbatches)`。
- 如果手动改 schedule，先检查所有 rank 的 `len(micro_batch_indices[r])`。

源码入口：来源：tests/test_dp_schedule.py L51-L82

## 2. `global_batch_size` 为什么是 rollout 数

`global_batch_size` 在这里表示每个训练 step 包含多少 rollout id，不是最终训练 sample 数。compact/subagent 可能让一个 rollout 展开成多条 sample，这些 sibling 必须留在同一 step。

源码入口：来源：slime/utils/dp_schedule.py L127-L150

验证方法：

- 构造 `rollout_indices=[0,0,0,1,1,2,2,2,2]`。
- 设置 `global_batch_size=1`。
- 预期三个 step 分别覆盖 rollout 0、1、2 的 sibling。

源码入口：来源：tests/test_dp_schedule.py L252-L287

## 3. 尾部 rollout 为什么会丢弃

`num_steps = len(rollout_ids) // global_batch_size`。尾部不满一个完整 step 的 rollout 不参与训练。这是为了保持每个 step 的 rollout 语义和 loss 归一化稳定。

源码入口：来源：slime/utils/dp_schedule.py L135-L150

验证方法：

- `tests/test_dp_schedule.py` 中 5 个 rollout、`global_batch_size=2` 只产生两个 step。
- 尾部 rollout 4 对应 sample 不出现在 partitions 中。

源码入口：来源：tests/test_dp_schedule.py L290-L313

## 4. dynamic batch 超过 `max_tokens_per_gpu` 是 bug 吗

不一定。`first_fit_pack` 只保证如果单条 sample 自己不超过 cap，那么 bin sum 不超过 cap；单条超长 sample 会独占一个 mbs，仍然可能超过 cap。

源码入口：来源：slime/utils/seqlen_balancing.py L180-L198

验证方法：

- 找到超 cap mbs，如果它只有一个 sample，这是设计边界。
- 如果一个超 cap mbs 含多个 sample，再查 first-fit 或后续手工改动。

源码入口：来源：tests/test_dp_schedule.py L193-L224

## 5. `balance_by_flops` 为什么仍可能 OOM

`balance_by_flops` 用 FLOPs workload 做分区，不严格 enforce token cap。它适合 FLOPs 均衡，但不适合把 `max_tokens_per_gpu` 当硬上限的场景。

源码入口：来源：slime/utils/dp_schedule.py L65-L76

验证方法：

- 打开 `balance_by_flops` 后，打印每个 mbs 的 token sum。
- 如果 token sum 超 cap，但 FLOPs 分区逻辑正常，就不是 `first_fit_pack` 的路径。

## 6. static batch 为什么不能自动 split

static path 的语义是固定 `micro_batch_size`。如果为了对齐 DP/VPP 自动拆 micro-batch，就会破坏固定大小假设，所以源码直接抛断言并要求调整配置。

源码入口：来源：slime/utils/dp_schedule.py L167-L185

修法：

- 调整 `global_batch_size`。
- 调整 `micro_batch_size`。
- 开启 dynamic batch。
- VPP 下同时考虑 `microbatch_group_size_per_vp_stage`。

## 7. `partition` 和 `micro_batch_indices` 为什么总混

它们处在不同下标空间：

| 字段 | 空间 | 能索引什么 |
|------|------|------------|
| `partition` | 全局 sample 下标 | split 前的全局 `data[key]` |
| `micro_batch_indices[r]` | rank-local 下标 | 本 rank 的 `rollout_data[key]` |

源码入口：来源：slime/ray/rollout.py L853-L887

源码入口：来源：slime/backends/megatron_utils/data.py L219-L233

验证方法：

- 在 RolloutManager split 后看 `partition`。
- 在 actor `DataIterator.get_next` 看 `indices`。
- 如果用 `micro_batch_indices` 去索引全局 `data`，一定是错的。

## 8. VPP 为什么需要多个 DataIterator

Virtual pipeline stages 会各自调用 forward step。每个 stage 需要独立 offset，否则一个 stage 取走 mbs 后，另一个 stage 会读到下一条。

源码入口：来源：slime/backends/megatron_utils/data.py L241-L245

验证方法：

- `vpp_size > 1` 时，`get_data_iterator` 返回多个 `DataIterator`。
- 每个 iterator 的 `offset` 独立。
- 每次 forward_only 或 train 前，相关 iterator 应 reset。

## 9. allgather-CP 和默认 CP 怎么选问题入口

默认 CP 是每条 sample 先 `slice_with_cp`，再 concat；allgather-CP 是先 concat 全局 stream，再按 CP rank chunk。两者的 token 和 mask 路径不同。

源码入口：来源：slime/backends/megatron_utils/data.py L69-L104

源码入口：来源：slime/backends/megatron_utils/data.py L120-L148

验证方法：

- 如果 `cu_seqlens` 或 token chunk 形状异常，先确认 `args.allgather_cp`。
- 默认 CP 下注意 `cu_seqlens * cp_size`。
- allgather-CP 下注意 global padding 必须能被 `cp_size * pad_size` 整除。

## 10. 为什么日志也在 Train Data 里

Train Data 最了解 `total_lengths/response_lengths/loss_masks/rollout_mask_sums` 的对齐方式。`log_rollout_data` 复用这些字段做 CP-correct、per-rollout mean 的日志聚合。

源码入口：来源：slime/backends/megatron_utils/data.py L248-L330

验证方法：

- 如果 rollout log metrics 和 train loss 口径不一致，查 `rollout_mask_sums` 是否存在。
- compact sibling 场景下，不要用单 sample mask sum 替代 rollout group mask sum。

## 11. checkpoint 里为什么不再要求 `test_seqlen_balancing.py`

当前源码树没有单独的 `tests/test_seqlen_balancing.py`。调度不变量的权威测试是 `tests/test_dp_schedule.py`，其中覆盖 dynamic、oversized sample、VPP、rollout grouping 和 trailing rollout trimming。

源码入口：来源：tests/test_dp_schedule.py L1-L5
