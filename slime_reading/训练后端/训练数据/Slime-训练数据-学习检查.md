---
title: "训练数据 · 学习检查"
type: exercise
framework: slime
topic: "训练数据"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 训练数据 · 学习检查

## 读者能做什么

- [ ] 能画出 `train_data → build_dp_schedule → rank rollout_data → DataIterator → get_batch → PackedSeqParams` 主线。
- [ ] 能解释 `partition`、`micro_batch_indices`、`cu_seqlens` 分别属于哪个下标空间。
- [ ] 能说明为什么 schedule 按 rollout id 切 step，而不是按 sample 数切。
- [ ] 能区分 static batch、dynamic batch、`balance_by_flops`、`balance_data`。
- [ ] 能解释 `loss_masks` 如何从 response 空间变成 `full_loss_masks`。

## 主线复述题

1. `global_batch_size` 的单位是什么？compact rollout 展开多条 sample 时会发生什么？
2. 为什么 `build_dp_schedule` 要先 pack micro-batch，再 distribute 给 DP rank？
3. `partitions[r]` 和 `micro_batch_indices[r]` 分别能索引什么？
4. 默认 CP 和 allgather-CP 在 token concat、padding、chunk 顺序上有什么差异？
5. `full_loss_masks.shape == tokens.shape` 这个断言保护了什么？

## 排障演练

| 场景 | 你应该检查 |
|------|------------|
| static path 报 mbs 对齐断言 | `step_size`、`micro_batch_size`、`dp_size`、VPP `mb_group` |
| dynamic batch 仍 OOM | 是否单条样本超 cap；是否启用 `balance_by_flops` |
| compact sibling 被拆开 | `rollout_ids` 是否正确共享 |
| DataIterator 取错样本 | `micro_batch_indices` 是否被当全局下标使用 |
| CP 下 mask shape 错 | `response_lengths/total_lengths/loss_masks` 是否一致 |
| 日志与 loss 分母不一致 | `rollout_mask_sums` 是否进入 rank rollout_data |

## 可执行验证

```powershell
node maintenance\audit_source_evidence.mjs --note "slime_reading\训练后端\训练数据\Slime-训练数据-源码走读.md"
node maintenance\audit_source_evidence.mjs --note "slime_reading\训练后端\训练数据\Slime-训练数据-数据流.md"
node maintenance\audit_wikilinks.mjs
```

调度单测：

```bash
pytest slime/tests/test_dp_schedule.py
```

预期覆盖：

- static stride 和 balance_data。
- dynamic batch、oversized sample。
- VPP mbs group 对齐。
- compact rollout sibling 同 step。
- trailing rollout trimming。

## 通过标准

- [ ] 不看 upstream，也能说明 Train Data 的输入输出形态。
- [ ] 打开 upstream 后，能在 5 分钟内定位到 `_split_train_data_by_dp`、`build_dp_schedule`、`process_rollout_data`、`DataIterator`、`get_batch`。
- [ ] 能用一个断点计划验证 `partition` 到 rank-local `total_lengths` 的转换。
- [ ] 能用一个最小例子说明 static path 为什么不能自动 split。
- [ ] 能判断下一步该读 [[Slime-训练步骤]]、[[Slime-Advantage计算]] 还是 [[Slime-上下文并行与路由重放]]。

## 下一步

| 目标 | 下一篇 |
|------|--------|
| 想看这些 batch 字段如何进入 actor/critic train | [[Slime-训练步骤]] |
| 想看 advantage 和 return 如何消费 batch 字段 | [[Slime-Advantage计算]] |
| 想看 policy loss 如何用 `rollout_mask_sums` | [[Slime-Policy-Loss]] |
| 想看 CP 与 routing replay 的细节 | [[Slime-上下文并行与路由重放]] |
