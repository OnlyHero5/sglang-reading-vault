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
updated: 2026-07-13
---
# 训练数据 · 学习检查

## 读者能做什么

- [ ] 能画出 `train_data → build_dp_schedule → rank rollout_data → DataIterator → get_batch → PackedSeqParams` 主线。
- [ ] 能解释 `partition`、`micro_batch_indices`、`cu_seqlens` 分别属于哪个下标空间。
- [ ] 能说明为什么 schedule 按 rollout id 切 step，而不是按 sample 数切。
- [ ] 能区分 static batch、dynamic batch、`balance_by_flops`、`balance_data`。
- [ ] 能解释 `loss_masks` 如何从 response 空间变成 `full_loss_masks`。
- [ ] 能证明尾部 trimming 后哪些 sample 被保留、哪些被丢弃。
- [ ] 能判断一个自定义字段是否会穿过 DP 分片白名单进入 actor。
- [ ] 能解释为什么全局 `raw_reward` 不能直接与 rank-local 长度列同下标遍历。

## 主线复述题

1. `global_batch_size` 的单位是什么？compact rollout 展开多条 sample 时会发生什么？
2. 为什么 `build_dp_schedule` 要先 pack micro-batch，再 distribute 给 DP rank？
3. `partitions[r]` 和 `micro_batch_indices[r]` 分别能索引什么？
4. 默认 CP 和 allgather-CP 在 token concat、padding、chunk 顺序上有什么差异？
5. `full_loss_masks.shape == tokens.shape` 这个断言保护了什么？
6. allgather-CP 下为什么不能要求 `cu_seqlens[-1] == tokens.numel()`？
7. `_convert_samples_to_train_data` 已写入 `metadata`，为什么 actor 仍看不到它？

## 排障演练

| 场景 | 你应该检查 |
|------|------------|
| static path 报 mbs 对齐断言 | `step_size`、`micro_batch_size`、`dp_size`、VPP `mb_group` |
| dynamic batch 仍 OOM | 是否单条样本超 cap；是否启用 `balance_by_flops` |
| compact sibling 被拆开 | `rollout_ids` 是否正确共享 |
| DataIterator 取错样本 | `micro_batch_indices` 是否被当全局下标使用 |
| CP 下 mask shape 错 | `response_lengths/total_lengths/loss_masks` 是否一致 |
| 日志与 loss 分母不一致 | `rollout_mask_sums` 是否进入 rank rollout_data |
| 开启 correct-only 日志后 DP rank 越界 | `raw_reward` 是否仍为全局列，而长度/mask 已变 rank-local |
| 插件字段在 actor hook 前消失 | 是否加入 `_split_train_data_by_dp` 的 per-sample 白名单 |

## 可执行验证

```powershell
rg -n 'build_dp_schedule|_split_train_data_by_dp|class DataIterator|def get_batch|PackedSeqParams' slime/slime
```

调度单测：

```powershell
Push-Location slime
python -m pytest tests/test_dp_schedule.py -q
Pop-Location
```

预期覆盖：

- static stride 和 balance_data。
- dynamic batch、oversized sample。
- VPP mbs group 对齐。
- compact rollout sibling 同 step。
- trailing rollout trimming。

静态边界实验：

1. 构造 5 个 rollout、`global_batch_size=2`，打印输入 sample 集与 partitions 并集的差集；预期差集只包含第 5 个 rollout 的全部 sibling。
2. 对照 `_convert_samples_to_train_data` 与 `_split_train_data_by_dp` 的字段列表；预期当前 `metadata` 只出现在生产端。
3. 阅读 `log_correct_samples`，列出其中全局列和 rank-local 列；预期能指出 DP>1 的错配位置，而不是只说“可能日志不准”。

## 通过标准

- [ ] 能脱离当前页面说明 Train Data 的输入输出形态；修改调度或 batch 实现时仍回到 upstream。
- [ ] 打开 upstream 后，能在 5 分钟内定位到 `_split_train_data_by_dp`、`build_dp_schedule`、`process_rollout_data`、`DataIterator`、`get_batch`。
- [ ] 能用一个断点计划验证 `partition` 到 rank-local `total_lengths` 的转换。
- [ ] 能用一个最小例子说明 static path 为什么不能自动 split。
- [ ] 能写出 schedule 输入契约：长度列与 rollout id 列等长、`global_batch_size > 0`、每 step 样本数不少于 DP size。
- [ ] 能为 allgather-CP 和默认 CP 分别写出 `tokens` 与 `cu_seqlens` 的坐标解释。
- [ ] 能判断下一步该读 [[Slime-训练步骤]]、[[Slime-Advantage计算]] 还是 [[Slime-上下文并行与路由重放]]。

## 下一步

| 目标 | 下一篇 |
|------|--------|
| 想看这些 batch 字段如何进入 actor/critic train | [[Slime-训练步骤]] |
| 想看 advantage 和 return 如何消费 batch 字段 | [[Slime-Advantage计算]] |
| 想看 policy loss 如何用 `rollout_mask_sums` | [[Slime-Policy-Loss]] |
| 想看 CP 与 routing replay 的细节 | [[Slime-上下文并行与路由重放]] |
