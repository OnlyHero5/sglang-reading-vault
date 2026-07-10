---
title: "分布式权重同步 · 学习检查"
type: exercise
framework: slime
topic: "分布式权重同步"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 分布式权重同步 · 学习检查

## 读者能做什么

- [ ] 能画出 `train.py → RayTrainGroup.update_weights → actor.update_weights → UpdateWeightFromDistributed → SGLangEngine` 主线。
- [ ] 能说明为什么本专题只覆盖 `full + nccl + non-colocate`。
- [ ] 能解释 PP source rank、TP gather、EP gather、HF convert、bucket、NCCL broadcast 的职责边界。
- [ ] 能区分 Ray metadata 和 NCCL tensor payload。
- [ ] 能说出 `rollout_engine_lock` 和 `weight_version` 分别防什么问题。

## 主线复述题

1. 为什么系统启动后、第一次 rollout 前也要执行一次 `actor_model.update_weights()`？
2. RolloutManager 返回的 `rollout_engine_lock` 覆盖了哪些操作？为什么不能只锁 broadcast？
3. `world_size = 1 + sum(engine_gpu_counts)` 中的 rank 0 是谁？
4. 非 PP source rank 不 broadcast，为什么还必须进入 `update_weights()`？
5. compressed-tensors 模型的 pre/post `post_process_weights` 分别发生在什么位置？

## 排障演练

| 场景 | 你应该检查 |
|------|------------|
| 第二轮 rollout 仍像旧策略 | `weight_version`、CI 抽查、engine metadata 是否到达 |
| NCCL hang | `group_name`、`engine_gpu_counts`、metadata 顺序、lock 范围 |
| 只有一个 rank 有 tqdm | `_is_pp_src_rank`，这是正常现象 |
| MoE 同步 OOM | expert pass 的 EP size 放大和 buffer size |
| offload + critic 后同步失败 | `wake_up`、`connect_rollout_engines`、`destroy_process_groups` |
| 配置了 delta 但想走 NCCL | delta 只能 disk，路径选择错误 |

## 可执行验证

```powershell
node maintenance\audit_source_evidence.mjs --note "slime_reading\权重同步\分布式权重同步\Slime-分布式权重同步-源码走读.md"
node maintenance\audit_source_evidence.mjs --note "slime_reading\权重同步\分布式权重同步\Slime-分布式权重同步-数据流.md"
node maintenance\audit_wikilinks.mjs
```

训练环境允许时，用同一模型对比：

```bash
# NCCL 路径
--update-weight-mode full --update-weight-transport nccl

# Disk 对照路径
--update-weight-mode full --update-weight-transport disk --update-weight-disk-dir <shared-dir>
```

预期关注：

- NCCL 路径不依赖共享 checkpoint 目录。
- `[slime-pp_i] Update weights` bucket 数随 `--update-weight-buffer-size` 改变。
- `--ci-test` 不触发 `Weight version mismatch`。
- MoE 模型有 expert pass 的额外同步成本。

## 通过标准

- [ ] 不看 upstream，也能解释同步闸门为什么要 pause、flush、broadcast、continue。
- [ ] 打开 upstream 后，能在 5 分钟内定位到 updater 选型、actor `update_weights`、distributed updater、SGLangEngine metadata wrapper。
- [ ] 能画出 Ray 控制面和 NCCL 数据面两条线。
- [ ] 能给出一个 NCCL hang 的断点计划：lock、metadata、broadcast、engine refs 四处各看什么。
- [ ] 能判断下一步该读 [[Slime-磁盘权重同步]]、[[Slime-Megatron到HF转换]] 还是 [[Slime-SGLang-Engine-数据流]]。

## 下一步

| 目标 | 下一篇 |
|------|--------|
| 想看 disk、delta、colocate 路径 | [[Slime-磁盘权重同步]] |
| 想看 Megatron 到 HF 命名转换 | [[Slime-Megatron到HF转换]] |
| 想看 engine 侧 HTTP 和 recv | [[Slime-SGLang-Engine-数据流]] |
| 想回到训练侧触发点 | [[Slime-训练步骤]] |
