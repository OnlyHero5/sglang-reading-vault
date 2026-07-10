---
title: "训练步骤 · 学习检查"
type: exercise
framework: slime
topic: "训练步骤"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 训练步骤 · 学习检查

## 读者能做什么

- [ ] 能画出 `rollout_data_ref → critic values → actor advantages → model.train → optimizer.step → update_weights` 主线。
- [ ] 能区分 Ray `async_train`、Actor `train_actor/train_critic`、Megatron `train_one_step` 三层职责。
- [ ] 能说明 PPO + Critic 为什么必须把 `value_refs` 作为 `external_data` 传给 actor。
- [ ] 能解释为什么非 PP last stage 没有 `values/log_probs/advantages`。
- [ ] 能指出 `num_microbatches`、`global_batch_sizes`、`rollout_mask_sums` 分别影响哪一层。

## 主线复述题

1. 一个 `rollout_data_ref` 从 `train.py` 到 `MegatronTrainRayActor.train` 经过哪些 Ray ObjectRef？
2. Critic 路径为什么先 `forward_only(get_values)`，再把 `loss_type` 改成 `value_loss`？
3. Actor 路径在哪一步拿到 critic values？为什么只有 last PP stage 做这件事？
4. `forward_only` 与 `train_one_step` 都会调用 Megatron pipeline，它们的 `forward_only` 参数和副作用有什么不同？
5. `loss_function` 为什么要同时看到 `num_microbatches` 和 `step_global_batch_size`？

## 排障演练

| 场景 | 你应该检查 |
|------|------------|
| PPO advantage 全是异常值 | `external_data` 是否传入，last PP stage 是否有 `values` |
| rollout 0 没有 actor loss | `num_critic_only_steps` 是否大于 0 |
| 训练侧 log-prob 比 rollout log-prob 多算一次 | `can_reuse_log_probs_in_loss` 哪个条件失败 |
| 动态 batch 日志里 step batch size 不一致 | `global_batch_sizes[step_id]` 是否来自 DP schedule |
| 非 last PP rank 缺 `advantages` | 这是正常路径，去 last PP rank 验证 |
| 初始 KL CI 失败 | 权重加载、routing replay、权重同步顺序 |

## 可执行验证

```powershell
node maintenance\audit_source_evidence.mjs --note "slime_reading\训练后端\训练步骤\Slime-训练步骤-源码走读.md"
node maintenance\audit_source_evidence.mjs --note "slime_reading\训练后端\训练步骤\Slime-训练步骤-数据流.md"
node maintenance\audit_wikilinks.mjs
```

训练环境允许时，可阅读或运行：

```bash
pytest slime/tests/test_qwen3_4B_ppo.py -k execute
```

预期关注：

- `--num-critic-only-steps 1` 让 rollout 0 只训 critic。
- `--advantage-estimator ppo` 和 critic 配置触发 values 传递。
- `--use-dynamic-batch-size` 触发 per-step `global_batch_size` 日志。
- `--ci-test` 触发初始 KL 与 rollout/train log-prob 检查。

## 通过标准

- [ ] 不看 upstream，也能讲清本专题六个核心对象：`rollout_data_ref`、`rollout_data`、`DataIterator`、`external_data`、`advantages/returns`、`loss batch`。
- [ ] 打开 upstream 后，能在 5 分钟内定位到 `async_train`、`train_actor`、`train_critic`、`train_one_step`、`loss_function`。
- [ ] 能用一个断点计划验证 PPO + Critic 的 values 传递。
- [ ] 能解释一个配置变化后主线如何分叉：无 critic、critic-only warmup、dynamic batch、offload、routing replay 任选四个。
- [ ] 能说出下一篇应读 [[Slime-训练数据]]、[[Slime-Advantage计算]] 还是 [[Slime-分布式权重同步]]。

## 下一步

| 目标 | 下一篇 |
|------|--------|
| 想看 `rollout_data` 如何被切成 DP micro-batch | [[Slime-训练数据]] |
| 想看 PPO/GRPO/GSPO advantage 算法细节 | [[Slime-Advantage计算]] |
| 想看 policy loss、ratio、clip、KL 的公式 | [[Slime-Policy-Loss]] |
| 想看训练后的权重如何进入 SGLang | [[Slime-分布式权重同步]] |
