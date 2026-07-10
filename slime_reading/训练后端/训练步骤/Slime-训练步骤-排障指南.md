---
title: "训练步骤 · 排障指南"
type: troubleshooting
framework: slime
topic: "训练步骤"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 训练步骤 · 排障指南

本页是训练 step 的排障入口。读完后，你应该能把 Ray 异步调用、critic-only warmup、log-prob 复用、PP last stage、dynamic batch loss 缩放、offload 生命周期和初始 KL CI 失败分别归到第一检查点。

## 排障总表

| 症状 | 优先看哪里 | 常见原因 |
|------|------------|----------|
| 误以为训练已经并发执行 | `RayTrainGroup.async_train` | `async_train` 只是 Ray ObjectRef fan-out |
| PPO actor advantage 异常 | `train.py`、`train_actor` | 没把 critic `value_refs` 作为 `external_data` 传入 |
| 前几个 rollout 没有 actor 更新 | `actor_trains_this_step` | `num_critic_only_steps` 生效 |
| log-prob 重算次数比预期多 | `can_reuse_log_probs_in_loss` | reuse 条件极窄 |
| 非 last PP stage 没有 values/advantages | `compute_advantages_and_returns` | PP last stage 才产生非 loss 数据 |
| 动态 batch loss 缩放异常 | `model.train`、`loss_function` | `global_batch_sizes` 与 `num_microbatches` 不匹配 |
| offload 下 train 前 OOM 或空模型 | `MegatronTrainRayActor.train` | wake/sleep 生命周期错位 |
| 初始 KL CI 失败 | `model.train` 日志检查 | ref/actor 权重、routing replay 或权重同步顺序不一致 |

## 1. `async_train` 是 Megatron 异步训练吗

不是。它只是 Ray 非阻塞 RPC 的包装：给每个 actor handler 发 `train.remote`，然后把 ObjectRef 列表还给主进程。

源码入口：来源：slime/ray/actor_group.py L131-L149

验证方法：

- 在 `async_train` 入口断点，预期不会进入 `forward_backward_func`。
- 继续到 `MegatronTrainRayActor.train`，才会看到数据恢复和 role 分派。
- 真正 generate/train 重叠的主循环在 `train_async.py`，但 actor 内部 train 路径相同。

源码入口：来源：train_async.py L30-L49

## 2. `num_critic_only_steps` 为什么让 actor 不训练

这是有意设计。前 N 个 rollout 只训练 critic，让 value baseline 先稳定；actor 需要等 `rollout_id >= num_critic_only_steps`。

源码入口：来源：train.py L72-L81

```python
# 来源：train.py L72-L81
actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

if args.use_critic:
    value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
    if actor_trains_this_step:
        ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
    else:
        ray.get(value_refs)
else:
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

验证方法：

- 在 `tests/test_qwen3_4B_ppo.py` 中可看到 `--num-critic-only-steps 1`。
- rollout 0 预期只等待 `value_refs`；rollout 1 才调用 actor train。

源码入口：来源：tests/test_qwen3_4B_ppo.py L78-L88

## 3. Actor 没拿到 critic values 会怎样

PPO 分支需要 values 计算 GAE。critic values 是通过 Ray 返回值传给 actor 的，不是存在某个共享变量里。

源码入口：来源：slime/backends/megatron_utils/actor.py L497-L503

症状：

- `advantage_estimator=ppo` 时 returns/advantages 异常。
- last PP stage 上 `rollout_data["values"]` 缺失。
- 非 last PP stage 看不到 values，但这不一定是错。

验证方法：

- 在 actor `train_actor` 的 values 注入处断点。
- 确认主循环调用是 `actor_model.async_train(..., external_data=value_refs)`。
- 确认当前 rank 是 `mpu.is_pipeline_last_stage()`，否则不应期待 values。

## 4. 为什么 log-prob 没有复用 rollout_log_probs

训练侧 log-prob 复用条件很窄。只要需要 critic、ref KL、mismatch metrics、old actor、OPD、routing replay 或 GSPO，就会重新 forward。

源码入口：来源：slime/backends/megatron_utils/actor.py L466-L493

验证方法：

- 打印 `can_reuse_log_probs_in_loss` 的每个条件。
- 如果打开 `get_mismatch_metrics`，即使用 rollout log-prob，训练侧也会额外重算以对比差异。
- 如果使用 PPO + Critic，`not self.args.use_critic` 条件不满足，所以不会复用。

## 5. `loss_type` 为什么在 critic 路径里被改成 `value_loss`

Critic 和 actor 复用同一套 `model.train`，分歧在进入 loss 前的 `args.loss_type`。critic 训练前显式改成 `value_loss`，actor 通常保留 policy loss。

源码入口：来源：slime/backends/megatron_utils/actor.py L411-L422

风险：

- 自定义 hook 如果长期修改 `args.loss_type`，可能污染后续 actor train。
- 读 loss 日志时要区分 role 前缀，不要把 critic value loss 当 policy loss。

验证方法：

- 在 `loss_function` 的 `match args.loss_type` 处断点。
- critic 路径预期命中 `value_loss`；actor 路径预期命中 `policy_loss` 或用户配置的 loss。

源码入口：来源：slime/backends/megatron_utils/loss.py L1264-L1274

## 6. 为什么非 last PP stage 没有 advantage

因为 intermediate PP stage 没有最终 logits/value 输出。`compute_advantages_and_returns` 在非 last PP stage 直接返回。

源码入口：来源：slime/backends/megatron_utils/loss.py L686-L698

验证方法：

- 在 PP last rank 观察 `kl/advantages/returns`。
- 在非 last rank 观察这些字段缺失；这不是数据丢失。
- 如果 last rank 也缺字段，再回查 log-prob/value 是否已经写入 `rollout_data`。

## 7. 动态 batch 下为什么一个 rollout 有多个 train step

RolloutManager 的 DP schedule 可以把一个 rollout batch 切成多个 step。`model.train` 用 `len(num_microbatches)` 决定 step 数，每个 step 传入自己的 `global_batch_sizes[step_id]`。

源码入口：来源：slime/backends/megatron_utils/model.py L734-L835

验证方法：

- 看日志 `train/*global_batch_size`，预期它随 step 记录实际 rollout 数。
- 若断言 `num_microbatches and global_batch_sizes must have the same length` 失败，先查 [[Slime-训练数据]] 和 RolloutManager DP schedule。

## 8. offload 模式下 train 为什么 wake 后 sleep

offload train 时，模型和 optimizer 可能处于释放显存状态。真正进入数据预处理和训练前必须 wake，训练后释放 rollout_data 并 sleep。

源码入口：来源：slime/backends/megatron_utils/actor.py L380-L400

验证方法：

- 打开 `--offload` 或相关 offload 配置时，观察 `wake_up` 是否在 `_get_rollout_data` 前发生。
- 训练后如果显存不降，检查 `del rollout_data` 和 `sleep()` 是否执行。
- 注意权重同步阶段还有独立 wake/reload 逻辑，见 [[Slime-分布式权重同步]]。

## 9. 初始 KL CI 为什么要求接近 0

CI 模式下，初始 actor 与 ref 或 old policy 应该一致；如果 rollout/train log-prob 或 actor/ref KL 明显不为 0，说明权重、路由或同步顺序可能出错。

源码入口：来源：slime/backends/megatron_utils/model.py L892-L907

```python
# 来源：slime/backends/megatron_utils/model.py L892-L907
if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
    assert log_dict["train/train_rollout_logprob_abs_diff"] <= 0.1, f"{log_dict=}"

if args.ci_test and not args.ci_disable_kl_checker:
    if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
        assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
    if (
        accumulated_step_id == 0
        and not getattr(args, "use_rollout_routing_replay", False)
        and "train/kl_loss" in log_dict
    ):
        assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"
```

验证方法：

- 先确认 `--ci-test` 是否打开。
- 若使用 routing replay，注意 R3 路径对初始 KL 有额外例外。
- 若 colocate 或 offload 同时打开，优先查权重同步时序。

## 10. sync 主循环和 async 主循环差异在哪里

Train Step actor 内部路径相同；差异在主循环如何安排 generate 与 update_weights。

| 维度 | `train.py` | `train_async.py` |
|------|------------|------------------|
| generate 与 train | 串行 | 预取下一轮 rollout |
| colocate | 支持 | 不支持 |
| train actor 调用 | `async_train` | 同样是 `async_train` |
| update_weights 时机 | 每轮 train 后 | 到间隔时先同步 pending generate 再更新 |

源码入口：来源：train_async.py L10-L11

源码入口：来源：train_async.py L30-L69

排障建议：如果问题只在 async 主循环出现，先看 generate prefetch 和 update interval；如果 `train_actor` 内部同样失败，回到本专题主线排查。

## 运行验证

Train Step 的排障要同时看主循环、Ray actor 包装、Megatron model 前向和 loss 消费字段。下面的检索覆盖这四层。

```powershell
rg -n 'def async_train|num_critic_only_steps|external_data|rollout_log_probs|loss_type|value_loss|is_pipeline_last_stage|dynamic_batch|offload|wake_up|sleep|ci_test|update_weights|rollout_data_next_future' slime/train.py slime/train_async.py slime/slime/ray/train_actor.py slime/slime/backends/megatron_utils/model.py slime/slime/backends/megatron_utils/loss.py
```

读输出时先看 `train.py` / `train_async.py` 的调用顺序，再看 `train_actor.py` 的 `train/wake_up/sleep/update_weights`。如果问题是 advantage 或 value 缺失，转到 `model.py` 的 PP last stage 和动态 batch；如果是 logprob/CI/KL 异常，继续看 `loss.py` 对 `rollout_log_probs`、`loss_type` 和 `ci_test` 的处理。
