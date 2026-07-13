---
title: "训练主循环 · 学习检查"
type: exercise
framework: slime
topic: "训练主循环"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 训练主循环 · 学习检查

这份清单用来判断你是否真的理解 Slime 的训练闭环，而不是只记住 `generate -> train`。

## 读者能画出来

- [ ] 能画出 `parse_args -> create_placement_groups -> create_rollout_manager -> create_training_models -> update_weights` 的 bootstrap。
- [ ] 能画出 sync 单步：generate、offload rollout、critic/actor train、save、offload train、update weights、eval。
- [ ] 能画出 colocate 和 decoupled placement group 的 GPU 切片差异。
- [ ] 能画出 `train_async.py` 的一步 ahead future：generate N+1 与 train N 重叠。
- [ ] 能画出 `actor_model.update_weights()` 在首次 generate 前和每轮 train 后的两个位置。

## 读者能解释清楚

- [ ] 为什么 RolloutManager 必须先于 Actor/Critic 创建。
- [ ] 为什么首次 generate 前必须推一次 actor 权重。
- [ ] 为什么 sync `train.py` 里的 `async_train` 仍然是同步 step 语义。
- [ ] 为什么 `train_async.py` 禁止 colocate。
- [ ] 为什么 PPO critic-only 阶段只训练 critic，但同步 step 尾部仍会调用 actor 权重发布。
- [ ] 为什么 save 最后一轮会触发，而 eval 不一定。
- [ ] 为什么 debug 参数会在进入主循环前改写资源和 offload 配置。

## 读者能排障

- [ ] placement group 卡住时，能根据 GPU 总数、可用数和 layout 判断资源缺口。
- [ ] 首轮 rollout 权重异常时，能检查 bootstrap `onload_weights -> update_weights -> onload_kv`。
- [ ] colocate OOM 时，能检查 `offload_train/offload_rollout` 的 validate 后值和主循环顺序。
- [ ] async staleness 时，能解释一步 ahead 预取的固有滞后一拍、`update_weights_interval` 的额外影响和 drain future 的必要性。
- [ ] eval-only 没训练时，能说明它是循环外特例。
- [ ] debug train only / rollout only 行为异常时，能先回到 `arguments.py` 校验分支。

## 读者能做最小验证

```powershell
$env:PYTHONPATH='F:\源码阅读\slime'
python -m pytest slime/tests/test_qwen2.5_0.5B_short.py -q
python -m pytest slime/tests/test_qwen2.5_0.5B_async_short.py -q
python -m pytest slime/tests/test_qwen3_4B_ppo_train_critic_only.py -q
```

预期现象：

- sync short 使用 `train.py` 和 `--colocate`。
- async short 使用 `train_script="train_async.py"`，不使用 colocate。
- PPO critic-only 测试覆盖 actor/critic 分支。

这些是端到端测试，需要模型、数据、GPU、Ray 和 SGLang 环境；本地依赖不足时，测试可能 collection 或准备阶段失败，不能记为通过。

## 复盘迁移

- [ ] 接到 [[Slime-Ray参数]] 时，能说明哪些主循环分支其实由参数校验提前改写。
- [ ] 接到 [[Slime-PlacementGroup]] 时，能说明 actor/rollout/critic 的 PG 关系。
- [ ] 接到 [[Slime-RolloutManager]] 时，能说明主循环只消费 `rollout_data_ref`，不直接处理 Sample。
- [ ] 接到 [[Slime-Megatron-Actor初始化]] 时，能说明 start rollout id 如何从训练模型恢复。
- [ ] 接到 [[Slime-分布式权重同步]] 时，能说明主循环只决定 update 时机，不决定传输细节。

## 通过标准

全部满足后，应该能做到：

- 借助时序图能准确复述同步与流水异步主循环；修改实现或处理争议时会回到当前 baseline 源码。
- 根据症状判断问题在资源、rollout、训练、权重同步、周期动作还是参数校验。
- 改训练入口时知道哪些顺序不能乱动，尤其是 offload/onload 和 update weights。
