---
title: "训练主循环 · 排障指南"
type: troubleshooting
framework: slime
topic: "训练主循环"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 训练主循环 · 排障指南

## 你为什么要读

这篇按症状排障。先判断问题落在资源布局、bootstrap 推权、同步 step、async 预取、周期动作，还是参数校验默认值。

## 症状一：启动卡在创建资源或角色

判断：主循环第一步是 placement group。Ray 资源不足时，`_create_placement_group` 会循环等待并打印 GPU 注册和可用数量。

```python
# 来源：slime/ray/placement_group.py L51-L67
    # Wait for the placement group to be scheduled. Poll rather than a bare
    # ray.get(pg.ready()) so the wait is observable: when it can't be placed yet
    # (a node's GPUs haven't registered with the GCS, or an autoscaler is still
    # bringing nodes up) log the GPU counts periodically instead of hanging with no
    # output. The wait stays unbounded, so autoscaling clusters — where a pending
    # placement group is what drives scale-up — are unaffected.
    ready_ref = pg.ready()
    elapsed = 0
    log_interval = 30
    while not ray.wait([ready_ref], timeout=log_interval)[0]:
        elapsed += log_interval
        total = ray.cluster_resources().get("GPU", 0)
        available = ray.available_resources().get("GPU", 0)
        logger.info(
            f"Waiting for placement group of {num_gpus} GPUs (elapsed {elapsed}s): "
            f"{total:g} GPUs registered with Ray, {available:g} available."
        )
```

处理：

- 看日志中 `Creating placement group with <n> GPUs` 和等待日志。
- 对照 `_get_placement_group_layout` 判断是 colocate、decoupled 还是 debug 模式算错了 GPU 数。
- external rollout 下不要误以为本地一定需要 rollout GPU。

## 症状二：第一次 rollout 像是用了旧权重

判断：第一次 generate 前必须执行 bootstrap `actor_model.update_weights()`。如果 rollout offload，还要先 onload weights。

```python
# 来源：train.py L22-L32
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
```

处理：

- 打开 `check_weight_update_equal` 比较初始推权。
- 检查 offload 路径是否在 update 前成功 onload weights。
- 真正传输错误继续看 [[Slime-分布式权重同步]]。

## 症状三：colocate 下显存爆或 onload/offload 顺序异常

判断：参数校验会让 colocate 默认打开 `offload_train` 和 `offload_rollout`；主循环每轮按 rollout offload、train、train offload、rollout onload weights、update、onload KV 的顺序交接。

```python
# 来源：slime/utils/arguments.py L1885-L1899
    # always true on offload for colocate at the moment.
    if args.colocate:
        if args.offload_train is None:
            args.offload_train = True
        if args.offload_rollout is None:
            args.offload_rollout = True
        if args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        elif args.rollout_num_gpus == 0:
            logger.info("rollout_num_gpus is 0 under colocate; no local SGLang engines will be launched.")

    if args.offload_train is None:
        args.offload_train = False
    if args.offload_rollout is None:
        args.offload_rollout = False
```

```python
# 来源：train.py L86-L92
        offload_train(actor_trains_this_step)
        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())
```

处理：

- 不要只看命令行是否显式传 offload；看 validate 后的 args。
- 确认 train 后先释放训练侧，再 onload rollout weights。
- KV onload 在 update 后，不要提前。

## 症状四：PPO 前几步 actor 没做 optimizer step，但版本仍变化

判断：这是 `num_critic_only_steps` 的设计。`actor_trains_this_step` 为 false 时，只训练 critic 并等待 `value_refs`；但同步 step 尾部仍调用 actor 发布门面。

```python
# 来源：train.py L72-L79
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
```

处理：

- 检查 `advantage_estimator` 是否为 PPO，因为参数校验用它设置 `use_critic`。
- 检查当前 `rollout_id` 是否小于 `num_critic_only_steps`。
- 保存逻辑也用同一判定，critic-only 阶段不会保存未训练 actor。
- 不要用 weight version 是否递增判断 actor 是否做过 optimizer step；应检查训练分支、参数/optimizer 状态或数值比较。

## 症状五：eval-only 没有进入训练循环

判断：`num_rollout == 0` 时主循环为空，eval-only 在循环外单独触发一次。

```python
# 来源：train.py L34-L36
    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))
```

处理：

- 配置 eval datasets 和 `eval_interval`。
- 不要期待 `range(args.start_rollout_id, args.num_rollout)` 进入。
- eval-only 仍会走资源创建和首次推权。

## 症状六：`train_async.py` 报 colocate 不支持

判断：这是源码断言，不是参数拼写错误。async 预取需要 rollout GPU 在 train 时继续工作，colocate 的时间复用模型不满足。

```python
# 来源：train_async.py L9-L11
# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
```

处理：

- colocate 用 `train.py`。
- decoupled 资源才用 `train_async.py`。
- fully-async rollout 是另一层优化，见 [[Slime-其他Rollout路径]]。

## 症状七：async 下 rollout policy 滞后

判断：流水预取先启动 `generate(N+1)`，再训练 N，所以即使 interval 为 1，N+1 也相对 train N 的新权重至少滞后一拍。`update_weights_interval > 1` 会进一步让多个训练步之间不发布。update 前源码会 drain 正在预取的 generate，避免同一次生成中途换权重。

```python
# 来源：train_async.py L65-L69
        if (rollout_id + 1) % args.update_weights_interval == 0:
            # sync generate before update weights to prevent update weight in the middle of generation
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            actor_model.update_weights()
```

处理：

- `update_weights_interval=1` 只能缩小额外发布间隔，不能消除流水线固有的一拍 staleness。
- 增大 interval 前确认算法能容忍 off-policy/staleness。
- 如果看到吞吐提高但 reward 变差，优先排查这里。

## 症状八：save 最后一步触发了，但 eval 没触发

判断：save 调 helper 时传入 `num_rollout`，最后一步返回 True；eval 没传 `num_rollout`，只按 interval 或 epoch 边界。

```python
# 来源：slime/utils/misc.py L119-L126
    if interval is None:
        return False

    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True

    step = rollout_id + 1
    return (step % interval == 0) or (num_rollout_per_epoch is not None and step % num_rollout_per_epoch == 0)
```

处理：

- 最后一轮保存是预期行为。
- 如果要求最后一轮 eval，需要让 interval 或 epoch 边界覆盖它，或改调用语义。

## 症状九：debug 只训练或只 rollout 的主循环看起来少了一半

判断：参数校验会改写 debug 模式。`load_debug_rollout_data` 会设置 `debug_train_only=True`；`debug_rollout_only` 会调整 actor GPU 并关闭 colocate/offload。

```python
# 来源：slime/utils/arguments.py L1844-L1849
    if args.load_debug_rollout_data is not None:
        logger.info(
            f"load_debug_rollout_data {args.load_debug_rollout_data} is set, "
            "will not instantiate sglang servers and will only run the training process."
        )
        args.debug_train_only = True
```

```python
# 来源：slime/utils/arguments.py L1866-L1879
    if args.debug_rollout_only:
        if args.colocate and args.rollout_num_gpus is None:
            args.rollout_num_gpus = args.actor_num_gpus_per_node * args.actor_num_nodes
        elif args.rollout_num_gpus == 0:
            args.actor_num_gpus_per_node = 0
            args.actor_num_nodes = 0
        else:
            args.actor_num_gpus_per_node = min(8, args.rollout_num_gpus)
            args.actor_num_nodes = args.rollout_num_gpus // args.actor_num_gpus_per_node
        args.colocate = False
        args.offload_train = args.offload_rollout = False
        if args.train_memory_margin_bytes > 0:
            logger.warning("Force train_memory_margin_bytes=0 since debug_rollout_only does not support it")
            args.train_memory_margin_bytes = 0
```

处理：先看 validate 后 args，再读主循环。很多“主循环少了某段”的现象其实是参数层已经改写。

## 运行验证

这篇的排障入口可以用一次检索覆盖：资源创建、首次权重同步、offload/onload、critic warmup、async 预取、周期动作和 debug 分支。

```powershell
rg -n 'def _create_placement_group|update_weights|offload_train|offload_rollout|num_critic_only_steps|eval_only|assert not args.colocate|rollout_data_next_future|should_run_periodic_action|debug_train_only|debug_rollout_only|load_debug_rollout_data' slime/train.py slime/train_async.py slime/slime/ray/placement_group.py slime/slime/utils/arguments.py slime/slime/utils/misc.py
```

读输出时先分清“参数校验已经改写了什么”和“主循环实际执行了什么”。`arguments.py` 解释 debug/offload/colocate 的前置改写；`train.py` 解释同步 rollout 后的训练与权重更新；`train_async.py` 解释 async 下的 policy 滞后和 colocate 禁用。
