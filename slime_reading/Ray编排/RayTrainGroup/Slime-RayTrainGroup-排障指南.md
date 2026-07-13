---
title: "RayTrainGroup · 排障指南"
type: troubleshooting
framework: slime
topic: "RayTrainGroup"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# RayTrainGroup · 排障指南

这篇按症状排障。先判断问题发生在 actor 创建、master addr/port、LOCAL_RANK、distributed init、async refs、还是同步生命周期操作。

## 症状总表

| 症状 | 先查什么 | 源码入口 | 验证方法 |
|------|----------|----------|----------|
| rank 1..N 连不上 master | rank 0 是否成功返回 master addr/port | `slime/ray/actor_group.py` L105-L119 | 看 rank 0 actor 创建和端口日志 |
| `LOCAL_RANK` 不符合预期 | Ray 是否设置了 `CUDA_VISIBLE_DEVICES` | `slime/ray/train_actor.py` L20-L49 | 在 actor 内打印 CVD、Ray GPU ids、LOCAL_RANK |
| distributed init 卡住 | MASTER/RANK/WORLD env 是否一致 | `slime/ray/train_actor.py` L50-L70 | 看 `dist.init_process_group` 前后的日志 |
| `async_train` 返回值用错 | 是否忘记 `ray.get` 或重复组织 refs | `slime/ray/actor_group.py` L131-L149 | 对照 `train.py` L72-L82 |
| update 后 rollout 用旧权重 | 是否在 generate 前完成 `update_weights` | `slime/ray/actor_group.py` L155-L157 | 对照 `train.py` L83-L92 |
| critic values 没进 actor | `external_data` 是 refs list 还是广播 dict | `slime/ray/actor_group.py` L140-L149 | 检查 list 长度和 actor 数 |
| routing replay 在 critic 里无效 | env 只对 actor role 注入 | `slime/ray/actor_group.py` L86-L88 | 检查 actor/critic runtime env |
| offload_train 启动前失败 | torch_memory_saver 动态库是否存在 | `slime/ray/actor_group.py` L64-L84 | 看 `FileNotFoundError` |

## Q1：为什么 `async_init` 不 `ray.get`，但 `update_weights` 要同步？

因为 init/train API 把等待权留给 driver，而 update_weights 是一致性闸门，下一轮 generate 前必须完成。但要区分能力和现状：当前 `create_training_models` 先等待 critic init，再启动并等待 actor init，并未并行初始化两组。

异步 API：

```python
# 定位骨架（据 `slime/ray/actor_group.py` L121-L149 删节）：
def async_init(self, args, role, with_ref=False, with_opd_teacher=False):
    """
    Allocate GPU resourced and initialize model, optimzier, local ckpt, etc.
    """
    self.args = args
    return [
        actor.init.remote(args, role, with_ref=with_ref, with_opd_teacher=with_opd_teacher)
        for actor in self._actor_handlers
    ]

def async_train(self, rollout_id, rollout_data_ref, external_data=None):
    """Do one rollout training. Returns a list of Ray refs (one per worker).
```

同步 API：

```python
# 来源：slime/ray/actor_group.py L151-L157
def save_model(self, rollout_id, force_sync=False):
    """Save actor model"""
    return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

def update_weights(self):
    """Broadcast weights from rank 0 to all other ranks."""
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])
```

主循环也按这个边界使用：

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

排障动作：看到 `async_train` 后，确认 driver 最终对返回 refs 做了 `ray.get`；看到 `update_weights`，确认不要再把它当 refs list 使用。

## Q2：rank 0 master 端口为什么以 20000 到 21000 的随机值起搜？

rank 0 actor 自己在所在节点上找可用端口，避免依赖固定端口。这个端口是 torch distributed rendezvous，不是 Ray head 端口。随机值只是 `get_free_port()` 的起点；若被占用，函数会持续递增，因此最终端口可能高于 21000。

```python
# 来源：slime/ray/train_actor.py L34-L42
if master_addr:
    self.master_addr, self.master_port = master_addr, master_port
else:
    self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
        start_port=random.randint(20000, 21000)
    )

os.environ["MASTER_ADDR"] = self.master_addr
os.environ["MASTER_PORT"] = str(self.master_port)
```

传播发生在 group 创建循环：

```python
# 来源：slime/ray/actor_group.py L105-L118
# Create worker actors
self._actor_handlers = []
master_addr, master_port = None, None
for rank in range(world_size):
    actor = TrainRayActor.options(
        num_cpus=num_gpus_per_actor,
        num_gpus=num_gpus_per_actor,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=reordered_bundle_indices[rank],
        ),
    ).remote(world_size, rank, master_addr, master_port)
    if rank == 0:
        master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
```

排障动作：多机环境要确认节点间能访问 rank 0 最终选出的端口，不要只放行 20000–21000；端口冲突或防火墙会表现为 distributed init 卡住。

## Q3：fractional GPU 会不会让多个 rank 随机共享一张 GPU？

不会按随机方式共享。每个 rank 都被固定到 `reordered_bundle_indices[rank]`。`num_gpus_per_actor=0.4` 是 Ray 资源 accounting，真正的 rank 到 bundle 映射由 placement group scheduling strategy 决定。

```python
# 来源：slime/ray/placement_group.py L140-L149
def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor", actor_cls=None):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,
        role=role,
        actor_cls=actor_cls,
    )
```

```python
# 来源：slime/ray/actor_group.py L109-L116
actor = TrainRayActor.options(
    num_cpus=num_gpus_per_actor,
    num_gpus=num_gpus_per_actor,
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=reordered_bundle_indices[rank],
    ),
).remote(world_size, rank, master_addr, master_port)
```

排障动作：如果 rank 与 GPU 对不上，先查 [[Slime-PlacementGroup]] 的 `reordered_bundle_indices`，再查 RayTrainGroup 的 rank 到 bundle index，而不是只盯 0.4。

## Q4：为什么不直接 pop `CUDA_VISIBLE_DEVICES`？

代码注释说明 Ray 已经影响了 `torch.cuda.device_count()`，直接 pop 不解决问题。Slime 选择把 Ray GPU id 映射到 CVD 列表里的 local ordinal。

```python
# 来源：slime/ray/train_actor.py L20-L25
def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))
```

```python
# 来源：slime/ray/train_actor.py L45-L48
# TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
# os.environ.pop("CUDA_VISIBLE_DEVICES", None)
# os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
os.environ["LOCAL_RANK"] = str(get_local_gpu_id())
```

排障动作：在 actor 进程里同时打印 `CUDA_VISIBLE_DEVICES`、`ray.get_gpu_ids()` 和 `LOCAL_RANK`，确认 local ordinal 是否可被 `torch.cuda.set_device` 使用。若 Ray GPU id 不在 CVD 列表中，当前实现会在 `.index(...)` 直接抛 `ValueError`。

## Q5：critic 能启用 routing replay 吗？

不能。环境变量只在 `role == "actor"` 时注入。

```python
# 来源：slime/ray/actor_group.py L86-L88
# We cannot do routing replay for critic.
if self.args.use_routing_replay and self.role == "actor":
    env_vars["ENABLE_ROUTING_REPLAY"] = "1"
```

排障动作：routing replay 问题如果发生在 critic 路径，不要期待 critic runtime env 里有 `ENABLE_ROUTING_REPLAY`。相关算法边界看 [[Slime-上下文并行与路由重放]]。

## Q6：`external_data` 什么时候传 list？

当每个 rank 需要不同外部输入时传 list；否则传单个对象广播给所有 rank。critic values refs 是典型 list 场景。下面只截取控制分支，不摘录文档字符串。

```python
# 定位骨架（据 `slime/ray/actor_group.py` L131-L149 删去 docstring 与广播分支）：
def async_train(self, rollout_id, rollout_data_ref, external_data=None):
    if isinstance(external_data, list):
        assert len(external_data) == len(self._actor_handlers)
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
            for actor, ed in zip(self._actor_handlers, external_data, strict=False)
        ]
```

排障动作：如果传 list，长度必须等于 actor 数。长度不匹配会 assert；通过后，第 `i` 个元素只交给第 `i` 个 actor rank。传单个 dict 才是所有 ranks 收到同一个对象。

## Q7：`set_rollout_manager` 为什么只有 rank 0 上报 parallel config？

训练并行配置只需要一份。group 仍会对所有 ranks 调用该方法，使每个 rank 保存 RolloutManager handle；rank 0 才额外将 `train_parallel_config` 推给 RolloutManager，供 rollout 侧构造 batch 时使用。

```python
# 来源：slime/ray/train_actor.py L125-L128
def set_rollout_manager(self, rollout_manager):
    self.rollout_manager = rollout_manager
    if not self.args.debug_rollout_only and self.args.rank == 0:
        ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
```

排障动作：如果 RolloutManager 缺少 train parallel config，查 rank 0 的 `set_rollout_manager` 是否执行成功，以及 actor init 是否已经填好了 `train_parallel_config`。

## Q8：offload_train 相关的 `LD_PRELOAD` 失败怎么定位？

`LD_PRELOAD` 在 actor 创建前注入。若 `torch_memory_saver` 动态库找不到，会在 `_allocate_gpus_for_actor` 里直接报 `FileNotFoundError`。

```python
# 来源：slime/ray/actor_group.py L64-L84
if self.args.offload_train and self.args.train_backend == "megatron":
    import torch_memory_saver

    for path in [
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ]:
        dynlib_path = os.path.join(
            os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
            path,
        )
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(
            "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
        )

    env_vars["LD_PRELOAD"] = dynlib_path
    env_vars["TMS_INIT_ENABLE"] = "1"
    env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
```

排障动作：确认 `torch_memory_saver` 安装路径下存在源码枚举的任一 `.so`。不要等到 Megatron actor init 后再查，这个错误发生在 Ray actor 创建前。若启动成功但 sleep/wake_up 卡住，再查 process group 的销毁/重载、memory saver 的 pause/resume，以及 PPO 非 colocate 场景的 rollout-engine 断连与重连。

## Q9：async train loop 为什么 update 前要 drain generate？

异步训练会预取下一轮 generate。如果此时更新权重，可能在一次 generate 途中改变 rollout engine 权重。`train_async.py` 在 update 前显式等待预取完成并清空 future。

```python
# 来源：train_async.py L30-L39
# async train loop.
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # Sync the last generation
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)

    # Start the next rollout early.
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
```

```python
# 来源：train_async.py L65-L69
if (rollout_id + 1) % args.update_weights_interval == 0:
    # sync generate before update weights to prevent update weight in the middle of generation
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None
    actor_model.update_weights()
```

排障动作：异步训练旧权重问题优先看 update interval 和 drain future 逻辑，而不是 RayTrainGroup 的 `update_weights` 本身。

下一篇 [[Slime-RayTrainGroup-学习检查]] 用场景题检查是否掌握。
