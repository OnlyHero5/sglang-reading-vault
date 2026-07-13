---
title: "RayTrainGroup · 源码走读"
type: walkthrough
framework: slime
topic: "RayTrainGroup"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# RayTrainGroup · 源码走读

这篇追踪一条真实路径：`create_training_models` 拿到 `pgs["actor"]` 后，构造 RayTrainGroup；RayTrainGroup 创建每个 rank actor；rank 0 选 master addr/port；`async_init` 触发 actor 子类初始化，其中基类建立 distributed、Megatron 子类继续建模型；主循环用 `async_train` 发训练，用 `update_weights` 做“训练侧向 rollout 发布权重”的同步闸门。

## 长文读法

这篇按“Ray actor handle 到 Megatron 训练进程”的边界读：RayTrainGroup 构造时只把 actor 放到 PG bundle 上，`TrainRayActor.__init__` 写分布式环境变量，`async_init` 才进入 torch distributed 和后端初始化，训练、保存、offload、权重同步都是 group 对每个 rank 发 remote 调用。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立 RayTrainGroup 主线 | 贯穿场景、步骤一到四 | 构造 group 只创建 Ray actor，不等于模型或 distributed 已就绪 |
| 排查 rank 环境变量 | 步骤四到五 | rank 0 先生成 master addr/port，后续 actor 复用并写入 env |
| 排查 init 卡住 | 步骤六 | `init` 才设置 CUDA device、建 process group 和 gloo group |
| 排查训练调用没执行 | 步骤七 | `async_train` 只返回 ObjectRef，真正等待发生在主循环的 `ray.get` |
| 排查 update/save/offload 同步 | 步骤八 | group 方法对全部 actor 发 remote，再统一 `ray.get` 做闸门 |
| 排查后端职责 | 步骤九 | 抽象接口把 `train/save/update_weights/sleep/wake_up` 留给 Megatron actor 实现 |

读的时候不要把 RayTrainGroup 当训练逻辑本身。它是 rank actor 编排层，真正的训练和权重细节在 Megatron actor。

## 贯穿场景

同步训练入口里，RayTrainGroup 出现在 PG 和 RolloutManager 之后：

```python
# 来源：train.py L9-L20
def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
```

`create_training_models` 内部通过 `allocate_train_group` 创建 RayTrainGroup：

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

## 步骤一：RayTrainGroup 构造时只创建 actor，不初始化模型

系统压力：训练 actor 要先被 Ray 调度到 PG bundle 上，但 Megatron 模型初始化很重，必须由后续显式 init 控制。

设计选择：构造函数保存拓扑和 role，立即调用 `_allocate_gpus_for_actor` 创建 Ray actors。

```python
# 来源：slime/ray/actor_group.py L29-L46
def __init__(
    self,
    args,
    num_nodes,
    num_gpus_per_node,
    pg: tuple[PlacementGroup, list[int], list[int]],
    num_gpus_per_actor: float = 1,
    role: str = "actor",
    actor_cls=None,
) -> None:
    self.args = args
    self._num_nodes = num_nodes
    self._num_gpus_per_node = num_gpus_per_node
    self.role = role
    self._actor_cls = actor_cls

    # Allocate the GPUs for actors w/o instantiating them
    self._allocate_gpus_for_actor(pg, num_gpus_per_actor)
```

执行逻辑：`RayTrainGroup` 构造成功只说明 Ray actor handles 已建立，不说明 torch distributed 或 Megatron model 已就绪。

## 步骤二：解包 PG 三元组并准备 runtime env

系统压力：actor 进程创建前要决定环境变量。特别是 `LD_PRELOAD`，进程启动后再设置已经太晚。

设计选择：先计算 world size，解包 PG 三元组，再构造 NCCL、TransformerEngine、visible-devices、用户 train env、offload preload、routing replay env。

```python
# 来源：slime/ray/actor_group.py L48-L88
def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
    world_size = self._num_nodes * self._num_gpus_per_node

    # Use placement group to lock resources for models of same type
    assert pg is not None
    pg, reordered_bundle_indices, _reordered_gpu_ids = pg

    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1"),
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **self.args.train_env_vars,
    }

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

    # We cannot do routing replay for critic.
    if self.args.use_routing_replay and self.role == "actor":
        env_vars["ENABLE_ROUTING_REPLAY"] = "1"
```

不变量：

- `pg` 不能为 None。
- `reordered_bundle_indices` 长度要覆盖 world size。
- offload_train + Megatron 必须能找到 torch_memory_saver 动态库。
- routing replay 只给 actor，不给 critic。

## 步骤三：选择 actor class 并包装成 Ray remote class

系统压力：默认后端是 Megatron，但测试和扩展需要替换 actor class；NIXL transport 需要在 Ray actor options 上开启。

设计选择：没有自定义 class 时导入 `MegatronTrainRayActor`，再用 `ray.remote(**actor_options)` 包装。

```python
# 来源：slime/ray/actor_group.py L90-L103
if self._actor_cls is None:
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor

    actor_impl = MegatronTrainRayActor
else:
    actor_impl = self._actor_cls

actor_options = {
    "num_gpus": 1,
    "runtime_env": {"env_vars": add_default_ray_env_vars(env_vars)},
}
if getattr(self.args, "rollout_data_transport", "object-store") == "nixl":
    actor_options["enable_tensor_transport"] = True
TrainRayActor = ray.remote(**actor_options)(actor_impl)
```

读者抓手：这里得到的是 Ray actor class，不是具体 actor 实例。实例在下一步按 rank 创建。

## 步骤四：按 rank 创建 actor，rank 0 先产生 master

系统压力：所有 rank 要加入同一个 process group，因此必须共享 rank 0 的 master addr/port。

设计选择：按 rank 顺序发起 actor 创建；rank 0 的构造参数里 master 为空，它以 20000–21000 的随机值为起点向上寻找空闲端口；driver 立刻取回 master，再传给后续 rank。后续 rank 的 `.remote()` 不逐个 `ray.get`，因此“按 rank 发起”不等于“逐个等待其完全初始化”。

```python
# 来源：slime/ray/actor_group.py L105-L119
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
    self._actor_handlers.append(actor)
```

执行逻辑：

- rank 绑定 `reordered_bundle_indices[rank]`。
- rank 0 创建后同步取 master addr/port。
- actor handles 按 rank 顺序保存在 `_actor_handlers`。
- `num_gpus_per_actor=0.4` 只影响 Ray accounting；bundle 绑定仍是一 rank 一个 logical slot。

## 步骤五：TrainRayActor 构造函数写 distributed env

系统压力：Ray actor 是独立进程，不能继承 driver 的 distributed 状态。每个 actor 必须在自己进程里写 MASTER/RANK/WORLD/LOCAL_RANK。

设计选择：rank 0 自选地址，其他 rank 用传入地址；`LOCAL_RANK` 通过 `get_local_gpu_id` 映射。

```python
# 来源：slime/ray/train_actor.py L28-L48
class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port):
        configure_logger()

        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())
```

`get_local_gpu_id` 处理 Ray CVD 重映射：

```python
# 来源：slime/ray/train_actor.py L20-L25
def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))
```

若 CVD 已存在却找不到 Ray 返回的 GPU id，`.index(...)` 会抛 `ValueError`；这通常意味着 Ray 资源视图与 actor 进程可见设备列表不一致。

## 步骤六：`init` 才真正初始化 distributed

系统压力：actor 创建只是进程启动；模型、optimizer、process group 初始化要由 driver 明确触发。API 返回 refs，具备组合空间，但当前 `create_training_models` 会先 `ray.get` critic init，再发起并等待 actor init，实际调用路径不是 actor/critic 同时初始化。

设计选择：`async_init` 返回 refs，actor 内 `init` 设置 CUDA device、初始化 process group、初始化 Gloo group、写回 args rank/world_size，并尝试设置 NUMA affinity。

```python
# 定位骨架（据 `slime/ray/train_actor.py` L50-L92 删节）：
def init(self, args, role, with_ref=False, with_opd_teacher=False):
    self.args = args
    self.role = role
    self.with_ref = with_ref
    self.with_opd_teacher = with_opd_teacher

    torch.serialization.add_safe_globals([slime.utils.eval_config.EvalDatasetConfig])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(f"cuda:{local_rank}")

    backend = args.distributed_backend

    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
    init_gloo_group()

    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    try:
        if torch.version.hip is not None:
            logger.info("Detected ROCm/HIP environment, skipping NUMA affinity setup")
            # will find the coresponding API to implement ROCm version as below
        else:
            import pynvml
```

默认子类把这段基础初始化接到 Megatron 初始化上：

```python
# 来源：slime/backends/megatron_utils/actor.py L46-L62
class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)

        init(args)
```

该子类在模型与恢复状态初始化结束后才 `return start_rollout_id`；因此 group 收到的恢复 ID 来自默认子类协议，不是基类 distributed bootstrap 的返回值。

```python
# 来源：slime/utils/distributed_utils.py L20-L33
def init_gloo_group():
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP
```

不变量：构造函数里写好的 MASTER/RANK/WORLD 环境变量必须存在，否则 `env://` rendezvous 无法完成。

## 步骤七：`async_init` 和 `async_train` 只返回 refs

系统压力：driver 需要组合 actor/critic 结果。例如 PPO 中 critic 先返回 values refs，再作为 actor 训练的 external data。

设计选择：`async_init`、`async_train` 都返回 ObjectRef 列表，不在 group 内等待。下面是删去 `async_train` docstring 和广播分支的定位骨架，不是连续逐行摘录。

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
    if isinstance(external_data, list):
        assert len(external_data) == len(self._actor_handlers)
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
            for actor, ed in zip(self._actor_handlers, external_data, strict=False)
        ]
    return [
```

主循环消费这个边界：

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

读者抓手：critic group 返回一份“每 critic rank 一个 ref”的列表；actor group 先断言列表长度等于 actor 数，再按位置把第 `i` 个 ref 交给第 `i` 个 actor。它不是把一个 values dict 广播给所有 actor。`rollout_data_ref` 才是同一个 ObjectRef 传给所有 ranks。

## 步骤八：update/save/offload 这类操作在 group 内同步

系统压力：权重同步、保存、onload/offload 都是生命周期闸门。下一轮 generate 或训练不能在它们还没完成时继续。

设计选择：group 方法内部直接 `ray.get` 所有 actor 的远程调用。这里引用到的 `update_weights` docstring 已经漂移：当前 group 只做“全 rank 调用 + 等待”，默认 Megatron actor 才负责连接 rollout engines 并通过 `weight_updater.update_weights()` 发布权重。

```python
# 来源：slime/ray/actor_group.py L151-L169
def save_model(self, rollout_id, force_sync=False):
    """Save actor model"""
    return ray.get([actor.save_model.remote(rollout_id, force_sync=force_sync) for actor in self._actor_handlers])

def update_weights(self):
    """Broadcast weights from rank 0 to all other ranks."""
    return ray.get([actor.update_weights.remote() for actor in self._actor_handlers])

def onload(self):
    return ray.get([actor.wake_up.remote() for actor in self._actor_handlers])

def offload(self):
    return ray.get([actor.sleep.remote() for actor in self._actor_handlers])

def clear_memory(self):
    return ray.get([actor.clear_memory.remote() for actor in self._actor_handlers])

def set_rollout_manager(self, rollout_manager):
    return ray.get([actor.set_rollout_manager.remote(rollout_manager) for actor in self._actor_handlers])
```

同步训练循环里，`update_weights` 是 generate 前的硬闸门：

```python
# 来源：train.py L83-L92
if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
    save(rollout_id)

offload_train(actor_trains_this_step)
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())
actor_model.update_weights()

if args.offload_rollout:
    ray.get(rollout_manager.onload_kv.remote())
```

异步训练里也会在 update 前 drain 预取的 generate：

```python
# 来源：train_async.py L65-L69
if (rollout_id + 1) % args.update_weights_interval == 0:
    # sync generate before update weights to prevent update weight in the middle of generation
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None
    actor_model.update_weights()
```

## 步骤九：TrainRayActor 抽象接口把后端细节留给 Megatron actor

系统压力：Group 层要能调用统一接口，但不应该知道 Megatron 里如何 train、save、sleep、update weights。

设计选择：TrainRayActor 定义抽象方法；默认 `MegatronTrainRayActor` 实现它们。

```python
# 来源：slime/ray/train_actor.py L101-L128
@abc.abstractmethod
def sleep(self, tags):
    raise NotImplementedError

@abc.abstractmethod
def wake_up(self, tags):
    raise NotImplementedError

@abc.abstractmethod
def train(self, rollout_id, rollout_data_ref, external_data=None):
    raise NotImplementedError

@abc.abstractmethod
def save_model(self, rollout_id, force_sync=False):
    raise NotImplementedError

@abc.abstractmethod
def update_weights(self):
    raise NotImplementedError

@abc.abstractmethod
def _get_parallel_config(self):
    raise NotImplementedError

def set_rollout_manager(self, rollout_manager):
    self.rollout_manager = rollout_manager
    if not self.args.debug_rollout_only and self.args.rank == 0:
        ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
```

`set_rollout_manager` 是基类实现。group 会对所有 ranks 调用，所以所有 ranks 都保存 manager handle；仅 rank 0 额外同步上报训练并行配置。

这里还有一处扩展接口漂移：抽象声明是 `sleep(self, tags)` / `wake_up(self, tags)`，但 group 当前无参调用，默认 `MegatronTrainRayActor.sleep()` / `wake_up()` 也采用无参签名。默认路径能运行；实现新后端时应以实际 group 调用约定为准，并把这处签名差异视为待上游收口的问题。

## 运行验证

轻量测试：

```powershell
Set-Location slime
python -m pytest tests/test_megatron_argument_validation.py -q
```

若环境具备 `ray` 和 `sglang`，再跑：

```powershell
Set-Location slime
python -m pytest tests/utils/test_megatron_role_config.py -q
```

预期现象：

- role config 测试会覆盖 `create_training_models` 使用 actor override 的路径。
- 缺 `ray` 或 `sglang` 时会在 import 阶段失败，不代表 RayTrainGroup 文档行为错误。
- 当前基线实测参数校验为 `14 passed`；role config 为 6 个 import 失败（5 个缺 `sglang`，1 个缺 `ray`）。

## 复盘迁移

读 RayTrainGroup 源码时，按两个问题切：

1. 创建期：rank actor 如何坐到 PG bundle 上，如何共享 master addr/port。
2. 运行期：哪些 API 返回 refs，哪些 API 在 group 内同步。

下一篇 [[Slime-RayTrainGroup-数据流]] 把这些调用放回训练闭环。
