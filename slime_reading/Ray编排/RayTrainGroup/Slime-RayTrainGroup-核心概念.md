---
title: "RayTrainGroup · 核心概念"
type: concept
framework: slime
topic: "RayTrainGroup"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-12
---
# RayTrainGroup · 核心概念

## 你为什么要读

这篇先建立 RayTrainGroup 的心理模型。它不是 Megatron 训练器，而是训练侧 Ray actor 组的编排层：按 rank 创建 actor，注入环境变量，维护 actor handles，并把 driver 的远程调用发给每个 rank。

## 三层边界

| 层 | 对象 | 责任 |
|----|------|------|
| 资源座位表 | PlacementGroup 三元组 | 决定 rank actor 绑定哪个 Ray bundle |
| actor 编排 | `RayTrainGroup` | 创建 rank actor、保存 handles、分发 remote calls |
| 训练后端 | `TrainRayActor` / `MegatronTrainRayActor` | 初始化 distributed、加载模型、训练、保存、向 rollout 发布权重 |

源码 docstring 已经把 group 边界说清楚：`async` 开头的方法返回 object refs。

```python
# 来源：slime/ray/actor_group.py L10-L27
class RayTrainGroup:
    """
    A group of ray actors
    Functions start with 'async' should return list of object refs

    Args:
        args (Namespace): Arguments for the actor group.
        num_nodes (int): Number of nodes for this actor group.
        num_gpus_per_node (int): Number of gpus for this actor group.
        pg (PlacementGroup, optional): Placement group to schedule actor on.
            If none, create new placement group automatically. Defaults to None.
        num_gpus_per_actor (float, optional): Number of gpus allocated for each actor.
            If < 1.0, multiple models can share same gpu. Defaults to 1.
        resources (Dict[str, float], optional): Custom resources to allocate for each actor.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
        num_resources_per_node (int, optional): Number of custom resources to allocate for each node.
            See https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
    """
```

读者抓手：RayTrainGroup 的“训练”只是远程调用分发；真正的训练逻辑在 actor implementation。

## rank actor 的创建边界

RayTrainGroup 构造时会立刻调用 `_allocate_gpus_for_actor`。这一步创建 Ray actor，但还不初始化 Megatron 模型。

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

不变量：`world_size = num_nodes * num_gpus_per_node`，也就是要创建的 actor 数。

## master addr/port 是 rank 0 产生的

每个 actor 构造时都会写入 distributed env。rank 0 没有传入 master，所以自己找当前节点 IP 和空闲端口；后续 rank 复用 rank 0 返回的地址。`20000..21000` 只是随机搜索起点：`get_free_port()` 会从起点不断递增，因此最终端口不保证仍落在这个区间。

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

基础能力来自 `RayActor`：

```python
# 来源：slime/ray/ray_actor.py L4-L10
class RayActor:
    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        return get_current_node_ip(), get_free_port(start_port=start_port, consecutive=consecutive)

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port
```

心理模型：rank 0 是 rendezvous 的“报到点”，其他 rank 在构造函数里拿到同一个地址后，后续 `init()` 才真正加入 process group。

## `LOCAL_RANK` 不是直接等于物理 GPU id

Ray 可能已经设置了 `CUDA_VISIBLE_DEVICES`。如果可见设备被重映射，torch 需要的是本进程内 ordinal，而不是物理 id。

```python
# 来源：slime/ray/train_actor.py L20-L25
def get_local_gpu_id():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cvd is None:
        return ray.get_gpu_ids()[0]
    else:
        return cvd.split(",").index(str(ray.get_gpu_ids()[0]))
```

这段解释了为什么只看 `ray.get_gpu_ids()` 不够。`LOCAL_RANK` 要服务 `torch.cuda.set_device`。还有一个硬失败边界：CVD 存在但不包含 `str(ray.get_gpu_ids()[0])` 时，`.index(...)` 会直接抛 `ValueError`，这比“映射成错误 ordinal”更早暴露不一致。

## runtime env 是 actor 创建前注入的

训练 actor 的环境变量在 `_allocate_gpus_for_actor` 里准备。offload 的 `LD_PRELOAD` 必须在进程创建前生效，routing replay 也只给 actor role 开。

```python
# 来源：slime/ray/actor_group.py L55-L88
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

边界：critic 不启用 routing replay；offload_train 缺少 `torch_memory_saver` 动态库时会在 actor 创建前失败。

## 基类 `init` 建 distributed，Megatron 子类继续建模型

Ray actor 构造完成，只代表进程起来且 env 写好了。基类 `TrainRayActor.init()` 才调用 `dist.init_process_group`；默认使用的 `MegatronTrainRayActor.init()` 会先调用这个基类方法，再初始化 Megatron、模型和 optimizer，并最终返回 `start_rollout_id`。不要把“返回恢复 ID”误归给不返回该值的基类方法。

```python
# 来源：slime/ray/train_actor.py L50-L70
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
```

这一步之后，actor 内的 `args.rank/world_size` 才是 distributed 真实值。

## async API 与同步 API

RayTrainGroup 的 API 分两类：

| API | 返回什么 | 设计含义 |
|-----|----------|----------|
| `async_init` | ObjectRef list | API 把等待权交给 driver；当前 caller 是 critic 完成后才启动并等待 actor，不是二者并发 init |
| `async_train` | ObjectRef list | critic 每个 rank 的返回 ref 按位置交给对应 actor rank；rollout ref 则同一份传给所有 ranks |
| `save_model` | 已 `ray.get` 的结果 | 保存是生命周期操作，必须完成 |
| `update_weights` | 已 `ray.get` 的结果 | 下一轮 generate 前必须完成 |
| `onload` / `offload` | 已 `ray.get` 的结果 | 显存生命周期必须同步 |
| `clear_memory` | 已 `ray.get` 的结果 | 清理要覆盖所有 rank |
| `set_rollout_manager` | 已 `ray.get` 的结果 | rollout manager handle 和 parallel config 要下发完 |

源码边界：

```python
# 来源：slime/ray/actor_group.py L121-L149
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

    For critics, each ref resolves to ``{"values": [cpu tensors...]}`` (or ``{}``
    for non-last-PP-stage workers). Actor refs resolve to ``None``.

    ``external_data`` may be a list (one item per worker) or a single dict
    broadcast to all workers.
    """
    if isinstance(external_data, list):
        assert len(external_data) == len(self._actor_handlers)
        return [
            actor.train.remote(rollout_id, rollout_data_ref, external_data=ed)
            for actor, ed in zip(self._actor_handlers, external_data, strict=False)
        ]
    return [
        actor.train.remote(rollout_id, rollout_data_ref, external_data=external_data)
        for actor in self._actor_handlers
    ]
```

注意源码中 `update_weights` 的 docstring 已与当前 Megatron 实现漂移：group 层并没有实现“训练 rank 0 向其他训练 ranks 广播”，而是对每个 rank 调用后端 `update_weights()` 并等待。默认后端随后连接 RolloutManager 管理的 rollout engines，由 `weight_updater` 发布训练权重。

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

## `set_rollout_manager` 由 rank 0 上报 parallel config

训练 actor 初始化完成后，group 会向所有 rank 下发同一个 RolloutManager handle，每个 rank 都保存到 `self.rollout_manager`；只有 rank 0 额外把训练并行配置传给 RolloutManager。

```python
# 来源：slime/ray/train_actor.py L125-L128
def set_rollout_manager(self, rollout_manager):
    self.rollout_manager = rollout_manager
    if not self.args.debug_rollout_only and self.args.rank == 0:
        ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
```

这让 rollout 侧知道训练侧 DP/CP/VPP 等信息，后续才能构造能被 Megatron 消费的 batch。

## 复盘

RayTrainGroup 的关键不是“会训练”，而是四件事：

1. 按 [[Slime-PlacementGroup]] 的座位表创建 rank actor。
2. 让 rank 0 产生 distributed rendezvous 地址。
3. 把 actor 环境和 CUDA local rank 设置到 actor 进程内。
4. 用 async/sync API 区分训练数据流和生命周期一致性操作。

下一篇 [[Slime-RayTrainGroup-源码走读]] 沿这四件事读源码。
