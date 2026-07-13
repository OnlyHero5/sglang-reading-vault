---
title: "PlacementGroup · 源码走读"
type: walkthrough
framework: slime
topic: "PlacementGroup"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# PlacementGroup · 源码走读

这篇追踪一条真实启动路径：`train.py` 拿到最终 args 后，先生成 PG 资源座位表，再创建 RolloutManager，最后创建 actor/critic RayTrainGroup。

读完后，你应该能解释：资源为什么卡在 PG ready、为什么要用 InfoActor 探测、为什么 rollout 视图来自同一重排列表的切片、为什么 colocate 和 external 的 GPU 数不能按直觉相加。

## 长文读法

这篇按“最终 args 如何变成 Ray 的座位表”读：`train.py` 先创建 placement group；`_get_placement_group_layout` 把 debug、colocate、external、zero rollout 编译成 `(num_gpus, rollout_offset)`；PG ready 后用 InfoActor 探测真实 GPU 顺序；actor、critic、rollout 都只是同一张座位表上的不同切片。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立资源主线 | 贯穿场景、步骤一 | PG 是 rollout manager 和 train group 的前置依赖 |
| 排查 GPU 数不符合直觉 | 步骤一 | colocate 取 max，external 只申请 actor，debug 模式会重写布局 |
| 排查 PG ready 卡住 | 步骤二 | ready 轮询会周期性打印 Ray 注册和可用 GPU 数 |
| 理解 InfoActor | 步骤三到四 | InfoActor 是为了拿真实节点/GPU id，再按稳定规则重排 bundle |
| 排查 actor/critic/rollout 绑卡 | 步骤五到八 | actor/critic 复用前缀，rollout 从 offset 开始；0.4/0.2 是 Ray accounting，不是显存切片 |
| 排查 init 顺序 | 步骤九 | RolloutManager 先于 train models 创建，actor/critic init 后再接 rollout manager |

读的时候不要把“申请了多少 GPU”和“某个组件实际使用哪段 GPU”混在一起。前者由 layout 决定，后者由 slice 和 bundle index 决定。

## 贯穿场景

同步训练入口中，PG 是所有 Ray 组件前置依赖：

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

这段说明一个顺序约束：先有资源座位表，才有 rollout engine 和 Megatron actor。

## 步骤一：把 args 编译成 `(num_gpus, rollout_offset)`

系统压力：启动参数经过 [[Slime-Ray参数]] 校验后，仍要落到 Ray 能理解的 bundle 数。colocate、external、debug、zero rollout 都会改变本地 PG 申请量。

设计选择：`_get_placement_group_layout` 只输出两个数，让后续统一切片。

```python
# 来源：slime/ray/placement_group.py L100-L117
def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0

    if args.rollout_external:
        if args.debug_rollout_only:
            return 0, 0
        return actor_num_gpus, actor_num_gpus

    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0

    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0

    return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus
```

执行逻辑：

- 普通分离：申请 actor + rollout，rollout 从 actor 后面开始。
- colocate：申请两者最大值，rollout 从 0 开始；真实共享范围是共同前缀。
- external：只申请 actor，rollout offset 等于 actor 数，切出来为空。
- debug rollout only：只申请 rollout。
- zero rollout：普通分离下 rollout 视图为空，但 actor 仍保留。

测试把这个矩阵固定为可验证事实：

```python
# 来源：tests/test_placement_group.py L30-L46
@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (0, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected
```

## 步骤二：创建 PG 时不静默等待

系统压力：Ray PG 可能因为 GPU 尚未注册或 autoscaler 正在扩容而 pending。裸等 `pg.ready()` 会让用户不知道是资源不足还是进程挂住。

设计选择：创建 `PACK` placement group 后，用 `ray.wait` 每 30 秒轮询 ready ref，同时打印 Ray 注册 GPU 和可用 GPU。

```python
# 来源：slime/ray/placement_group.py L42-L67
def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return None, [], []

    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

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

不变量与失败模式：

- `num_gpus == 0` 返回空三元组，不创建 Ray PG。
- 等待无上限；资源永远不足时会持续打印等待日志。
- 日志反映 Ray 资源视图，不保证后续 NCCL 或 CUDA 初始化一定成功。

## 步骤三：InfoActor 探测 bundle 的节点和 GPU

系统压力：Ray 原始 bundle 顺序不是稳定的物理拓扑顺序。训练 rank 和 SGLang engine 需要统一的 logical order，否则日志和故障定位会混乱。

设计选择：每个 bundle 创建一个临时 `InfoActor`，读取所在节点 IP 和 Ray GPU id，然后立刻 kill。

```python
# 来源：slime/ray/placement_group.py L69-L82
# use info actor to get the GPU id
info_actors = []
for i in range(num_bundles):
    info_actors.append(
        InfoActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=i,
            ),
        ).remote()
    )
gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
for actor in info_actors:
    ray.kill(actor)
```

执行逻辑：

- `InfoActor` 被固定到指定 bundle。
- `get_ip_and_gpu_id` 读 Ray 视角的节点和 GPU。
- 探测完成立刻释放 actor，避免占用后续训练或 rollout 资源。

## 步骤四：按节点和 GPU id 生成 logical order

系统压力：后续组件要复用同一套座位表。训练 rank 和 rollout engine 不能各自重新理解 Ray bundle 顺序。

设计选择：`sort_key` 先按 IP 数字排序，hostname 可解析时转 IP，无法解析时用字符序列兜底；最终返回 logical index 到原始 bundle index 和 GPU id 的映射。

```python
# 来源：slime/ray/placement_group.py L21-L39
def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)
```

```python
# 来源：slime/ray/placement_group.py L84-L97
bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
# Map from logical index -> physical GPU ID
pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

for i in range(num_bundles):
    actual_bundle_index = pg_reordered_bundle_indices[i]
    logger.info(
        f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
        f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
    )

return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids
```

读者抓手：日志里的 `bundle i` 是 logical index，`actual_bundle_index` 是 Ray 原始 bundle index。

## 步骤五：切出 actor、rollout、critic 三个角色视图

系统压力：下游既有训练 actor，又有 rollout engine，还有可选 critic。它们要共享同一套 PG 拓扑，但角色看到的 slice 不同。

设计选择：`create_placement_groups` 对重排列表做切片；critic 复用 actor 视图。

```python
# 来源：slime/ray/placement_group.py L120-L137
def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result
```

不变量：

- `pgs["actor"]` 总是从 logical 0 开始。
- `pgs["rollout"]` 从 `rollout_offset` 开始。
- external 普通训练中 rollout 视图为空，但 external server 仍可被 HTTP 控制。
- critic 视图等于 actor 视图，不会单独申请 PG；参数校验也把 critic GPU 数强制等于 actor。
- colocate 时两侧从 0 开始，但只在前 `min(A,R)` 个 slot 重叠，较大一侧有独占后缀。

## 步骤六：RolloutManager 是控制面，不占 GPU

系统压力：RolloutManager 要先创建，因为它能推导 `num_rollout`；但它自己不跑 SGLang GPU 计算。

设计选择：RolloutManager Ray actor 用 `num_gpus=0`，拿到 rollout PG 三元组，在内部启动 SGLang engines；同时处理 `num_rollout` 推导、权重一致性快照和 rollout offload。

```python
# 来源：slime/ray/placement_group.py L220-L246
def create_rollout_manager(args, pg):
    from .rollout import RolloutManager

    rollout_manager_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": add_default_ray_env_vars()},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        rollout_manager_options["enable_tensor_transport"] = True
    rollout_manager = RolloutManager.options(**rollout_manager_options).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
```

读者抓手：GPU 真正被 SGLangEngine actor 使用，不是 RolloutManager 本身。

## 步骤七：SGLang engine 使用 rollout 视图绑定 bundle

系统压力：rollout 视图可能是完整 PG、后半段切片、空切片，也可能与 actor 重叠。SGLang engine 必须按同一套 logical order 找到 bundle 和 base GPU。

设计选择：`ServerGroup.start_engines` 解包 rollout PG 三元组，用 `gpu_offset` 选择 logical GPU，再把 actor 调度到对应 bundle。

```python
# 来源：slime/ray/rollout.py L154-L187
num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)

pg, reordered_bundle_indices, reordered_gpu_ids = self.pg
validate_server_group_gpu_indices(
    worker_type=self.worker_type,
    gpu_offset=self.gpu_offset,
    num_gpus_per_engine=self.num_gpus_per_engine,
    num_gpu_per_engine=num_gpu_per_engine,
    num_engines=len(self.all_engines),
    num_available_gpus=len(reordered_gpu_ids),
    rollout_num_gpus=self.args.rollout_num_gpus,
    rollout_num_gpus_per_engine=self.args.rollout_num_gpus_per_engine,
)

RolloutRayActor = ray.remote(SGLangEngine)

rollout_engines = []
for i in range(len(self.all_engines)):
    if self.all_engines[i] is not None:
        continue

    global_rank = self.rank_offset + i
    num_gpus = 0.2
    num_cpus = num_gpus

    # Get the base GPU ID from placement group using gpu_offset.
    gpu_index = self.gpu_offset + i * num_gpu_per_engine
    base_gpu_id = int(reordered_gpu_ids[gpu_index])

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=reordered_bundle_indices[gpu_index],
    )
```

执行逻辑：

- `gpu_offset` 是 rollout 视图内部偏移，不是全局 actor 偏移。
- `base_gpu_id` 来自 `reordered_gpu_ids`。
- Ray scheduling 用 `reordered_bundle_indices[gpu_index]`。
- `num_gpus=0.2` 是 Ray 调度份额，engine 进程内部仍按 SGLang 参数使用对应 GPU。

## 步骤八：RayTrainGroup 使用 actor/critic 视图绑定 rank

系统压力：Megatron rank 必须被固定到对应 bundle。否则 rank 到物理 GPU 的关系会漂移，NCCL、checkpoint 和日志都难排查。

设计选择：RayTrainGroup 解包 actor PG 三元组，每个 rank 用 `PlacementGroupSchedulingStrategy` 绑定到 `reordered_bundle_indices[rank]`。

```python
# 来源：slime/ray/actor_group.py L48-L62
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
```

```python
# 来源：slime/ray/actor_group.py L97-L116
actor_options = {
    "num_gpus": 1,
    "runtime_env": {"env_vars": add_default_ray_env_vars(env_vars)},
}
if getattr(self.args, "rollout_data_transport", "object-store") == "nixl":
    actor_options["enable_tensor_transport"] = True
TrainRayActor = ray.remote(**actor_options)(actor_impl)

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
```

注意：`ray.remote(**actor_options)` 先声明 actor 类默认资源，`TrainRayActor.options(num_gpus=num_gpus_per_actor)` 在每个 rank 创建时覆盖调度份额。当前调用方给 `num_gpus_per_actor=0.4`；SGLang engine control actor 使用 0.2。它们是 Ray admission accounting，Slime 同时设置 NOSET visible-device 环境变量，实际模型仍会看见并使用整张目标设备。PPO 下 actor 0.4 + critic 0.4 + colocated rollout control actor 0.2 恰好可落在同一 bundle，但显存安全仍依赖 offload。

## 步骤九：create_training_models 选择 start rollout id，但不交叉校验 actor/critic

系统压力：resume 时 actor/critic 的恢复点若不一致，数据游标和 checkpoint 可能错位；因此必须看清代码究竟校验了什么。

设计选择：先按角色解析 args 并创建 RayTrainGroup，再分别 `async_init`。启用 critic 时只把 critic 返回列表赋给 `start_rollout_ids`，随后检查该列表内部集合大小为 1；actor 返回列表没有与 critic 比较。

```python
# 来源：slime/ray/placement_group.py L152-L168
def create_training_models(args, pgs, rollout_manager, actor_cls=None):
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model_kwargs = {}
    if actor_cls is not None:
        actor_model_kwargs["actor_cls"] = actor_cls
    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        **actor_model_kwargs,
    )
```

```python
# 来源：slime/ray/placement_group.py L189-L217
        critic_start_rollout_ids = ray.get(critic_model.async_init(critic_model.args, role="critic", with_ref=False))

    actor_start_rollout_ids = ray.get(
        actor_model.async_init(
            actor_args,
            role="actor",
            with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
            with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        )
    )
    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if args.use_critic:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    if args.rollout_global_dataset:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model
```

真实边界：

- 无 critic 时，actor 各 rank 的 ids 必须一致。
- 有 critic 时，critic 各 rank 的 ids 必须一致；actor ids 在这条 assert 中未被检查，更没有 actor/critic 交叉比较。
- 显式 `start_rollout_id` 只阻止自动写回，不改变上述检查范围。
- `rollout_manager` 创建必须早于 training models。
- `rollout_global_dataset` 要等 `start_rollout_id` 决定后再 load。

## 运行验证

布局矩阵：

```powershell
Set-Location slime
python -m pytest tests/test_placement_group.py -q
```

角色配置和参数校验：

```powershell
Set-Location slime
python -m pytest tests/utils/test_megatron_role_config.py -q
python -m pytest tests/test_megatron_argument_validation.py -q
```

预期现象：

- `test_placement_group_layout` 覆盖 normal、debug、colocate、external、zero rollout。
- role config 测试证明 `create_training_models` 会把 actor override 应用到 actor args。
- argument validation 测试证明 colocate 自动打开 offload，并拒绝 delta weight update。

环境分级：当前环境中参数校验 14 项实跑通过；placement 测试因缺 `ray`、role-config 测试因缺 `sglang/ray` 无法 collection。作为静态替代，从当前 `placement_group.py` AST 抽取布局函数执行测试文件同款 10 个 case，全部通过。真实 PG ready、InfoActor 探测和 fractional actor 共席仍需 Ray 集群验证。

## 复盘迁移

读 PG 源码时按四层分开：

1. 布局层：`num_gpus` 和 `rollout_offset`。
2. 拓扑层：InfoActor 探测和重排。
3. 视图层：actor、rollout、critic 三元组。
4. 消费层：RolloutManager、ServerGroup、RayTrainGroup。

下一篇 [[Slime-PlacementGroup-数据流]] 把这四层压成资源流图。
