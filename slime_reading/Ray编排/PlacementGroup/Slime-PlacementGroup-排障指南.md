---
title: "PlacementGroup · 排障指南"
type: troubleshooting
framework: slime
topic: "PlacementGroup"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# PlacementGroup · 排障指南

这篇按症状排障。先判断问题发生在本地 Ray PG 申请量、bundle 重排、角色视图切片、Ray actor 调度，还是 colocate/external 的资源语义。

## 症状总表

| 症状 | 先查什么 | 源码入口 | 验证方式 |
|------|----------|----------|----------|
| 启动卡在 `Waiting for placement group` | `num_gpus` 是否超过 Ray 可用 GPU | `slime/ray/placement_group.py` L42-L67 | 看日志里的 registered/available GPU |
| external rollout 仍以为要本地 rollout GPU | `rollout_external` 是否让 rollout 视图为空 | `slime/ray/placement_group.py` L106-L109 | `tests/test_placement_group.py` external case |
| colocate 下显存 OOM | offload 标志和 onload/offload 生命周期 | `slime/utils/arguments.py` L1885-L1901 | 参数校验测试和运行日志 |
| SGLang engine 绑定错 GPU | `reordered_bundle_indices` 与 `gpu_offset` | `slime/ray/rollout.py` L154-L187 | 看 `bundle -> actual_bundle_index` 日志 |
| Megatron rank 资源错位 | RayTrainGroup 是否用 reordered bundle index | `slime/ray/actor_group.py` L105-L116 | 对照 rank 与 PG 日志 |
| actor/critic resume id 看似不一致却未报错 | 当前只校验被选中的 critic 各 rank，并不比较 actor 与 critic | `slime/ray/placement_group.py` L189-L208 | 分别记录两侧 init 返回值，必要时显式指定恢复策略 |
| rollout-only 后缀也发生 offload | `needs_offload` 是否按跨边界 group 的起点粗粒度判定 | `slime/ray/rollout.py` L1113-L1146 | 看 group `gpu_offset/abs/needs_offload` 日志并拆 group 验证 |
| zero rollout 还创建训练 PG | 这是不是 non-colocate 场景 | `tests/test_placement_group.py` L39-L40 | 跑布局测试 |

## Q1：PG 等待会不会自动超时失败？

不会。源码故意保留无上限等待，因为 pending placement group 可以驱动 Ray autoscaler 扩容。但它会每 30 秒打印资源状态。

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

排障动作：

- 如果 registered GPU 小于 `num_gpus`，先查 Ray 节点注册或 autoscaler。
- 如果 registered 足够但 available 不足，查是否已有 job 占用资源。
- 如果 Ray ready 之后才失败，问题通常转向 actor 初始化、NCCL、CUDA 或模型加载。

## Q2：为什么需要 InfoActor 重排 bundle？

因为 Ray 原始 bundle index 不承诺按物理节点和 GPU id 排好。Slime 要给 Megatron rank 和 SGLang engine 一个稳定 logical order。

```python
# 来源：slime/ray/placement_group.py L69-L88
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

bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
# Map from logical index -> physical GPU ID
pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
```

判断方法：如果日志里 logical `bundle i` 和 `actual_bundle_index` 不一致，这是正常的重排行为，不是资源重复。

## Q3：colocate 下 actor 和 rollout 会不会同时占同一张 GPU？

Ray 视角上它们可以共享同一套 bundle；显存能否同时占用由 offload/onload 生命周期控制。参数校验会在 colocate 下默认打开 train 和 rollout offload。

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

测试覆盖了 colocate 的几个关键边界：

```python
# 来源：tests/test_megatron_argument_validation.py L264-L285
def test_slime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(colocate=True, rollout_num_gpus=0)

    module.slime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_slime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.slime_validate_args(args)
```

排障动作：colocate OOM 时，不要只看 PG layout。先算前 `min(A,R)` 个重叠 slot，再看 `offload_rollout`、`offload_train`、`rollout_manager.offload`、actor wake/sleep 以及 SGLang memory saver。若一个 ServerGroup 横跨 actor 边界，当前 group-start 判定会让整组进入 offload 生命周期。

## Q4：external rollout 时 PG 如何分配？

external rollout server 不属于本地 Ray job 的 GPU 资源。本地 PG 只保留 actor；rollout offset 等于 actor GPU 数，所以 rollout 视图为空。

```python
# 来源：slime/ray/placement_group.py L106-L109
if args.rollout_external:
    if args.debug_rollout_only:
        return 0, 0
    return actor_num_gpus, actor_num_gpus
```

单测固定这个期望：

```python
# 来源：tests/test_placement_group.py L39-L46
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (0, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected
```

如果 external 模式下看到本地 PG 申请 actor 之外的 GPU，先检查 `rollout_external_engine_addrs` 是否真的触发了 `rollout_external`，以及参数校验是否被绕过。

## Q5：为什么 `num_gpus_per_actor=0.4`，但 actor class options 里又有 `num_gpus=1`？

当前执行路径是两层 Ray options：先把 actor implementation 包成 remote class，默认 `num_gpus=1`；真正创建每个 rank actor 时，再用 `.options(num_gpus=num_gpus_per_actor)` 覆盖调度份额。

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

解释边界：0.4 是 Ray 调度 accounting，不表示 Megatron 只使用 0.4 张 GPU。rank 仍被固定到一个 placement group bundle，actor 内部设备可见性由 Ray 和 Slime 环境变量共同控制。

## Q6：为什么 actor 和 critic 的 `start_rollout_id` 不一致也可能不报错？

源码只要求“最终被选择的列表”在其内部一致。启用 critic 时，最终列表是 `critic_start_rollout_ids`；`actor_start_rollout_ids` 虽然被计算，却没有与 critic 列表比较。待办注释说明这个所有权尚未妥善决定。

```python
# 来源：slime/ray/placement_group.py L189-L208
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
```

排障动作：

- 分别打印或检查 actor、critic 的 init 返回值，不要拿当前 assert 当成交叉一致性证明。
- critic 各 rank 不一致会在 assert 处失败；actor 各 rank 在 use_critic 路径中不会被该 assert 检查。
- 显式 `--start-rollout-id` 只阻止自动写回，并不会新增 actor/critic 交叉校验；恢复策略仍需操作者保证。

## Q7：delta weight update 为什么拒绝 colocate？

这是参数校验层的规则，不是 PG 层直接判断。原因是 colocate 通过 CUDA IPC 传权重 handle，delta 的 snapshot/diff/encode 是额外开销。

```python
# 来源：slime/utils/arguments.py L1988-L1997
            raise ValueError(
                "--update-weight-mode=delta requires --update-weight-transport=disk, "
                f"got {args.update_weight_transport!r}."
            )
        if args.colocate:
            raise ValueError(
                "--update-weight-mode=delta is not supported with --colocate. Colocate transfers "
                "weights via CUDA IPC (only a handle crosses processes), so the delta bookkeeping "
                "(snapshot + diff + encode) is pure overhead."
            )
```

对应测试：

```python
# 来源：tests/test_megatron_argument_validation.py L320-L331
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_slime_arguments_module(monkeypatch)
    args = make_slime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/ckpt",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.slime_validate_args(args)
```

排障动作：colocate 权重同步问题应优先看 `UpdateWeightFromTensor` 和 CUDA IPC 路径，而不是 delta disk 路径。

## Q8：zero rollout GPU 为什么普通分离下仍申请 actor PG？

普通分离下 `rollout_num_gpus=0` 的结果是 `(actor_num_gpus, actor_num_gpus)`，也就是本地 PG 只保留 actor，rollout 视图为空。它不是“整个训练不需要 GPU”。

```python
# 来源：tests/test_placement_group.py L33-L40
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
```

判断标准：如果你的意图是只跑 rollout，不要靠 `rollout_num_gpus=0`，而是看 `debug_rollout_only`。如果你的意图是训练但不启动本地 rollout server，则 zero rollout 或 external 语义才接近。

下一篇 [[Slime-PlacementGroup-学习检查]] 用场景推导检查是否掌握。
