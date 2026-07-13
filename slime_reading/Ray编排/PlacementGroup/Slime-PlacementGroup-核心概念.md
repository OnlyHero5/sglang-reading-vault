---
title: "PlacementGroup · 核心概念"
type: concept
framework: slime
topic: "PlacementGroup"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-12
---
# PlacementGroup · 核心概念

## 你为什么要读

这篇先建立资源模型。Slime 的 PG 不是“给 actor 一套、给 rollout 一套”的两个池子，而是一个 Ray PlacementGroup 加上多套角色视图。

## 五个对象

| 对象 | 是什么 | 不是 |
|------|--------|------|
| `num_gpus` | 本地 Ray PG 要申请的 bundle 数 | 整个系统的 GPU 总数 |
| `rollout_offset` | rollout 视图在重排后 bundle 列表里的起点 | 物理 GPU id |
| `pg` | Ray PlacementGroup 对象 | Megatron parallel group |
| `reordered_bundle_indices` | logical index 到 Ray 原始 bundle index 的映射 | CUDA ordinal |
| `reordered_gpu_ids` | logical index 到 Ray 探测到的 GPU id | 训练数据分片 id |

源码返回的角色视图是同一种结构：

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

读者抓手：`actor` 和 `rollout` 大多数时候共享同一个 `pg` 对象，只是第二、第三个列表不同。

## 布局矩阵

`_get_placement_group_layout` 只返回两个数：本地申请多少 bundle，以及 rollout 从哪里切。

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

测试把这些场景固定下来：

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

| 场景 | `num_gpus` | `rollout_offset` | 资源含义 |
|------|------------|------------------|----------|
| 普通分离 | actor + rollout | actor | actor 用前段，rollout 用后段 |
| colocate | max(actor, rollout) | 0 | 两侧都从 logical 0 开始；共享长度是 `min(actor, rollout)`，其余是较大一侧独占后缀 |
| external | actor | actor | rollout 视图为空，本地不申请 external GPU |
| debug rollout only | rollout | 0 | 本地只保留 rollout 侧 |
| debug train only | actor | 0 | 本地只保留训练侧 |
| zero rollout non-colocate | actor | actor | rollout 视图为空 |

## 为什么要重排 bundle

Ray PG 的原始 bundle 顺序不一定等于节点和 GPU 的物理顺序。Slime 用一次性 `InfoActor` 探测每个 bundle 上的 `(node_ip, gpu_id)`，再按 IP 和 GPU id 排序。

```python
# 来源：slime/ray/placement_group.py L15-L39
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


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

重排后的列表由 `_create_placement_group` 返回：

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

心理模型：`bundle 0` 是 Slime 的逻辑 0 号座位，不一定是 Ray 原始 bundle 0。真正调度 actor 时要用 `reordered_bundle_indices[rank]`。`PACK` 是 Ray 的放置策略偏好，不是“所有 bundle 必在一台节点”的保证；资源跨节点时仍会形成多节点逻辑序列。

## colocate 是共同前缀，不是两侧全量同构

colocate 时 `rollout_offset=0`，actor 和 rollout 都从同一套 bundle 开始，但各自消费数量不同：actor 只取前 `A` 个 rank，rollout 视图可用前 `R` 个 slot。因此真实重叠是前 `min(A,R)` 个，若 `R>A`，后 `R-A` 个是 rollout-only；若 `A>R`，后 `A-R` 个是 train-only。参数校验会在未显式设置时打开 train/rollout offload。

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

这里的重点不是 Ray 层面有没有重叠，而是哪些 slot 重叠，以及显存生命周期如何切换。当前 `ServerGroup.needs_offload` 是按 group 起点判断：只要 group 起点落在 Megatron 前缀内，整组就标记为需要 offload；它不是逐 GPU 的区间求交。默认单 group 跨过 actor 边界时，rollout-only 后缀也可能随整组一起执行 offload/onload。

## critic 与 actor 也是同席，不是额外申请一套 PG

参数校验把 critic GPU 数强制设为 actor GPU 数，`pgs["critic"]` 又直接复用 actor 视图。训练 actor 和 critic actor 都绑定同一批 logical bundle；每个 Ray actor 当前各申报 0.4 GPU，因此同一 slot 上二者合计 0.8 的调度份额。`use_critic` 还会强制 `offload_train=True`，真正的模型显存所有权仍靠训练侧 wake/sleep 切换，而不是靠 Ray 的 0.4 份额做显存隔离。

```python
# 来源：slime/utils/arguments.py L1856-L1859
args.use_critic = args.advantage_estimator == "ppo"
# Critic always uses the same GPU count as actor.
args.critic_num_gpus_per_node = args.actor_num_gpus_per_node
args.critic_num_nodes = args.actor_num_nodes
```

```python
# 来源：slime/utils/arguments.py L1901-L1902
if args.use_critic:
    args.offload_train = True
```

## external 是本地 PG 不拥有 rollout GPU

external rollout server 已经由外部系统占 GPU。本地 Slime job 保留训练资源，rollout 视图为空切片。

```python
# 来源：slime/ray/placement_group.py L106-L109
if args.rollout_external:
    if args.debug_rollout_only:
        return 0, 0
    return actor_num_gpus, actor_num_gpus
```

下游 rollout 入口也会在 external 场景走外部 server：

```python
# 来源：slime/ray/rollout.py L1089-L1104
def start_rollout_servers(args, pg) -> tuple[dict[str, Any], list[Any]]:
    """Start rollout servers without waiting for final engine initialization.

    Each model defined in the sglang config gets its own router and set
    of server groups.  Server groups within a model may have different
    ``num_gpus_per_engine`` (e.g. for PD disaggregation where prefill
    and decode use different TP sizes).

    Returns ``(servers, init_handles)`` where servers maps model name to
    ``RolloutServer`` and init_handles contains pending ``engine.init`` refs.

    Note: ``init_http_client`` should be called separately before this,
    as the HTTP client is shared across all servers.
    """
    if args.rollout_external:
        return start_external_rollout_servers(args, start_router=_start_router)
```

## Ray 默认环境和 Lock

PlacementGroup 专题还会碰到两个 Ray 工具：

```python
# 来源：slime/ray/utils.py L16-L37
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
]

RAY_DEFAULT_ENV_VARS = {
    # Ray's uvloop integration has caused intermittent async actor issues.
    "RAY_USE_UVLOOP": "0",
}


def add_default_ray_env_vars(env_vars: dict[str, str] | None = None) -> dict[str, str]:
    return RAY_DEFAULT_ENV_VARS | (env_vars or {})


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)
```

`NOSET_VISIBLE_DEVICES_ENV_VARS_LIST` 让 Slime 显式阻止 Ray 重写多类 accelerator visible-devices 环境变量；这也是 0.4/0.2 只能解释为调度 accounting、不能解释为设备切片的原因之一。`RAY_USE_UVLOOP=0` 用于规避 async actor 的间歇性问题。

## 复盘

PlacementGroup 这一层只回答资源座位表问题：

1. 申请多少本地 Ray bundle。
2. 把 Ray 原始 bundle 顺序重排成 Slime logical order。
3. 给 actor、rollout、critic 切出不同视图。
4. 把视图交给 RayTrainGroup 和 RolloutManager。

下一篇 [[Slime-PlacementGroup-源码走读]] 沿 `train.py` 启动顺序读源码。
