---
title: "Ray参数 · 排障指南"
type: troubleshooting
framework: slime
topic: "Ray参数"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Ray参数 · 排障指南

## 你为什么要读

Ray 参数最终会变成 PlacementGroup、bundle 和远程 actor 的资源所有权。本文先追默认值与派生值，再核对 PG 创建和 actor placement；这样能区分“参数看起来不对”和“集群真的把 GPU 放错了位置”。

这篇按症状排障。每个问题都落到一个源码入口，避免把参数行为解释成经验规则。

## 我没传 `--offload-train`，为什么最后变成 True

先看你是否打开了 `--colocate` 或 PPO critic。

```python
# 来源：slime/utils/arguments.py L1885-L1906
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

if args.use_critic:
    args.offload_train = True

if args.offload_train:
    args.disable_grad_buffers_cpu_backup = True
    args.disable_param_buffers_cpu_backup = True
```

判断方式：

- colocate 下，未显式设置的 `offload_train` 和 `offload_rollout` 会默认 True。
- PPO critic 下，`use_critic=True` 会再把 `offload_train` 设为 True。
- 普通 decoupled 且不用 critic 时，未设置的 offload 会落到 False。

## `--offload` 和 `--offload-train` 是什么关系

`--offload` 是便利开关，不是后续运行长期使用的字段。validate 会把它展开成两个字段，然后删除。

```python
# 来源：slime/utils/arguments.py L1861-L1864
if args.offload:
    args.offload_train = True
    args.offload_rollout = True
del args.offload
```

排障时不要在 validate 后找 `args.offload`；应该看 `args.offload_train` 和 `args.offload_rollout`。

## `rollout_num_gpus=0` 到底会不会启动本地 engine

参数 help 明确把 0 定义成“只启动 router，不启动本地 SGLang engine”的选择。

```python
# 来源：slime/utils/arguments.py L44-L54
parser.add_argument(
    "--rollout-num-gpus",
    type=int,
    default=None,
    help=(
        "Number of GPUs for inference. Note that when using --colocate, "
        "i.e. the training and the inference engines are on the same gpus, this param will be set as "
        "actor_num_gpus_per_node * actor_num_nodes unless it is explicitly set. "
        "Set it to 0 to launch routers without local SGLang engines."
    ),
)
```

但 Ray layout 仍要分场景看：

- non-colocate + `rollout_num_gpus=0`：layout 是 `(actor_gpu, actor_gpu)`，actor PG 还在，rollout 切片为空。
- colocate + `rollout_num_gpus=0`：layout 是 `(actor_gpu, 0)`，同样不启动本地 engine。
- debug rollout-only + `rollout_num_gpus=0`：validate 会把 actor GPU 也清零。

对应测试固定了前两个 layout：

```python
# 来源：slime/tests/test_placement_group.py L36-L42
pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
pytest.param({"rollout_external": True}, (16, 16), id="external"),
pytest.param({"rollout_external": True, "debug_rollout_only": True}, (0, 0), id="external_debug_rollout"),
```

## non-colocate 没传 `--rollout-num-gpus` 会默认多少

不要把 colocate 的默认逻辑套到普通 decoupled 上。源码只在 colocate 分支处理 `rollout_num_gpus is None`。

```python
# 来源：slime/utils/arguments.py L1885-L1894
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
```

结论：普通 decoupled 运行应显式传 `--rollout-num-gpus`，或走 external discovery 让它写回数字。否则 placement group 消费时需要整数，会暴露为资源布局问题。

## `debug_rollout_only` 为什么改了 actor GPU

因为 rollout-only 调试不需要训练 actor，但仍要构造一个能承载 rollout server 的 Ray 资源视图。

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

排障提示：如果你看到 actor nodes 被改成 0 或按 rollout GPU 反推，先检查是否打开了 `--debug-rollout-only`。

## 为什么 external 地址会改写 `rollout_num_gpus`

external 模式下，远端 SGLang server 的 GPU 拓扑才是事实。Slime 通过 `/server_info` 获取 `tp_size`、`pp_size`、`num_gpus` 等字段，然后写回本地 args。

```python
# 来源：slime/backends/sglang_utils/external.py L58-L67
def get_server_info(url: str, timeout: float = 30.0) -> dict:
    errors = []
    for endpoint in ("/server_info", "/get_server_info"):
        try:
            response = requests.get(f"{url}{endpoint}", timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError(f"Failed to fetch SGLang server info from {url}: {'; '.join(errors)}")
```

```python
# 来源：slime/backends/sglang_utils/external.py L117-L119
args.rollout_external_engine_infos = [info.to_dict() for info in infos]
args.rollout_num_engines = len(infos)
args.rollout_num_gpus = sum(info.num_gpus for info in infos)
```

验证方式：先直接请求外部 server 的 `/server_info`；再在 `apply_external_engine_info_to_args` 后检查 `args.rollout_num_gpus` 和 `args.rollout_num_engines`。

## 为什么 delta weight sync 不能和 colocate 一起用

源码把这个组合直接拒绝。原因写在异常文本里：colocate 的权重传递走 CUDA IPC handle，delta 的 snapshot、diff、encode 对这个路径是额外开销。

```python
# 来源：slime/utils/arguments.py L1986-L1997
if args.update_weight_mode == "delta":
    if args.update_weight_transport != "disk":
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

如果你要用 delta，走 non-colocate + disk transport，对照 [[Slime-磁盘权重同步]]。

## 为什么 `train_async.py + --colocate` 会失败

参数层不禁止这个组合，但 async 训练入口会 assert。

```python
# 来源：train_async.py L9-L15
# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)
```

结论：如果你想用 `train_async.py`，不要传 `--colocate`。如果你必须同卡运行，走同步主循环或专门支持的 fully-async 方案，不要只改 parser。

## 少于 8 卡的 colocate 为什么要设置 `--num-gpus-per-node`

cluster 参数说明里提醒：colocate 下如果每节点少于 8 卡，要显式设置 rollout 的 `num_gpus_per_node`。

```python
# 来源：slime/utils/arguments.py L61-L69
parser.add_argument(
    "--num-gpus-per-node",
    type=int,
    default=8,
    help=(
        "Number of gpus per node for rollout."
        "Notice: If you are going to use less than 8 gpus per node under colocate mode, you should set this number."
    ),
)
```

这张证据只证明参数帮助明确要求“小于 8 卡/节点的 colocate 要显式设置”；它没有单独证明某一种故障必然发生。排障时同时记录 `actor_num_gpus_per_node`、`num_gpus_per_node`、PG bundle 的实际 node/GPU 映射和每个 ServerGroup 的 engine 布局，再判断是节点粒度、资源切片还是引擎拓扑不一致。

## 该在哪里打断点

按问题选入口：

| 症状 | 断点 |
|------|------|
| 字段值和 CLI 不一致 | `slime_validate_args(args)` 前后 |
| SGLang TP 不对 | `slime/backends/sglang_utils/arguments.py::validate_args` |
| external GPU 数不对 | `apply_external_engine_info_to_args` |
| Ray GPU 申请数不对 | `_get_placement_group_layout` |
| async + colocate 失败 | `train_async.py::train` |

下一篇 [[Slime-Ray参数-学习检查]] 用推导题检查这些边界是否真正掌握。
