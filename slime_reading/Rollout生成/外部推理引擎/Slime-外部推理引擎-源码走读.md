---
title: "外部推理引擎 · 源码走读"
type: walkthrough
framework: slime
topic: "外部推理引擎"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# 外部推理引擎 · 源码走读

这篇追踪一条真实主线：外部系统已经启动了一组 SGLang server，用户把地址传给 Slime，Slime 如何把它们接入 rollout 闭环。

读完后，读者应该能解释三件事：Slime 什么时候访问外部 server；为什么 external 模式不占 rollout GPU；外部 server 既然不归 Slime launch，为什么后续仍能 generate 和 update weights。

---

## 读者任务

| 你遇到的问题 | 本文要帮你定位到 |
|--------------|------------------|
| 传了地址但 Slime 没识别 external | `arguments.py` 的 external 开关与 discovery 触发 |
| `/server_info` 探测失败 | 地址规范化、fallback endpoint、proxy/no_proxy |
| Ray 还在等 rollout GPU | `placement_group.py` 的 external 布局 |
| router 没有 external worker | `start_external_rollout_servers` 与 `_init_external` |
| generate 并发打不上去 | `http_utils.init_http_client` 的 engine 数和连接池 |
| update 权重路径选错 | NCCL、full disk、delta disk 的部署边界 |

---

## 长文读法

这篇按 external rollout 的接入链路读：用户只传外部 SGLang 地址，Slime 先通过 `/server_info` 探测出拓扑，再把结果写回 `args`；Placement Group 因此只给训练侧预留 GPU；RolloutManager 后续创建 zero GPU 的 `SGLangEngine` adapter，由 adapter 做 sanity check、注册 router，并把外部 server 接入 generate / update weights 的数据面。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立 external 全景 | 主线图、1 到 5 | external 的关键不是 launch server，而是把已存在 server 的拓扑变成 Slime 内部参数 |
| 排查地址或 `/server_info` 失败 | 1 到 4 | 地址先规范化，再依次尝试 `/server_info` 和 `/get_server_info`，返回值决定 worker type、GPU 数和 PD bootstrap |
| 判断 GPU 为什么没有给 rollout 预留 | 6 到 7 | external 不能和 Slime-managed 拓扑混用，PG 只覆盖训练侧资源 |
| 排查 router 没有外部 worker | 8 到 10 | zero GPU adapter 不是服务进程，它负责校验外部 server args 并向 router 注册 worker |
| 排查 generate 并发或 proxy 影响 | 11 | HTTP client 并发按 rollout engine 数放大，并明确不走系统代理 |
| 判断 recover 和权重更新边界 | 12 到 13 | 外部进程不归 Slime recover；disk delta 先补丁本地完整 checkpoint，再让 SGLang 普通 reload |
| 评估 external PD 测试能否直接运行 | 13 | 测试拓扑仍有参考价值，但参数与同步注释已漂移，不能当当前执行规范 |

读的时候不要把 external adapter 理解成“又启动了一套 engine”。它更像 Slime 控制面里的代理对象：占 Ray actor 名额，但不占 rollout GPU，也不拥有外部进程生命周期。

---

## 主线图

```mermaid
sequenceDiagram
    participant U as 用户参数
    participant A as arguments.py
    participant D as external.py
    participant P as placement_group.py
    participant R as RolloutManager
    participant E as SGLangEngine adapter
    participant S as 外部 SGLang server
    participant Router as sglang_router

    U->>A: --rollout-external-engine-addrs
    A->>D: apply_external_engine_info_to_args
    D->>S: GET /server_info 或 /get_server_info
    D-->>A: infos, rollout_num_engines, rollout_num_gpus
    A->>P: create_placement_groups
    P-->>A: 只预留训练 GPU
    R->>D: start_external_rollout_servers
    D->>E: ray.remote(SGLangEngine, num_gpus=0)
    E->>S: get_server_info sanity check
    E->>Router: register worker
```

---

## 1. 用户传的是地址，Slime 需要主动发现拓扑

系统压力：外部 server 的 TP、PP、worker type、GPU 数不在 Slime YAML 里。只保存地址不够，后续 HTTP 并发、权重 rank、router PD 注册都需要结构化拓扑。

设计选择：CLI 只接收地址列表；参数收尾阶段把 `rollout_external` 标成真，并在非 train-only 模式下立即访问外部 server。

```python
# 来源：slime/utils/arguments.py L555-L561
parser.add_argument(
    "--rollout-external-engine-addrs",
    type=str,
    default=None,
    nargs="+",
    help="Address and ports of the external engines.",
)
```

```python
# 来源：slime/utils/arguments.py L1851-L1854
args.rollout_external = args.rollout_external_engine_addrs is not None

if args.rollout_external and not args.debug_train_only:
    apply_external_engine_info_to_args(args, logger=logger)
```

执行逻辑：

- `None` 和空列表语义不同：没有传参数时不进入 external；传了但没有地址会在 discovery 层报错。
- `debug_train_only` 跳过 discovery，因为训练调试不需要启动 rollout server。
- discovery 发生在创建 Placement Group 之前，所以 PG 能基于 external 模式决定不预留 rollout GPU。

---

## 2. 地址规范化是第一道输入边界

系统压力：用户可能传 `host:port`，也可能传 `http://host:port/`；IPv6 必须带括号，否则 URL 解析会歧义。

设计选择：`normalize_external_engine_addr` 统一成无尾斜杠的 HTTP base URL，并要求 scheme、hostname、port 都存在。

```python
# 定位骨架（非逐行摘录）：slime/backends/sglang_utils/external.py L32-L44
def normalize_external_engine_addr(addr: str) -> str:
    """Normalize ``host:port`` or ``http://host:port`` to an HTTP base URL."""
    if "://" not in addr:
        addr = f"http://{addr}"
    addr = addr.rstrip("/")
    parsed = urlparse(addr)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError(
            f"Invalid external SGLang engine address {addr!r}. "
            "Use host:port or http://host:port (IPv6 must be bracketed)."
        )
    return addr
```

不变量与失败模式：

- external 只接受 HTTP base URL，不接受 OpenAI API path。
- IPv6 地址要写成 `http://[addr]:port`。
- 地址规范化失败时，Slime 还没开始 Ray 编排。

## 3. `/server_info` 是 external 的拓扑来源

系统压力：SGLang 版本可能暴露 `/server_info` 或 `/get_server_info`，Slime 需要兼容两种 endpoint，但不能在完全失败时继续启动。

设计选择：`get_server_info` 依次尝试两个 endpoint，收集错误，全部失败后抛异常。`timeout` 是每次请求的上限，不是整次 discovery 的总 deadline；两个 endpoint 都卡满时，单地址最坏约等待 `2×timeout`，还未计连接器与调度误差。

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

执行逻辑：

- 先试新旧兼容 endpoint。
- 只返回 JSON dict，不在这里判断 worker type 或 GPU 数。
- 完全失败要 fail fast，因为后续并发、PG 和权重同步都依赖这个拓扑。

运行验证：external server 启动后手动请求 `http://host:port/server_info`，应能看到 `tp_size`、`pp_size`、`disaggregation_mode` 等字段。

## 4. discovery 把 server_info 转成 `ExternalEngineInfo`

系统压力：router 需要 worker type；HTTP client 需要 engine 数；权重同步需要每个 engine 的逻辑 GPU rank 数；PD prefill 还需要 bootstrap port。

设计选择：`discover_external_engines` 保留原始 `server_info`，同时推导 `worker_type`、`num_gpus` 和 `disaggregation_bootstrap_port`。

```python
# 来源：slime/backends/sglang_utils/external.py L70-L104
def _infer_worker_type(server_info: dict) -> str:
    if server_info.get("encoder_only"):
        return "encoder"
    mode = server_info.get("disaggregation_mode")
    if mode in ("prefill", "decode"):
        return mode
    return "regular"


def discover_external_engines(addrs: list[str], timeout: float = 30.0) -> list[ExternalEngineInfo]:
    infos = []
    for addr in addrs:
        url = normalize_external_engine_addr(addr)
        parsed = urlparse(url)
        assert parsed.hostname is not None and parsed.port is not None
        server_info = get_server_info(url, timeout=timeout)

        pp_size = int(server_info.get("pp_size") or server_info.get("pipeline_parallel_size") or 1)
        tp_size = int(server_info.get("tp_size") or server_info.get("tensor_parallel_size") or 1)
        num_gpus = int(server_info.get("num_gpus") or server_info.get("num_gpus_per_engine") or tp_size * pp_size)
        bootstrap_port = server_info.get("disaggregation_bootstrap_port")
        bootstrap_port = int(bootstrap_port) if bootstrap_port is not None else None

        infos.append(
            ExternalEngineInfo(
                url=url,
                host=parsed.hostname,
                port=parsed.port,
                worker_type=_infer_worker_type(server_info),
                num_gpus=num_gpus,
                disaggregation_bootstrap_port=bootstrap_port,
                server_info=server_info,
            )
        )
    return infos
```

测试覆盖了 PD 场景：prefill 和 decode 被分别识别，prefill 的 bootstrap port 被保留。GPU 数的来源必须读得更精确：优先用 server 返回的 `num_gpus` 或 `num_gpus_per_engine`，否则只回退到 `tp_size * pp_size`。这个回退不会乘 `dp_size`，也不根据 EP 修正，因此它只是当前 Slime 的缺省逻辑容量，不足以单独证明外部部署的物理 GPU 总数；复杂并行拓扑应显式返回总 GPU 数。

```python
# 来源：slime/tests/test_external_sglang_engines.py L96-L106
    apply_external_engine_info_to_args(args)

    assert args.rollout_external is True
    assert args.router_pd_disaggregation is False
    assert args.rollout_num_gpus == 6
    assert args.rollout_num_engines == 2
    assert get_rollout_num_engines(args) == 2
    assert [info["worker_type"] for info in args.rollout_external_engine_infos] == ["prefill", "decode"]
    assert [info["num_gpus"] for info in args.rollout_external_engine_infos] == [2, 4]
    assert [info["server_info"]["dp_size"] for info in args.rollout_external_engine_infos] == [1, 2]
    assert args.rollout_external_engine_infos[0]["disaggregation_bootstrap_port"] == 12090
```

发现逻辑还有三个没有替读者兜底的输入契约：

- 地址列表逐项 append，不去重；重复 URL 会重复计 engine/GPU、重复创建 adapter，并放大后续 offset 与权重 rank。
- `_infer_worker_type` 对 `encoder_only` 使用 truthiness。JSON boolean `false` 正常，但字符串 `"false"` 仍会被识别为 encoder。
- 启动 Router 只判断 `any(info.is_pd_worker)`；单独一侧 prefill 或 decode 也会进入 PD 模式，没有成对完整性检查。

读者抓手：PD external 注册失败时，先看 prefill server 的 `server_info` 是否带 `disaggregation_bootstrap_port`，再确认 prefill/decode 两侧都存在、地址无重复、布尔字段真的是 JSON boolean。

## 5. discovery 写回 args，后续模块不再重复探测

系统压力：如果每个模块都自己请求 server_info，会出现时序和一致性问题。Slime 需要一个 single source of truth。

设计选择：`apply_external_engine_info_to_args` 把 infos、engine 数和 GPU 总数写到 `args`。

```python
# 来源：slime/backends/sglang_utils/external.py L107-L131
def apply_external_engine_info_to_args(args, logger=None) -> None:
    """Detect external engines and store the derived topology on ``args``."""
    addrs = args.rollout_external_engine_addrs
    if not addrs:
        raise ValueError("apply_external_engine_info_to_args requires --rollout-external-engine-addrs.")

    infos = discover_external_engines(addrs)
    if not infos:
        raise ValueError("--rollout-external-engine-addrs did not contain any engines.")

    args.rollout_external_engine_infos = [info.to_dict() for info in infos]
    args.rollout_num_engines = len(infos)
    args.rollout_num_gpus = sum(info.num_gpus for info in infos)

    if logger is not None:
        summary = [
            {
                "url": info.url,
                "worker_type": info.worker_type,
                "num_gpus": info.num_gpus,
                "disaggregation_bootstrap_port": info.disaggregation_bootstrap_port,
            }
            for info in infos
        ]
        logger.info(f"Detected external SGLang engines: {summary}")
```

这段解释了为什么 external 模式下可以不传 `--rollout-num-gpus`：它来自外部 server 的发现结果。但所谓 single source of truth 只覆盖 discovery 之后的 `args`；每个 adapter 的 `_init_external` 还会再次请求 server info。两次请求不是事务快照，外部 server 若在启动窗口换配置，有限 sanity check 看到的可能是另一份状态。

## 6. 参数校验禁止 external 与 Slime-managed 拓扑混用

系统压力：`--prefill-num-servers` 和 `--sglang-config` 都意味着 Slime 要管理 server topology；external 地址则意味着拓扑已由外部系统给出。混用会让所有权不清。

设计选择：SGLang 参数校验阶段断言互斥。

```python
# 来源：slime/backends/sglang_utils/arguments.py L162-L173
# Mutual-exclusion checks for PD disaggregation / sglang-config.
assert not (
    getattr(args, "prefill_num_servers", None) is not None and getattr(args, "rollout_external", False)
), "prefill_num_servers cannot be set with --rollout-external-engine-addrs."

assert not (
    getattr(args, "sglang_config", None) is not None and getattr(args, "rollout_external", False)
), "sglang_config cannot be set with --rollout-external-engine-addrs."

assert not (
    getattr(args, "sglang_config", None) is not None and getattr(args, "prefill_num_servers", None) is not None
), "sglang_config and prefill_num_servers are mutually exclusive. Use server_groups in the YAML config instead."
```

读者抓手：需要 frozen reference/reward 或多模型 serving 时，不要试图把 external 地址和 `sglang_config` 拼在一起；应先决定谁拥有 server topology。

## 7. Placement Group 不为 external rollout 预留 GPU

系统压力：external server 已占用它自己的 GPU。如果 Slime 再在训练 Ray PG 里预留 rollout GPU，会浪费资源，甚至阻塞 Ray 调度。

设计选择：`_get_placement_group_layout` 在 external 模式下返回 `(actor_num_gpus, actor_num_gpus)`，让 rollout slice 为空。

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

```python
# 来源：slime/tests/test_placement_group.py L41-L50
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (0, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])
```

执行逻辑：

- 普通 external：PG 只有训练 GPU，rollout offset 等于 actor GPU 数。
- debug rollout only + external：训练也没有 GPU，PG 为空。
- `args.rollout_num_gpus` 仍有逻辑意义，用于并发、metrics 和权重同步 GPU count，不代表 Ray PG 资源。

## 8. RolloutManager 创建 zero GPU adapter，而不是 server 进程

系统压力：后续代码希望拿到 engine actor handle 来调用 pause、flush、update、profile 等控制端点；但 external server 进程已经存在，不能再 launch。

设计选择：`start_external_rollout_servers` 仍创建 `SGLangEngine` actor，但 `num_gpus=0`，并把 external info 转成 `engine.init` kwargs。

```python
# 来源：slime/backends/sglang_utils/external.py L178-L217
def start_external_rollout_servers(args, *, start_router) -> tuple[dict[str, ExternalRolloutServer], list]:
    import ray

    from slime.backends.sglang_utils.sglang_engine import SGLangEngine
    from slime.ray.utils import add_default_ray_env_vars

    infos = external_engine_infos_from_args(args)
    router_ip, router_port = start_router(args, has_pd_disaggregation=any(info.is_pd_worker for info in infos))
    args.sglang_router_ip = router_ip
    args.sglang_router_port = router_port

    engines = []
    engine_gpu_counts = []
    engine_gpu_offsets = []
    init_handles = []
    RolloutRayActor = ray.remote(SGLangEngine)
    gpu_offset = 0
    for rank, info in enumerate(infos):
        rollout_engine = RolloutRayActor.options(
            num_cpus=0.2,
            num_gpus=0,
            runtime_env={"env_vars": add_default_ray_env_vars()},
        ).remote(
            args=args,
            rank=rank,
            worker_type=info.worker_type,
            base_gpu_id=0,
            num_gpus_per_engine=info.num_gpus,
        )
        engines.append(rollout_engine)
        engine_gpu_counts.append(info.num_gpus)
        engine_gpu_offsets.append(gpu_offset)
        gpu_offset += info.num_gpus
        init_handles.append(
            rollout_engine.init.remote(
                **external_engine_init_kwargs(info),
                router_ip=router_ip,
                router_port=router_port,
            )
        )
```

```python
# 来源：slime/backends/sglang_utils/external.py L46-L55
def external_engine_init_kwargs(info: ExternalEngineInfo) -> dict:
    init_kwargs = {
        "dist_init_addr": f"{info.host}:{info.port}",
        "nccl_port": None,
        "host": info.host,
        "port": info.port,
    }
    if info.worker_type == "prefill":
        init_kwargs["disaggregation_bootstrap_port"] = info.disaggregation_bootstrap_port
    return init_kwargs
```

不变量与失败模式：

- actor 的 `base_gpu_id=0` 不表示它会占用外部 server 的 GPU；它只是满足 `SGLangEngine` 构造参数。
- `engine_gpu_counts` 来自 discovery，后续权重同步会使用。
- prefill 的 bootstrap port 必须进入 `init_kwargs`，否则 router 注册无法构建 PD payload。
- 当前是一条 external 地址创建一个 zero-GPU actor，且 `.options(...)` 没有按 external host 设置 node affinity。这个 actor 是控制代理，不天然与外部 server 同机。
- `_compute_server_args` 仍会用 adapter 的枚举 `rank` 和发现出的 GPU 数推导 `nnodes=max(1,num_gpus//num_gpus_per_node)`、`node_rank=rank%nnodes`；而 Router 注册、URL 暴露与多数控制请求在 `node_rank != 0` 时会跳过。例如每个地址报告 16 GPU、`num_gpus_per_node=8` 时，地址 0 得到 node-rank 0，地址 1 得到 node-rank 1；第二个独立 external engine 可能因此不注册。地址序号不是某个 engine 内部的节点序号，当前复用公式缺少这一等价关系的证明。
- `ExternalRolloutServer.update_weights` 固定为 `True`，并且只构造 `default` model；当前 external 入口不能表达一个 frozen external reference/reward model。

## 9. `SGLangEngine._init_external` 是控制面接管点

系统压力：外部 server 可能没有按 rollout 所需的模式启动。Slime 会在接入 router 前做一轮有限 sanity check，但这不是模型身份与完整并行拓扑证明。

设计选择：`_init_external` 重新请求 server info，对 `_compute_server_args` 认为必须检查的字段做一致性断言，然后注册 router。

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L184-L197
def _init_external(self, expect_server_args, external_engine_need_check_fields):
    logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

    def _sanity_check_server_args(actual_server_args, expect_server_args):
        for name in external_engine_need_check_fields:
            expect_value = expect_server_args.get(name)
            actual_value = actual_server_args.get(name)
            assert (
                actual_value == expect_value
            ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

    actual_server_args = get_server_info(f"http://{self.server_host}:{self.server_port}")
    _sanity_check_server_args(actual_server_args, expect_server_args)
    self._register_to_router(expect_server_args)
```

校验字段来自 `_compute_server_args` 的早期字段集合，但有两层收缩：第一，`model_path`、host、port、rank、TP/DP/PP/EP 等被 `_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS` 明确排除；第二，`external_engine_need_check_fields` 在遍历合并一般 `args.sglang_*` 和 YAML overrides **之前**就已经生成。实际重点覆盖 `enable_memory_saver`、worker-specific 的 disaggregation/负载均衡参数，以及按条件加入的 routed-expert 返回和 dtype；它不保证外部 server 的模型路径、并行规模或所有 SGLang 参数与 Slime 期望一致。

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L625-L639
    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    server_arg_fields = dataclasses.fields(ServerArgs)
    server_arg_field_names = {attr.name for attr in server_arg_fields}
    unused_keys = set(kwargs.keys())
    for attr in server_arg_fields:
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)
```

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L670-L690
_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "host",
    "port",
    "nccl_port",
    "nnodes",
    "node_rank",
    "dist_init_addr",
    "gpu_id_step",
    "base_gpu_id",
    "tp_size",
    "dp_size",
    "pp_size",
    "ep_size",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
]
```

读者抓手：sanity check 报 `expect_value` 和 `actual_value` 不一致时，不要改 Slime 代码绕过；先确认外部 server 启动参数是否和训练任务期望一致。反过来，sanity check 通过也只能证明被选中的少数字段一致，不能当成模型与并行拓扑的全量验收。

---

## 10. router 注册让外部 server 进入 generate 数据面

系统压力：rollout 生成请求打的是 router，不是每个 external server 地址。外部 server 发现成功后，还必须注册成 router worker。

设计选择：复用 `SGLangEngine._register_to_router`；regular/PD 走不同 payload，encoder 跳过注册。

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L204-L232
def _register_to_router(self, server_args_dict):
    if self.worker_type == "encoder":
        return

    if self.node_rank == 0 and self.router_ip and self.router_port:
        worker_url = f"http://{self.server_host}:{self.server_port}"
        if parse(sglang_router.__version__) <= parse("0.2.1"):
            assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
            )
        else:
            payload = {
                "url": worker_url,
                "worker_type": self.worker_type,
            }
            if self.worker_type == "prefill":
                bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")
                if bootstrap_port is None:
                    raise RuntimeError(
                        f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                        "cannot register it to the PD router."
                    )
                payload["bootstrap_port"] = bootstrap_port
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/workers",
                json=payload,
            )
        response.raise_for_status()
```

运行验证：启动后查 router `/workers`，必须把“预期 URL 集合”与“实际 URL 集合”逐项比较，而不是只确认至少有一个 worker。尤其要测试第二个多节点 external 地址；若 discovery 日志有它、Router 没有它，再检查 adapter 的 `node_rank`。

---

## 11. HTTP client 并发按 external engine 数扩展

系统压力：external server 数来自 discovery，不来自 `rollout_num_gpus / rollout_num_gpus_per_engine` 的默认公式。HTTP client 如果仍按默认公式估算，可能低估连接数。

设计选择：`get_rollout_num_engines` 优先读取 `args.rollout_num_engines`；`init_http_client` 用它乘以 `sglang_server_concurrency` 配置连接池。

```python
# 来源：slime/utils/http_utils.py L201-L226
def get_rollout_num_engines(args) -> int:
    """Return the number of rollout HTTP engines behind the router."""
    if (num_engines := getattr(args, "rollout_num_engines", None)) is not None:
        return int(num_engines)

    rollout_num_gpus = getattr(args, "rollout_num_gpus", None) or 0
    rollout_num_gpus_per_engine = getattr(args, "rollout_num_gpus_per_engine", None) or 1
    if rollout_num_gpus <= 0:
        return 0
    return max(1, rollout_num_gpus // rollout_num_gpus_per_engine)


def init_http_client(args):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    num_engines = get_rollout_num_engines(args)
    if num_engines <= 0:
        return

    _client_concurrency = args.sglang_server_concurrency * num_engines
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
            trust_env=False,  # internal SGLang comm only — never route through system proxy
        )
```

`trust_env=False` 是 external 部署常见问题的保护：内部 SGLang 通信不应被系统 HTTP proxy 劫持。

---

## 12. 外部进程不归 Slime recover

系统压力：Slime 可以杀掉自己启动的 SGLang 子进程并重建 actor，但 external server 由外部系统管理。Slime 不应该假装能 recover。

设计选择：`ExternalRolloutServer` 的 recover/offload/onload 都是 warning 或空操作；`SGLangEngine.shutdown` 在 external 模式下直接返回。

```python
# 来源：slime/backends/sglang_utils/external.py L135-L159
class ExternalRolloutServer:
    """Rollout server backed by pre-launched external SGLang engines."""

    engines: list
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    num_new_engines: int = 0
    server_groups: list = dataclasses.field(default_factory=list)

    @property
    def all_engines(self):
        return self.engines

    def recover(self):
        logger.warning("Fault tolerance is not supported for external rollout engines; skip recover.")

    def offload(self):
        return []

    def onload(self, tags: list[str] | None = None):
        return []
```

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L329-L331
def shutdown(self):
    if self.args.rollout_external:
        return
```

排障结论：external server 掉线时，Slime 的 Ray actor 和 router 可能仍存在，但真正恢复必须由外部部署系统完成，然后重新注册或重启训练侧控制面。`shutdown` 的 early return 不只是“不会 kill 外部进程”，还绕过了后面的 Router remove-worker 逻辑；训练侧 detach 后，旧 Router 可能继续保留 worker。当前安全做法是一起重建 Router，或由外围显式协调注销，不能把 `shutdown` 当作完整 detach。

---

## 13. Disk delta 的真实边界：SGLang 读完整 checkpoint，不直接读 delta

这一节必须把“传输格式”和“引擎加载格式”分开：

- trainer 发布的是按版本组织的稀疏 delta，目的是减少跨文件系统传输量。
- 每个 rollout 主机先把 delta 按顺序应用到一份完整的本地 HF checkpoint。
- SGLang 最后收到的是普通 `update_weights_from_disk(model_path=<完整本地目录>)`；当前 updater 没有传 `load_format="delta"`，也没有传 `files`。

updater 的类注释直接声明了这条边界：

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L30-L36
class UpdateWeightFromDiskDelta(UpdateWeightFromDistributed):
    """
    Delta weight sync over a shared filesystem. PP-src ranks diff each gathered HF tensor against
    a CPU snapshot of the previous sync and publish the changes as a canonical HF checkpoint dir;
    every rollout host applies the delta into its local checkpoint and reloads via the ordinary
    update_weights_from_disk path, so sglang needs no delta support.
    """
```

### 一次同步实际经历五步

```mermaid
sequenceDiagram
    participant T as Trainer ranks
    participant Shared as update_weight_disk_dir
    participant A as 每主机 SGLangEngine actor
    participant Local as local full HF checkpoint
    participant S as External SGLang server

    T->>T: gather HF tensor，和 CPU snapshot 求差
    T->>Shared: 发布 weight_vNNNNNN delta + index
    T->>A: sync_local_checkpoint(version)
    A->>Local: 按版本应用 delta，校验 checksum
    T->>S: update_weights_from_disk(model_path=Local)
    S-->>T: 普通完整 checkpoint reload 完成
```

真正决定调用形态的是 `_reload_engines`：先等待 `all_engine_actors` 完成本地补丁，再只对 `rollout_engines` 发完整目录 reload。对 Slime-managed 多节点 engine，这两个集合可分别表达“每主机 actor”和“node-0 actor”；对 external，`ExternalRolloutServer.all_engines` 只是同一份“一地址一 adapter”列表，不能自动推出每个外部主机都执行了补丁。

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L60-L73
    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        all_engine_actors: Sequence[ActorHandle] | None = None,
    ) -> None:
        # The local checkpoint is host-local, so every host applies its own copy:
        # all_engine_actors is one actor per host, vs rollout_engines (node 0 only). The
        # rollout_engine_lock the NCCL path uses isn't needed — a per-host flock serializes applies.
        self.rollout_engines = rollout_engines
        self.all_engine_actors = list(all_engine_actors or rollout_engines)
        self._is_pp_src_rank = (
```

这段注释表达的是 updater 对调用方的前提，不是 external construction 已经满足该前提的证明；external 的 `all_engines` 仍要回到上一节的一地址一 actor 构造核对。

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L169-L186
    def _reload_engines(self) -> None:
        """Commit the published files, have each host apply the delta, then reload the engines."""
        if self._commit_hook is not None:
            self._commit_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([actor.sync_local_checkpoint.remote(self.weight_version) for actor in self.all_engine_actors])
            ray.get(
                [
                    engine.update_weights_from_disk.remote(
                        model_path=self.args.update_weight_local_checkpoint_dir,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self.rollout_engines
                ]
            )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
```

### Adapter 在主机侧维护完整 checkpoint

`sync_local_checkpoint` 先幂等物化基础 checkpoint，再按版本应用共享目录里的 delta：

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L396-L413
    def sync_local_checkpoint(self, target_version: int):
        """Apply the published deltas into this host's local checkpoint up to target_version; the
        engine reloads it afterwards. Assumes this actor shares the checkpoint filesystem with the
        sglang it drives (true for slime-launched engines)."""
        from slime.utils.disk_delta import apply_deltas, init_local_checkpoint

        init_local_checkpoint(self.args.update_weight_local_checkpoint_dir, self.args.hf_checkpoint)  # idempotent
        # non-POSIX filesystems lack cross-host read-after-write consistency, so the trainer's
        # just-written delta isn't visible on this mount until the hook refreshes it.
        if self.args.custom_delta_pre_read_path:
            from slime.utils.misc import load_function

            load_function(self.args.custom_delta_pre_read_path)(self.args.update_weight_disk_dir, target_version)
        apply_deltas(
            self.args.update_weight_local_checkpoint_dir,
            self.args.update_weight_disk_dir,
            target_version,
        )
```

`apply_deltas` 还维护版本不变量：本地 checkpoint 必须从已应用版本逐个推进，跳版本或基线不匹配会直接失败，而不是带着错误权重继续 serving。

```python
# 来源：slime/utils/disk_delta.py L255-L264
def apply_deltas(local_ckpt_dir: str, delta_root: str, target_version: int) -> None:
    """Apply the delta chain in order to bring the local checkpoint up to target_version, in place.
    A per-tensor checksum guards every write and any mismatch raises (fail loud, never serve bad
    weights). Serialized per host by the lock (co-located actors collapse to one apply)."""
    with _apply_lock(local_ckpt_dir):
        applied = _read_applied_version(local_ckpt_dir)
        if applied is None:
            raise RuntimeError("local checkpoint not materialized")
        for version in range(int(applied) + 1, target_version + 1):
            _apply_version(local_ckpt_dir, os.path.join(delta_root, f"weight_v{version:06d}"))
```

### External 部署多了一个必须显式验证的前提

Slime-managed engine 的 adapter 与它启动的 SGLang 天然共享本机文件系统；external engine 没有这个天然保证。要让 disk delta 成立，必须同时满足：

1. trainer 和执行 `sync_local_checkpoint` 的 actors 都能读到 `update_weight_disk_dir` 中刚发布的版本。
2. `update_weight_local_checkpoint_dir` 对负责补丁的 actor 和对应 external SGLang 进程表示同一份完整 checkpoint。当前 external adapter 没有 external-host node affinity，因此“同机 NVMe”不是代码自动保证的部署形态；除非外围编排能证明 actor 落在正确主机，否则应使用双方可见、同路径同内容的共享挂载或自定义同步/放置层。
3. 非 POSIX 或弱一致性文件系统需要正确实现 `custom_delta_pre_push_path` / `custom_delta_pre_read_path`，保证版本可见性。

只有 external 地址而没有这层文件系统关系，delta 目录即使写成功，SGLang 也可能从另一台机器上的同名空路径 reload。尤其是一个地址代表多节点 external engine 时，当前 construction 没有“一外部主机一补丁 actor”的证据，不能宣称 host-local checkpoint 已在所有节点推进到同一版本。

### 当前 external PD E2E 文件已经漂移

`tests/test_qwen3_4B_external_pd.py` 仍有两个值得保留的拓扑提示：训练 GPU 与 external prefill/decode GPU 分开，engine 数和 GPU 数从 `/server_info` 推导。但它不能作为当前 disk-delta 的可执行规范：

- 文件注释仍声称 external engine 直接调用 `update_weights_from_disk(load_format=delta)`，与上面的 updater 实现冲突。
- 它没有传当前参数校验要求的 `--update-weight-local-checkpoint-dir`。
- 它还传入源码参数定义中不存在的 `--update-weight-encoding deltas` 和 `--update-weight-delta-keep-files`。

因此，读者可以借它理解 external PD 的进程拓扑，不能直接复制其中的 delta 参数。判断同步语义时以 `UpdateWeightFromDiskDelta._reload_engines` 为准。

---

## 运行验证

1. 启动外部 SGLang 后请求 `/server_info`，确认 `tp_size`、`pp_size`、`disaggregation_mode`、prefill bootstrap port。
2. 启动 Slime 后搜索日志 `Detected external SGLang engines`，确认 `rollout_num_engines` 和 `rollout_num_gpus`。
3. 查看 Placement Group 日志，external 普通训练只应创建训练 GPU 数量的 PG。
4. 请求 router `/workers`，确认 external URL 已注册。
5. 若使用 disk delta，同时确认共享发布目录 `update_weight_disk_dir` 和完整本地目录 `update_weight_local_checkpoint_dir` 的可见性；后者必须是 external SGLang 实际能够 reload 的路径。
6. 若请求受 proxy 影响，设置 `no_proxy/NO_PROXY` 覆盖 external host，E2E 测试也这样做。
7. 静态核对当前同步调用：`rg -n "sync_local_checkpoint|model_path=self.args.update_weight_local_checkpoint_dir" slime/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py`，预期不出现 `load_format="delta"`。
8. 运行 `python -m pytest -q slime/tests/test_external_sglang_engines.py slime/tests/test_placement_group.py`。完整 Slime 环境应通过；只做文档审计的轻量环境若缺少 `httpx` 或 `ray`，会在 collection 阶段失败，此时运行 `python -m py_compile slime/slime/backends/sglang_utils/external.py slime/slime/backends/sglang_utils/arguments.py slime/slime/ray/placement_group.py slime/slime/backends/sglang_utils/sglang_engine.py slime/slime/utils/http_utils.py slime/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py slime/slime/utils/disk_delta.py slime/tests/test_external_sglang_engines.py slime/tests/test_placement_group.py` 作为静态替代，并把缺失依赖与代码失败分开记录。

---

## 复盘

1. external 模式的主线从 `/server_info` 开始，不从 Ray GPU 分配开始。
2. `rollout_num_gpus` 是发现出的逻辑容量，PG 是否占 GPU要看 `placement_group.py`。
3. zero GPU `SGLangEngine` actor 让 Slime 复用控制面接口，但不拥有外部 server 进程。
4. router 注册是 generate 数据面生效的关键动作。
5. Delta 只是发布和主机补丁格式；SGLang 看到的仍是完整 checkpoint。External 部署必须额外证明 adapter 与 server 共享正确的 checkpoint 语义，并证明所有需要读取该 checkpoint 的外部主机都完成了同一版本补丁。
6. discovery 成功只证明地址被探测；重复地址、单侧 PD、错误布尔 schema 与 `rank→node_rank` 都可能让运行拓扑仍不闭合。
7. external shutdown 保留外部进程所有权，但当前也不自动从 Router 注销 worker。
