---
title: "外部推理引擎 · 数据流"
type: dataflow
framework: slime
topic: "外部推理引擎"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/dataflow
  - source-reading
updated: 2026-07-12
---
# 外部推理引擎 · 数据流

## 你为什么要读

这篇只看数据和对象穿过哪些边界。external 模式里最容易混淆的是：`rollout_num_gpus` 是发现出的逻辑容量，Ray PG 不预留这些 GPU；`SGLangEngine` actor 是控制面代理，不是外部 server 进程；generate 请求打 router，权重数据走 NCCL 或磁盘。

---

## 长文读法

| 你的问题 | 直接阅读 |
|----------|----------|
| 地址如何变成 Slime 拓扑 | 发现流、资源流 |
| 为什么有 actor 却不占 rollout GPU | 控制流、router 流 |
| rollout 请求究竟打到哪里 | generate 流、网络流 |
| external 如何同步权重 | 权重流；disk-delta 用户再对照 [[Slime-外部推理引擎-源码走读]] 第 13 节 |
| 外部进程掉线谁负责 | 健康与恢复流 |

---

## 四条数据流

```mermaid
flowchart TB
    subgraph Discovery["发现流"]
        Addr["地址列表"] --> Info["/server_info"] --> Args["args.rollout_external_engine_infos"]
    end

    subgraph Control["控制流"]
        Adapter["SGLangEngine actor<br/>num_gpus=0"] --> Check["sanity check"]
        Adapter --> Register["router worker registration"]
    end

    subgraph Generate["请求流"]
        Client["http_utils.post"] --> Router["sglang_router"] --> Server["外部 SGLang server"]
    end

    subgraph Weights["权重流"]
        Updater["Megatron updater"] --> Meta["Ray/HTTP metadata"]
        Updater --> Data["NCCL 或 filesystem"]
        Meta --> Server
        Data --> Server
    end

    Args --> Adapter
```

---

## 发现流：从地址到拓扑事实

| 输入 | 转换 | 输出 |
|------|------|------|
| `host:port` 或 `http://host:port` | `normalize_external_engine_addr` | HTTP base URL |
| base URL | `get_server_info` | 原始 server info dict |
| server info | `_infer_worker_type`；GPU 数优先取显式字段，否则回退 `tp_size * pp_size` | `ExternalEngineInfo` |
| infos | `apply_external_engine_info_to_args` | `rollout_num_engines`、`rollout_num_gpus`、`rollout_external_engine_infos` |

这个流的源码入口是 `apply_external_engine_info_to_args`。

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

不变量：

- 后续模块读 `args.rollout_external_engine_infos`，不应再次猜测拓扑。
- `rollout_num_gpus` 不是 PG 资源申请量，而是 external fleet 的逻辑 serving 容量。
- 回退公式不乘 `dp_size`；没有显式 `num_gpus`/`num_gpus_per_engine` 时，这个值不是物理 GPU 总数证明。
- 输入地址不去重；重复项会同时放大 `rollout_num_engines`、`rollout_num_gpus`、offset 和 adapter 数。
- `encoder_only` 必须是 JSON boolean；字符串 `"false"` 在当前 truthiness 判断中仍会变成 encoder。
- server_info 不可达时应在启动早期失败。

---

## 资源流：Ray PG 不拥有 rollout GPU

external 的 PG 布局由 `rollout_external` 分支决定。普通 external 只需要训练 actor 的 GPU；纯 external rollout 调试甚至可以创建空 PG。

```python
# 来源：slime/ray/placement_group.py L100-L125
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


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    num_gpus, rollout_offset = _get_placement_group_layout(args)

    logger.info(f"Creating placement group with {num_gpus} GPUs...")
```

测试把两种 external 资源语义固定下来。

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

排障时，如果 Ray 在等 rollout GPU，说明你看的不是 external PG 分支，或参数还没正确设置 `rollout_external`。

---

## 控制流：zero GPU actor 接管外部 server

`start_external_rollout_servers` 创建的是控制面 actor。它不占 GPU，只保存每个 external engine 的逻辑 GPU 数和 offset，再调用 `engine.init.remote` 完成 sanity check 与 router 注册。

```python
# 来源：slime/backends/sglang_utils/external.py L184-L217
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

这条流里有两个对象容易混：

- `engine_gpu_counts` 用于权重同步和逻辑容量。
- `num_gpus=0` 是 Ray actor 资源申请，不是 external server 的实际 GPU 数。
- `.options(...)` 没有按 external host 设置 node affinity：adapter 是一地址一个控制代理，不天然与 server 同机，也不天然覆盖一个多节点 engine 的每台主机。

这里还混入了一套不属于 external 地址列表的坐标变换：adapter 的枚举 `rank` 会经 `_compute_server_args` 变成 `node_rank=rank%nnodes`。Router 注册和控制请求只由 node-rank 0 执行。若每个地址都代表一个独立的跨节点 engine，地址 1 却可能因 `rank=1` 被当成地址 0 的“第二节点”，最终 discovery 有记录、Ray actor 也存在，但该 URL 没进 Router、控制 POST 也直接返回。external 接入必须单独核对 `地址→adapter rank→node_rank→Router URL` 四列，不能只核对 actor 数量。

`engine.init` 的 sanity check 也不是全量配置验收：模型路径、TP/DP/PP/EP 等在 external skip list 中，一般 `args.sglang_*` 参数又是在 check-field 集合生成后才合并。它通过时，只能说明被挑中的 worker-mode、memory saver、dtype/routing 等字段一致。

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

---

## router 流：PD worker 由 discovery 驱动

external server 的 `worker_type` 来自 `server_info`。只要任一 worker 是 prefill 或 decode，Slime 启动 router 时就启用 PD 模式；这里是 `any`，不是“prefill 与 decode 都存在”的完整性校验。单侧 PD 会让 Router 进入 PD 模式，却未形成可工作的两侧拓扑。

```python
# 来源：slime/backends/sglang_utils/external.py L24-L29
@property
def is_pd_worker(self) -> bool:
    return self.worker_type in ("prefill", "decode")

def to_dict(self) -> dict:
    return dataclasses.asdict(self)
```

```python
# 来源：slime/ray/rollout.py L1048-L1056
if has_pd_disaggregation:
    router_args.pd_disaggregation = True
    # Disable circuit breaker to prevent RDMA transfer timeouts from
    # marking decode workers as dead. Timeouts are transient (PCIe
    # contention under high load) and do not indicate a dead server.
    router_args.disable_circuit_breaker = True

# We will not use the health check from router.
router_args.disable_health_check = True
```

prefill worker 注册还要依赖 discovery 得到的 bootstrap port。

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

---

## generate 流：请求走 router，不走 actor forward

external 模式下，`SGLangEngine` actor 不是 generate 数据面。rollout 函数通过 `http_utils.post` 发 HTTP 请求，目标通常是 router。

```python
# 来源：slime/utils/http_utils.py L165-L198
async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output
```

HTTP client 的连接池按 discovery 得到的 engine 数扩容。

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

---

## 权重流：metadata 仍经 adapter，数据通道按部署选

| 路径 | 控制面 | 数据通道 | external 部署含义 |
|------|--------|----------|-------------------|
| full + nccl | actor POST update metadata | NCCL group | 训练和 serving 必须网络/NCCL 互通 |
| full + disk | actor POST checkpoint path | 共享文件系统 | 最简单兜底，写完整 HF checkpoint |
| delta + disk | 先补丁完整 checkpoint，再 POST 目录路径 | 共享 delta 目录 + adapter 可写、server 可读的完整 checkpoint | 只有路径、内容可见性和每台 serving 主机的版本推进都被外围部署证明时才成立 |

Delta 模式的两个目录承担不同职责：`update_weight_disk_dir` 是 trainer 发布版本流的共享目录，`update_weight_local_checkpoint_dir` 是最终交给 SGLang reload 的完整 checkpoint。参数校验要求两者都存在，但参数名中的 `local` 不会自动建立主机共置关系：external zero-GPU adapter 没有 external-host node affinity，`ExternalRolloutServer.all_engines` 也只是一地址一 actor。因此 host-local NVMe、多节点 external engine 和同路径异内容挂载都必须由外围编排额外证明。

```python
# 来源：slime/utils/arguments.py L1980-L2002
    # disk-backed sync (full or delta) writes on the trainer and reads on the engines: needs a shared dir
    if args.update_weight_transport == "disk" and not args.update_weight_disk_dir:
        raise ValueError(
            "--update-weight-transport=disk requires --update-weight-disk-dir to point at "
            "a filesystem shared between the trainer and the rollout engines."
        )
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
        if not args.update_weight_local_checkpoint_dir:
            raise ValueError(
                "--update-weight-mode=delta requires --update-weight-local-checkpoint-dir "
                "(a rollout-host-local NVMe directory)."
            )
```

当前 updater 随后先调用 `sync_local_checkpoint`，再把本地完整目录作为 `model_path` 交给普通 reload：

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L174-L184
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
```

这段没有 `load_format="delta"` 或 `files=...`。它遍历的 `all_engine_actors` 在 updater 注释里被期望为“一主机一 actor”，但 external construction 没有自动满足这个前提。仓库里的 external PD E2E 文件仍保留旧注释和失效参数，只能作为进程拓扑样例，不能作为当前同步契约。

---

## 健康与恢复流：external 是空操作边界

`ExternalRolloutServer` 不填 server groups，所以即使打开 `use_fault_tolerance`，`RolloutManager` 的 `for group in srv.server_groups` 也不会为它创建 `RolloutHealthMonitor`；recover/offload/onload 仍是 no-op 或 warning。与此同时 `_start_router` 对所有模式都关闭 router health check，因此 external 的持续进程健康必须由外部系统承担。

```python
# 来源：slime/backends/sglang_utils/external.py L219-L231
    args.sglang_model_routers = {"default": (router_ip, router_port)}
    servers = {
        "default": ExternalRolloutServer(
            engines=engines,
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
            router_ip=router_ip,
            router_port=router_port,
            model_name="default",
            update_weights=True,
            num_new_engines=len(engines),
        )
    }
```

```python
# 来源：slime/ray/rollout.py L464-L470
        self._health_monitors = []
        if not self.args.debug_train_only and self.args.use_fault_tolerance:
            for srv in self.servers.values():
                for group in srv.server_groups:
                    monitor = RolloutHealthMonitor(group, args)
                    monitor.start()
                    self._health_monitors.append(monitor)
```

```python
# 来源：slime/backends/sglang_utils/external.py L152-L165
def recover(self):
    logger.warning("Fault tolerance is not supported for external rollout engines; skip recover.")

def offload(self):
    return []

def onload(self, tags: list[str] | None = None):
    return []

def onload_weights(self):
    return []

def onload_kv(self):
    return []
```

健康检查类仍然存在，但它服务的是 Slime-owned server group。

```python
# 来源：slime/utils/health_monitor.py L145-L158
def _check_engine_health(self, rollout_engine_id, engine) -> None:
    if engine is None:
        logger.info(f"Skipping health check for engine {rollout_engine_id} (None)")
        return

    try:
        ray.get(engine.health_generate.remote(timeout=self._check_timeout))
    except Exception as e:
        logger.error(
            f"Health check failed for rollout engine {rollout_engine_id} (ray timeout or error). Killing actor. Exception: {e}"
        )
        self._kill_engine(rollout_engine_id=rollout_engine_id)
    else:
        logger.debug(f"Health check passed for rollout engine {rollout_engine_id}")
```

排障结论：external server 的进程健康要接外部监控；Slime 的 HTTP retry 只能缓冲短暂抖动，不能重建外部进程或重新形成正确的多机 checkpoint 状态。external `shutdown` 又在 Router 注销逻辑之前早退，因此 detach 不会自动 remove worker；旧 Router 是否继续持有 URL 也属于外围生命周期契约。

---

## 网络流：proxy 和 host 地址

external server 与训练 job 常跨节点或跨集群，proxy 环境很容易劫持内部 HTTP。E2E 测试显式给 external host 加 `no_proxy`。

```python
# 来源：slime/tests/test_qwen3_4B_external_pd.py L355-L364
U.execute_train(
    train_args=train_args,
    num_gpus_per_node=NUM_TRAIN_GPUS,
    megatron_model_type=MODEL_TYPE,
    before_ray_job_submit=launch_external_engines,
    extra_env_vars={
        "no_proxy": f"127.0.0.1,localhost,{external_host}",
        "NO_PROXY": f"127.0.0.1,localhost,{external_host}",
    },
)
```

Slime 侧 `httpx.AsyncClient` 也禁用了 `trust_env`，用于内部 SGLang 通信。

---

## 运行验证

从知识库根目录执行：

```powershell
rg -n "sync_local_checkpoint|update_weight_local_checkpoint_dir" slime/slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py
rg -n "update-weight-local-checkpoint-dir|requires --update-weight-local-checkpoint-dir" slime/slime/utils/arguments.py
rg -n "update-weight-encoding|update-weight-delta-keep-files" slime/tests/test_qwen3_4B_external_pd.py slime/slime
```

预期：

- 第一条命中“先本地补丁、再从完整目录 reload”。
- 第二条命中 delta 模式的必需参数定义和校验。
- 第三条只在 E2E 测试命中两个旧参数，不应在当前参数定义中命中；这证明测试文件已经漂移，不能直接复制。

---

## 数据流复盘

1. discovery 产物是 external 模式的拓扑事实，后续模块只消费它。
2. Ray PG 资源流和 external serving 容量是两套账。
3. zero GPU adapter 让 Slime 复用 engine 控制接口，但不拥有 server 进程。
4. generate 数据面走 router；权重数据通道根据 NCCL 或 disk 单独选择。
5. Delta disk 的 wire format 是 delta，但 SGLang 的 load input 仍是补丁后的完整 checkpoint。
6. recover/offload/onload 在 external server 上没有 Slime-owned 实现。
7. 一地址一 adapter 不等于一主机一补丁 actor；external 多机 disk-delta 必须单独证明 actor 放置与文件系统可见性。
8. discovery 列表不是已验证拓扑：重复地址、单侧 PD、字符串布尔值与 `rank→node_rank` 都可能让计数正确而数据面不完整。
9. external shutdown 不杀外部进程，也不自动从 Router 注销 worker。
