---
title: "SGLang-Engine · 排障指南"
type: troubleshooting
framework: slime
topic: "SGLang-Engine"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# SGLang-Engine · 排障指南

这篇按症状组织。遇到 engine、router、flush、权重同步问题时，先从现象定位到边界，再去看对应源码入口。

---

## 快速排障表

| 症状 | 可能边界 | 源码入口 | 验证方法 |
|------|----------|----------|----------|
| Ray actor 已创建但没有可用 HTTP server | 本地 SGLang 子进程启动或 health 卡住 | `launch_server_process`、`_wait_server_healthy` | 查 `Launch HttpServerEngineAdapter` 后是否有子进程退出异常 |
| router `/workers` 缺少 worker | router 注册失败或 node 0 判断错误 | `_register_to_router` | 请求 router `/workers`，对照 `worker_type` 和 URL |
| PD prefill 注册失败 | 缺 `disaggregation_bootstrap_port` | `_allocate_rollout_engine_addr_and_ports_normal`、`_register_to_router` | 查看 prefill actor init 参数是否包含 bootstrap port |
| update 权重前 `flush_cache` 超时 | 还有 pending request 或 router 未停止派发 | `pause_generation`、`flush_cache`、`abort_servers_until_idle` | 查 `/v1/loads?include=core` 中 request 数 |
| distributed update hang | 权重 NCCL rank 或 broadcast 顺序不一致 | `connect_rollout_engines_from_distributed`、`rollout_engine_lock` | 对照 `engine_gpu_counts` 与 `world_size` |
| external engine 不符合预期 | 外部拓扑/overrides 可能根本不在有限 sanity check 内 | `discover_external_engines`、`_compute_server_args`、`_init_external` | 请求 server info 后按实际 check-list 与跳过字段分层比对 |
| update 后像旧权重 | version 未更新或新 engine 未重连 | `get_updatable_engines_and_lock`、`get_weight_version` | 对比 engine version 与 updater version |

---

## Q1：为什么只有 node 0 actor 发 HTTP？

在 managed 多节点模式中，Slime 把通用 HTTP 控制面收敛到 node 0。`ServerGroup.engines` 只返回每个 engine 的 node 0 actor，`_make_request` 在非 node 0 直接返回。external 模式是一公开地址一 adapter，不能用这组切片推导外部内部节点。

```python
# 来源：slime/ray/rollout.py L133-L135
def engines(self):
    """Node-0 engines only (for multi-node serving)."""
    return self.all_engines[:: self.nodes_per_engine]
```

```python
# 定位骨架（据 `slime/backends/sglang_utils/sglang_engine.py` L234-L245 删节）：
def _make_request(self, endpoint: str, payload: dict | None = None):
    if self.node_rank != 0:
        return

    url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
    response = requests.post(url, json=payload or {})
```

排障判断：

- 如果是普通 HTTP 端点，只查 node 0 actor。
- 如果是 disk delta 本地 checkpoint，要查 `all_engine_actors` 是否覆盖每个 host。
- 如果是 SGLang 内部 TP/PP 通信，node 0 actor 不代表全部 GPU 参与者。

---

## Q2：`nccl_port` 和权重 update 的 `master_port` 是同一个吗？

不是。`nccl_port` 是 SGLang server 启动时的推理并行通信端口；`master_port` 是训练 rank 0 创建权重更新 NCCL group 时临时绑定的端口。

```python
# 来源：slime/ray/rollout.py L991-L994
addr_and_ports.setdefault(current_rank, {})
addr_and_ports[current_rank]["host"] = get_addr()
addr_and_ports[current_rank]["port"] = get_port()
addr_and_ports[current_rank]["nccl_port"] = get_port()
```

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L284-L288
master_address = ray._private.services.get_node_ip_address()
with socket.socket() as sock:
    sock.bind(("", 0))
    master_port = sock.getsockname()[1]
world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0
```

排障判断：看到 NCCL hang 时，先确认日志是在 SGLang server 初始化阶段还是 weight update 阶段。两个阶段的端口、rank、参与者都不同。

---

## Q3：为什么 update 权重前必须 pause + flush？

因为权重 reload 和正在 decode 的请求共享模型权重与 KV cache 语义。Slime 先暂停新 generation，再清空 cache，最后才开始同步权重。

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L109-L120
if dist.get_rank() == 0:
    ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

    # int4/fp4 pre_process
    if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
        post_process_weights(
            restore_weights_before_load=True,
            post_process_quantization=False,
            rollout_engines=self.rollout_engines,
        )
dist.barrier(group=get_gloo_group())
```

`flush_cache` 非 200 会重试，60 轮仍失败后抛 `TimeoutError`。但 `requests.get` 没有 timeout，所以这不是 60 秒硬上限；单轮连接永久挂住时，循环本身也无法推进。

```python
# 定位骨架（据 `slime/backends/sglang_utils/sglang_engine.py` L303-L322 删节）：
def flush_cache(self):
    if self.node_rank != 0:
        return
    # flush cache will not return status_code 200 when there are pending requests
    for _ in range(60):
        try:
            response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
            if response.status_code == 200:
                break
            logger.info(f"Error flushing cache: HTTP {response.status_code} {response.text!r}")
            time.sleep(1)
        except NewConnectionError as e:
            raise e
        except Exception as e:
            logger.info(f"Error flushing cache: {e}")
            time.sleep(1)
            continue
    else:
        raise TimeoutError("Timeout while flushing cache.")
```

验证方法：

- update 前请求 `/v1/loads?include=core`，预期 request 数降到 0。
- 如果 request 数不降，先用 `abort_servers_until_idle` 清空，再排查 router 是否还在派发。

---

## Q4：`abort_servers_until_idle` 和 `flush_cache` 怎么分工？

`flush_cache` 是 SGLangEngine 的同步控制端点，适合标准 update 流程；`abort_servers_until_idle` 是异步清流量工具，适合已有请求无法自然结束的场景。

```python
# 来源：slime/backends/sglang_utils/server_control.py L12-L29
def num_requests_from_load(load: Any) -> int:
    if isinstance(load, list):
        return sum(num_requests_from_load(item) for item in load)

    if not isinstance(load, dict):
        return 0

    if "loads" in load:
        return num_requests_from_load(load["loads"])

    for key in ("num_reqs", "num_total_reqs", "total_reqs"):
        value = load.get(key)
        if isinstance(value, int):
            return value

    running = load.get("num_running_reqs", load.get("total_running_reqs"))
    waiting = load.get("num_waiting_reqs", load.get("total_waiting_reqs"))
    return (running if isinstance(running, int) else 0) + (waiting if isinstance(waiting, int) else 0)
```

注意边界：`abort_server_until_idle` 拿不到 load 时会 warning 后返回，不代表 server 一定空闲。关键更新前仍要观察 flush 或 load。

---

## Q5：external engine 模式下 Slime 还做什么？

Slime 不启动和不杀外部 SGLang 进程，但仍会发现 server、推导拓扑、创建 Ray adapter、做有限 server-args 校验并注册 router。

```python
# 来源：slime/backends/sglang_utils/external.py L79-L104
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

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L329-L331
def shutdown(self):
    if self.args.rollout_external:
        return
```

排障判断：

- external 地址格式错，先查 `normalize_external_engine_addr` 的异常。
- 只有进入 `external_engine_need_check_fields` 的字段不一致才会 assertion fail。该集合在通用 `sglang_*` 与 YAML overrides 合入前固定，而且 host/port、rank、TP/DP/PP/EP 等被跳过。
- external 模式不支持本地进程 recover，`ExternalRolloutServer.recover` 只记录 warning。
- external `shutdown()` 入口直接返回，因此也不会注销 router worker；外层需负责停 router 或清理陈旧注册。

---

## Q6：为什么 prefill worker 注册会缺 bootstrap port？

PD disaggregation 新版 router 注册 prefill worker 时需要 bootstrap room。Slime 在端口分配阶段只给 `worker_type == "prefill"` 加 `disaggregation_bootstrap_port`；注册阶段如果缺失就抛异常。

```python
# 定位骨架（据 `slime/ray/rollout.py` L996-L998 删节）：
if worker_type == "prefill":
    addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = get_port()
```

```python
# 定位骨架（据 `slime/backends/sglang_utils/sglang_engine.py` L216-L230 补外层上下文）：
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
```

验证方法：打印或断点查看 `addr_and_ports[rank]`，prefill 必须有 `disaggregation_bootstrap_port`；decode 不需要。

---

## Q7：distributed update hang 先查什么？

先查 rank 布局和锁。`world_size = sum(engine_gpu_counts) + 1`，每个 engine 的 `rank_offset` 从 cumulative 加 1 开始。heterogeneous TP 场景下不能假设每个 engine 都占相同 rank 数。

```python
# 定位骨架（据 `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` L288-L304 补注释）：
world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

# Compute cumulative rank offsets: engine i starts at cumulative[i] + 1.
cumulative = [0]
for c in engine_gpu_counts:
    cumulative.append(cumulative[-1] + c)

refs = [
    engine.init_weights_update_group.remote(
        master_address=master_address,
        master_port=master_port,
        rank_offset=cumulative[i] + 1,
        world_size=world_size,
        group_name=group_name,
        backend="nccl",
    )
    for i, engine in enumerate(rollout_engines)
]
```

检查顺序：

1. managed 模式下 `len(rollout_engines)` 是否等于 node 0 engine 数；external 下是否等于公开地址数。
2. `engine_gpu_counts` 是否和每个 engine 的实际 TP/PP GPU 数一致。
3. `world_size` 是否是全部 engine GPU 加训练 rank 0。
4. bucket 更新是否持有 `rollout_engine_lock`，避免 broadcast 顺序交错。

---

## Q8：为什么新恢复的 engine 可能用旧权重？

fault tolerance 恢复后，训练侧必须重新连接 engine 并把最新权重推过去。`MegatronTrainRayActor` 依靠 `num_new_engines` 触发 `connect_rollout_engines`。

```python
# 定位骨架（据 `slime/backends/megatron_utils/actor.py` L613-L624 删节）：
if num_new_engines > 0 or reconnect_rollout_engines:
    self.weight_updater.connect_rollout_engines(
        rollout_engines,
        rollout_engine_lock,
        engine_gpu_counts=engine_gpu_counts,
        engine_gpu_offsets=engine_gpu_offsets,
        all_engine_actors=all_engine_actors,
    )
    dist.barrier(group=get_gloo_group())
    if dist.get_rank() == 0:
        ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())
```

验证方法：打开 CI 检查时，训练侧会随机挑一个 engine 比对版本。它能发现抽中目标的漂移，但不能证明所有 engine 一致；生产验收应遍历全部控制面 engine。

```python
# 来源：slime/backends/megatron_utils/actor.py L630-L636
if self.args.ci_test and len(rollout_engines) > 0 and self.weight_updater.weight_version > 0:
    engine = random.choice(rollout_engines)
    engine_version = ray.get(engine.get_weight_version.remote())
    if str(engine_version) != str(self.weight_updater.weight_version):
        raise RuntimeError(
            f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
        )
```

---

## Q9：GPU 映射错了会表现成什么？

常见表现是 SGLang TP 占到训练卡、OOM、某些 engine 启动失败，或者性能异常但没有明确错误。`base_gpu_id` 的最终来源优先是 Placement Group 传入值，备用才是 `get_base_gpu_id`。

```python
# 定位骨架（据 `slime/backends/sglang_utils/sglang_engine.py` L24-L49 删节）：
def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index

def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )
```

验证方法：在 actor 内打印 `CUDA_VISIBLE_DEVICES`、`base_gpu_id`、`rank`、`worker_type`，再对照 Placement Group 的 `reordered_gpu_ids`。

---

## Q10：如何判断权重路径该选 distributed、tensor 还是 disk？

| 路径 | 优势 | 主要风险 | 适合场景 |
|------|------|----------|----------|
| distributed | 不经 HTTP 传大 tensor，跨节点直接 NCCL | rank/world_size/lock 配错会 hang | 默认 on-policy |
| tensor | colocate 同机延迟低 | 依赖 GPU IPC 和 engine/actor GPU offset | actor 与 rollout 同机 |
| disk | 语义简单，适合超大模型和跨节点 | 文件系统一致性、保存/清理开销 | 共享 FS 或 checkpoint 驱动部署 |
| disk delta | 传输少 | managed 每 host actor/挂载必须覆盖；external 不自动满足 | 带宽受限或远端 FS 慢 |

选择时先问两个问题：engine 和训练 actor 是否同机同 GPU 池；文件系统是否能可靠提供读后可见性。再看权重大小和更新频率。

## Q11：为什么本地 engine 初始化可能一直卡住而不是超时？

`_wait_server_healthy` 是无限 `while True`，HTTP GET 没有 timeout；它只在返回 200 时成功，或检测到子进程死亡时失败。若进程活着但端点永久不健康，`ray.get(init_handles)` 可以无限等待。

处理：同时观察子进程存活、端口监听和 `/health_generate`；在生产封装层为 init ref 增加外部超时与故障清理，不要只等源码循环自己退出。

预期：健康端点在限定时间内返回 200；超时后能明确终止/回收对应 actor 与子进程，而不是留下永久 initializing 状态。

## Q12：为什么第二个多节点 external 地址没有注册到 router？

external 构造按地址序号传 `rank`，`_compute_server_args` 又按 `rank % nnodes` 计算 `node_rank`。每个地址本应代表可访问的 HTTP 控制端点，但当 `info.num_gpus > num_gpus_per_node` 时，第二个地址可能得到 `node_rank=1`，随后 `_register_to_router` 和 `_make_request` 都按非 node 0 跳过。

处理：打印每个 external info 的地址、num_gpus、adapter rank、推导 nnodes/node_rank 和最终 router worker 列表。修复前不要宣称多 external × 多节点拓扑受支持；可在 adapter 构造中把“外部 engine 序号”与“SGLang 内部 node rank”拆成不同字段。

预期：每个公开 external HTTP 地址都有一个会注册、会接收控制请求的 adapter；内部多节点拓扑只作为 server info，不应把公开 adapter 降成非控制节点。
