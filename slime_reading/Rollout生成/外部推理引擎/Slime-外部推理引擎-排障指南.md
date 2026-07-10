---
title: "外部推理引擎 · 排障指南"
type: troubleshooting
framework: slime
topic: "外部推理引擎"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 外部推理引擎 · 排障指南

这篇按部署和排障症状组织。external 模式的常见误判是：以为 Slime 仍拥有 rollout engine 进程，或者以为发现出的 `rollout_num_gpus` 会进入 Ray PG。排障时先判断问题发生在发现、资源、router、generate、权重同步还是外部生命周期。

---

## 快速排障表

| 症状 | 先看边界 | 源码入口 | 验证 |
|------|----------|----------|------|
| 传了 external 地址但仍走内置 launch | 参数解析 | `args.rollout_external` | 日志是否调用 `apply_external_engine_info_to_args` |
| `/server_info` 访问失败 | 网络/proxy/server 版本 | `get_server_info` | 手动请求 `/server_info` 和 `/get_server_info` |
| router 没有 worker | 注册控制面 | `_init_external`、`_register_to_router` | 查 router `/workers` |
| Ray 还在等 rollout GPU | PG 布局 | `_get_placement_group_layout` | external 下 PG GPU 数应只等于训练 GPU |
| prefill 注册失败 | PD bootstrap | `external_engine_init_kwargs`、router payload | prefill server_info 是否含 bootstrap port |
| generate 长时间 retry | HTTP 数据面 | `http_utils._post` | external server 和 router 是否持续 5xx/连接失败 |
| update 权重失败 | 权重通道 | NCCL 或 disk/delta 选择 | trainer 与 engine 是否 NCCL 互通或共享同一路径 |
| engine 挂了 Slime 没 recover | 生命周期所有权 | `ExternalRolloutServer.recover` | 外部编排系统是否重启 server |

---

## Q1：什么时候该用 external，什么时候该用 `--sglang-config`？

使用 external 的前提是：SGLang server 已经由训练任务外部启动，Slime 只连接它们。使用 `--sglang-config` 的前提是：你希望 Slime 自己管理 topology、server group、router 和生命周期。

源码也强制了两者互斥。

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

判断规则：

- 已有独立 serving 集群、异构 GPU、跨集群推理：优先 external。
- 需要多模型、frozen reference/reward、per-group overrides：优先 `--sglang-config`。
- 只想让 Slime 少占 GPU，但仍要它管理 server：这不是 external 的目标，应重新设计 PG/colocate 配置。

---

## Q2：为什么 external 下 Slime 不 recover engine？

外部 server 进程不是 Slime 启动的，Slime 没有它的 PID、PG bundle 或重启参数。源码中 `ExternalRolloutServer` 明确把 recover/offload/onload 做成 warning 或空操作。

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

验证方法：

- external server 挂掉时，先看外部编排系统的重启状态。
- Slime 侧 HTTP retry 只能缓冲短暂抖动，不能替代外部健康管理。
- `SGLangEngine.shutdown` 在 external 下直接返回，不会 kill server。

---

## Q3：`rollout_num_gpus` 为什么有值但 PG 不占 GPU？

external discovery 写入的 `rollout_num_gpus` 是逻辑 serving 容量，来自所有 external engine 的 `num_gpus` 求和。Ray PG 布局另走 `rollout_external` 分支，不把这些 GPU 算进训练 job。

```python
# 来源：slime/ray/placement_group.py L100-L128
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

排查方法：

- 看 `Creating placement group with {num_gpus} GPUs` 日志，external 普通训练应等于 actor GPU 数。
- 运行 `pytest slime/tests/test_placement_group.py -k external -q` 可验证布局公式。
- 若 PG 仍包含 rollout GPU，说明 `rollout_external` 没在参数解析阶段变成真。

---

## Q4：sanity check 失败说明什么？

`_init_external` 会重新请求外部 server info，把 Slime 期望的可检查字段和实际字段比对。失败通常说明外部 server 启动参数和训练任务参数不一致。

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

检查顺序：

1. 请求外部 server `/server_info`，确认返回字段。
2. 对照 Slime 日志中的 `expect_server_args`。
3. 如果是 `dtype`、routing replay、memory saver 等字段不一致，修外部 launch 参数或 Slime CLI。
4. 不要直接跳过 sanity check；否则可能在 rollout 中产生 silent mismatch。

---

## Q5：为什么 PD prefill 注册失败？

PD prefill worker 注册 router 时需要 bootstrap port。这个 port 来自 external discovery 的 `disaggregation_bootstrap_port`，再通过 `external_engine_init_kwargs` 传给 `SGLangEngine.init`。

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

router 注册阶段如果 prefill 没有 bootstrap port，会直接报错。

```python
# 来源：slime/backends/sglang_utils/sglang_engine.py L216-L232
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

验证方法：外部 prefill server 的 `/server_info` 必须包含 `disaggregation_bootstrap_port`；decode worker 不需要这个字段。

---

## Q6：`/server_info` 请求为什么在集群里失败？

常见原因是训练任务的 HTTP proxy 环境劫持了内部请求，或者外部 server 只绑定了本地回环地址。`get_server_info` 会尝试两个 endpoint，但不会绕过网络问题。

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

E2E 测试显式设置 `no_proxy`。

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

验证方法：

- 在训练 job 环境里直接 `curl http://host:port/server_info`。
- 检查 `no_proxy/NO_PROXY` 是否包含 external host。
- 确认 external server 不是只监听 `127.0.0.1`。

---

## Q7：external 权重同步该选 NCCL 还是 disk？

选择依据不是“external 必须 disk”，而是 trainer 和 external engine 是否能建立稳定的数据通道。

| 条件 | 推荐 |
|------|------|
| 同集群、同网络、NCCL 可达 | full + nccl |
| 跨集群、防火墙、NCCL 不通 | full + disk |
| full checkpoint 太大 | delta + disk |
| serving GPU 与训练 GPU 异构 | disk 或 delta disk |

官方部署 checklist 明确了 disk 的路径要求和 external lifecycle。

```markdown
# 来源：slime/docs/en/advanced/external-rollout-engines.md L95-L101
- External engine HTTP addresses must be reachable from the training job.
- External engines can use an independent SGLang environment; they do not need the slime or Megatron training environment.
- Disk transport supports different GPU models or vendors between training and rollout, as long as SGLang supports the target hardware and model format.
- Disk transport requires trainer and SGLang engines to see the same `--update-weight-disk-dir` path; a path visible only to the trainer is not enough.
- External engines are not recovered by slime fault tolerance; their lifecycle belongs to the external deployment system.
- `--sglang-config` and `--rollout-external-engine-addrs` are mutually exclusive.
- Delta mode does not support `--colocate`, because colocated sync uses CUDA IPC handles and delta encoding does not reduce the actual transfer.
```

如果你选择 disk，最小验证是：训练容器写入 `--update-weight-disk-dir` 后，external server 容器用同一路径能读到文件。

---

## Q8：为什么 HTTP POST 会重试很多次？

`http_utils._post` 默认最多重试 60 次，每次失败睡 1 秒。它能处理短暂 5xx、连接 reset 或 router 切换，但外部 server 长时间不可用时仍会最终抛错。

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
```

排查方法：

- 看 URL 是 router 还是 direct external server。
- 对同一 URL 手动发轻量请求确认是否稳定。
- 如果每次都到 60 次，问题不应靠调大 retry 掩盖，应修 external server 或网络。

---

## Q9：external 可以和 Slime 自 launch engine 混用吗？

同一个训练 job 不能混用。原因不是技术上无法创建两个列表，而是 ownership 会变得不可判定：哪些 server 由 Slime recover，哪些不 recover；哪些 GPU 属于 PG，哪些不属于 PG；哪些模型可更新，哪些 frozen。

当前源码把 external server 包装成一个 default `ExternalRolloutServer`。

```python
# 来源：slime/backends/sglang_utils/external.py L219-L232
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
return servers, init_handles
```

需要多模型或 frozen 模型时，优先用 `--sglang-config`。
