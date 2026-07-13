---
title: "引擎拓扑 · 排障指南"
type: troubleshooting
framework: slime
topic: "引擎拓扑"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# 引擎拓扑 · 排障指南

这篇按排障组织。EngineTopology 的问题通常不是“哪个函数错了”，而是配置、资源、请求和权重控制四条流的边界被混淆了。

## 快速定位表

| 症状 | 最可能边界 | 源码入口 | 先验证什么 |
|------|------------|----------|------------|
| `sglang_config total GPUs` assert | YAML 与 `--rollout-num-gpus` 不一致 | `_resolve_sglang_config` | YAML 每个 group 的 `num_gpus` 之和 |
| PD worker 起了但请求不通 | Router 没进入 PD 或端口错误 | `_start_router`、端口分配 | Router 参数、`disaggregation_bootstrap_port` |
| ref/reward 模型被更新 | `update_weights` 配错 | `ModelConfig.resolve`、`get_updatable_engines_and_lock` | 模型级 `update_weights` |
| 某组没有 engine | `placeholder` 或 zero GPU | `ServerGroup.start_engines` | `worker_type` 与 `all_engines` |
| 多模型请求总打 actor | custom rollout 没选 Router | `get_model_url` | 是否用 `args.sglang_model_routers` |
| EPD regular/prefill 没拿到 encoder | encoder 未先 ready 或 URL 为空 | EPD 两阶段启动 | `EPD phase 1 done` 日志和 overrides |
| 两个同名模型只剩后一个可寻址 | `servers[name]` 字典覆盖 | `start_rollout_servers` | YAML 中 `name` 是否全局唯一 |
| Router 进程活着但请求拒绝或超时 | 启动检查只证明进程存活 | `_start_router` | Router HTTP 端点是否真的 ready |
| 多节点 engine 数量看起来翻倍 | 把 node actor 数当成逻辑 engine 数 | `_make_group`、`ServerGroup.engines` | 分开统计 `all_engines` 与 `engines` |
| 只配 prefill 或 decode 仍进入 PD | PD 判据是任一侧存在 | `has_pd_disaggregation` | 两类 group 是否成对且 GPU 数可用 |

## 1. 什么时候用 PD，什么时候用 regular？

先看 workload，而不是先看参数。上游文档把 PD 用在长上下文、多轮、decode 占主导、prefix cache locality 重要、prefill/decode 需要不同 TP 或内存设置的场景。

```markdown
# 定位骨架（据 `slime/docs/en/advanced/pd-disaggregation.md` L5-L15 删节）：
Use PD Disaggregation when:

- rollout contexts are long or grow across turns;
- decode dominates rollout time;
- prefix-cache locality matters for multi-turn sessions;
- prefill and decode need different TP, memory, or runtime settings;

For short single-turn tasks, the default regular SGLang engine layout is usually simpler.
```

落到 Slime 源码里，PD 的最小判据非常直接：只要一个 `ModelConfig` 里存在 `prefill` 或 `decode` group，这个模型的 Router 就按 PD 启动。

```python
# 来源：slime/backends/sglang_utils/sglang_config.py L102-L108
@property
def has_pd_disaggregation(self) -> bool:
    return any(g.worker_type in ("prefill", "decode") for g in self.server_groups)

@property
def has_encoder_disaggregation(self) -> bool:
    return any(g.worker_type == "encoder" for g in self.server_groups)
```

验证方法：

- 短单轮任务优先 regular，少一个 Router PD 状态和 bootstrap 端口维度。
- 长 prompt 或 agent 多轮任务再考虑 PD。
- 如果用了 PD，启动日志里 Router 参数应包含 `pd_disaggregation=True`。

这里还有一个配置门禁没有由源码替你完成：`has_pd_disaggregation` 用的是 `any(...)`，并不验证 prefill 与 decode 成对。因此“Router 进入 PD”只说明至少出现了一侧，不能证明拓扑完整。若只配置 prefill 或只配置 decode，解析仍可通过；发布前应人工或在配置生成器中断言两侧都存在。

## 2. `--prefill-num-servers` 和 `--sglang-config` 有什么边界？

`--prefill-num-servers` 是单模型、简单 PD 的快捷入口；`--sglang-config` 是完整拓扑声明入口。两者不能同时使用。

```python
# 定位骨架（据 `slime/backends/sglang_utils/arguments.py` L162-L173 删节）：
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

选型规则：

| 需求 | 入口 |
|------|------|
| 只想把默认 actor 分成 prefill/decode | `--prefill-num-servers` |
| prefill 和 decode 用不同 `num_gpus_per_engine` | `--sglang-config` |
| 多模型 actor/ref/reward | `--sglang-config` |
| placeholder、EPD、group overrides | `--sglang-config` |
| engine 已由外部系统启动 | `--rollout-external-engine-addrs`，不是本专题本地拓扑 |

## 3. 为什么 YAML GPU 总数必须等于 `--rollout-num-gpus`？

本地 rollout 的 Ray PG 已经按 `args.rollout_num_gpus` 创建 rollout 视图。YAML 如果声明更多 GPU，后面 group 会越界；声明更少 GPU，会留下未解释的 rollout 槽位。Slime 在解析阶段直接 assert，避免把错误拖到 Ray actor 创建阶段。

```python
# 定位骨架（据 `slime/ray/rollout.py` L1231-L1239 删节）：
if getattr(args, "sglang_config", None) is not None:
    config = SglangConfig.from_yaml(args.sglang_config)
    expected = args.rollout_num_gpus
    actual = config.total_num_gpus
    assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
    return config
```

排查步骤：

- 把 YAML 中每个 model、每个 group 的 `num_gpus` 相加。
- 对照训练命令里的 `--rollout-num-gpus`。
- 如果是 colocate 且没显式传 rollout GPU，先看参数解析是否把 rollout GPU 推断成 actor GPU。

## 4. placeholder 为什么会影响布局但没有 engine？

placeholder 是资源占位，不是禁用开关。它在 `_make_group` 中推进 `gpu_offset`，但 `ServerGroup.start_engines` 看到 `worker_type == "placeholder"` 会直接返回空 handle。

```python
# 定位骨架（据 `slime/ray/rollout.py` L1152-L1169 删节）：
group = ServerGroup(
    all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else [],
    num_gpus_per_engine=gpus_per_engine,
    worker_type=group_cfg.worker_type,
    rank_offset=engine_offset,
    gpu_offset=gpu_offset,
    router_ip=router_ip,
    router_port=router_port,
)
engine_offset += num_engines
gpu_offset += group_cfg.num_gpus
return group
```

```python
# 定位骨架（据 `slime/ray/rollout.py` L137-L152 删节）：
if port_cursors is None:
    port_cursors = {}
if self.args.debug_train_only or self.worker_type == "placeholder":
    self.num_new_engines = 0
    return [], port_cursors
```

验证方法：

- `server.server_groups` 里能看到 placeholder group。
- `server.engines` 不会包含 placeholder engine。
- 后续 group 的 `gpu_offset` 会跨过 placeholder 的 `num_gpus`。

## 5. 多模型场景下为什么默认请求没有打到 ref？

默认 generate 函数只使用 `args.sglang_router_ip/port`，也就是第一个模型的 Router。多模型部署只是把模型 Router 表写到 `args.sglang_model_routers`，不会自动让默认 generate 去访问所有模型。

```python
# 定位骨架（据 `slime/rollout/sglang_rollout.py` L153-L203 删节）：
async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    with trace_span(sample, "sglang_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
        output = await post(url, payload, headers=headers)
        span.update(build_sglang_meta_trace_attrs(output["meta_info"]))
```

如果 custom rollout 需要 ref、reward 或 judge 模型，应显式用：

```python
# 定位骨架（据 `slime/rollout/sglang_rollout.py` L65-L81 删节）：
def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"
```

验证方法：

- 打印或断点查看 `args.sglang_model_routers`。
- 检查 custom generate 是否传入正确 `model_name`。
- 如果 model name 不存在，`get_model_url` 会回退默认 Router，这可能掩盖拼写错误。

生产代码最好在调用 `get_model_url` 前先断言 `model_name in args.sglang_model_routers`。这个 helper 的回退是兼容策略，不是模型名校验；把 `rewrad` 拼错后，请求可能成功落到第一个 actor Router，形成比显式报错更危险的静默串模。

## 6. 为什么 ref/reward 没有收到或不该收到权重？

`update_weights` 是模型级开关。`ModelConfig.resolve` 会根据 `model_path` 和 `args.hf_checkpoint` 自动推断默认值；后续 `RolloutManager` 只拿第一个可更新 server。

```python
# 定位骨架（据 `slime/backends/sglang_utils/sglang_config.py` L90-L100 删节）：
if self.update_weights is None:
    if effective_model_path != args.hf_checkpoint:
        logger.warning(
            f"Model '{self.name}' uses model_path='{effective_model_path}' which differs "
            f"from hf_checkpoint='{args.hf_checkpoint}'. Defaulting update_weights to False. "
            f"Set update_weights explicitly in the config to suppress this warning."
        )
        self.update_weights = False
    else:
        self.update_weights = True
```

```python
# 定位骨架（据 `slime/ray/rollout.py` L511-L540 删节）：
def _get_updatable_server(self) -> Any | None:
    for srv in self.servers.values():
        if srv.update_weights:
            return srv
    return None

def get_updatable_engines_and_lock(self):
    srv = self._get_updatable_server()
    engines = srv.engines if srv else []
    gpu_counts = srv.engine_gpu_counts if srv else []
    gpu_offsets = srv.engine_gpu_offsets if srv else []
    num_new = srv.num_new_engines if srv else 0
    all_engine_actors = srv.all_engines if srv else []
    return engines, self.rollout_engine_lock, num_new, gpu_counts, gpu_offsets, all_engine_actors
```

排查建议：

- actor 模型应显式 `update_weights: true`。
- ref/reward 应显式 `update_weights: false`。
- 如果多个模型都设为 true，当前源码只取第一个可更新 server；不要把它当作完整多 actor 同步能力。
- 自动推断只比较两个路径字符串，不会做 `realpath`、符号链接解析或 checkpoint 内容比对。两个字符串不同但指向同一目录时会推断为不更新；同一字符串对应的目录内容后来被替换，也不会被识别。生产配置不要依赖这个启发式，始终显式写布尔值。

## 7. EPD 为什么必须等 encoder 先 ready？

EPD 的非 encoder worker 需要 `encoder_urls`。这不是性能优化，而是启动依赖：没有 encoder URL，regular/prefill worker 不知道图像或 encoder-only 服务在哪里。

```python
# 定位骨架（据 `slime/ray/rollout.py` L1171-L1205 删节）：
if has_epd:
    encoder_urls: list[str] = []
    for group_cfg in model_cfg.server_groups:
        if group_cfg.worker_type != "encoder":
            continue
        group = _make_group(group_cfg, router_ip, router_port)
        handles, port_cursors = group.start_engines(port_cursors)
        if handles:
            ray.get(handles)
        urls = ray.get([e.get_url.remote() for e in group.engines])
        encoder_urls.extend(u for u in urls if u is not None)
        server_groups.append(group)

    for group_cfg in model_cfg.server_groups:
        if group_cfg.worker_type == "encoder":
            continue
        overrides_extra = {}
        if encoder_urls and group_cfg.worker_type in ("prefill", "regular"):
            overrides_extra["language_only"] = True
            overrides_extra["encoder_urls"] = encoder_urls
```

验证方法：

- 看 `EPD phase 1 done` 是否收集到 URL。
- 断点看 non-encoder group 的 `sglang_overrides` 是否包含 `language_only` 和 `encoder_urls`。
- 单测 `test_start_rollout_servers_waits_for_epd_encoder_before_non_encoder` 覆盖了这个顺序。

注意“encoder actor 已经 init”不等于“收集到了 encoder URL”。源码只有在 `encoder_urls` 非空时才注入 overrides；空列表不会单独报错，non-encoder group 会继续启动。可靠验收应把日志中的 URL 数量设为硬条件：至少一个 encoder group 时，收集数必须大于零，且每个 regular/prefill group 都应看到同一组 URL。

## 8. 为什么 Router 进程活着，请求仍可能失败？

`_start_router` 对新 Router 的检查是固定等待 3 秒后执行 `process.is_alive()`；它没有请求健康端点。用户显式提供 `sglang_router_ip` 时，函数甚至直接复用地址，不做存活或可达性检查。因此“Router launched”日志只证明子进程没有在前三秒退出，不证明端口已监听、路由表可用或 HTTP 请求能成功。

排查步骤：

1. 对日志中的精确 IP/端口发健康请求，而不是只查进程列表。
2. 若是用户传入 Router，确认地址对 Ray worker 所在网络命名空间可达。
3. 若健康端点尚未 ready，增加调用侧重试与超时；不要用无限等待掩盖错误。
4. 预期结果是健康请求成功后再开始 engine 注册与 generate，而不是仅看到进程 PID。

## 9. 为什么同名模型会留下“孤儿”资源？

`SglangConfig.from_yaml` 会逐项创建 `ModelConfig`，没有检查 `name` 唯一。`start_rollout_servers` 也会为每一项先启动 Router 和 groups，最后才执行 `servers[model_cfg.name] = RolloutServer(...)`。因此后一个同名项覆盖前一个字典值，但前一个已经启动的 Router/actor 不会随覆盖自动关闭。

验证方法：

- 在启动前检查 YAML 的模型名集合大小是否等于模型条目数。
- 对照 Router 启动日志数量与 `args.sglang_model_routers` 的键数量；前者更多时应怀疑覆盖。
- 修复后的预期不是“选择正确的同名项”，而是模型名全局唯一并 fail fast。

## 10. 多节点时怎样数清 engine？

跨节点的一个逻辑 SGLang engine 由多个 Ray node actor 组成。`_make_group` 中局部变量 `num_engines = num_gpus // min(num_gpus_per_engine, num_gpus_per_node)` 实际决定 `all_engines` 的 node actor 槽位数；而 `RolloutServer.engines` 只保留每个逻辑 engine 的 node-0 actor。比如每节点 4 GPU、每逻辑 engine 8 GPU、group 共 16 GPU：`all_engines` 有 4 个 node actor，但逻辑 engine 只有 2 个。

排查时分别记录：

- `len(group.all_engines)`：node actor 数。
- `len(group.engines)`：逻辑 HTTP engine / node-0 actor 数。
- `nodes_per_engine = ceil(num_gpus_per_engine / num_gpus_per_node)`：一个逻辑 engine 跨几节点。

不要用 `all_engines` 数量推导 Router worker 数或请求并发槽位。

## 11. 为什么只重叠一部分 GPU，整组却被 offload？

`needs_offload` 只比较 group 起点 `group_abs_start < megatron_num_gpus`，不是逐 GPU 求交集。一个 group 如果从共享区开始、尾部跨入独占 rollout 区，整个 `ServerGroup` 仍被标记为需要 offload。反过来，起点已在独占区时整组都不会 offload。

验证方法：打印每组的半开区间 `[group_abs_start, group_abs_start + num_gpus)` 与 Megatron 区间 `[0, megatron_num_gpus)`。若仅部分相交，不要期待当前实现细粒度处理；应调整 group 边界，使共享与独占 GPU 不落在同一 group。

## 12. external engines 和 `--sglang-config` 为什么互斥？

两者的 ownership 不同。`--sglang-config` 表示 Slime 负责在本地 Ray PG 内启动 ServerGroup；external engines 表示外部系统已经启动 SGLang server，Slime 只发现、注册 Router 和控制权重更新。

上游 external 文档也明确说，这两条路径拥有不同边界：

```markdown
# 定位骨架（据 `slime/docs/en/advanced/external-rollout-engines.md` L41-L48 删节）：
`--rollout-external-engine-addrs` and `--sglang-config` are mutually exclusive because they own different boundaries:

- `--sglang-config`: slime owns the engine lifecycle.
- `--rollout-external-engine-addrs`: an external system owns the engine lifecycle.

If your main requirement is multi-model serving, frozen reference/reward models, PD disaggregation, or heterogeneous group configuration, prefer `--sglang-config`.
```

如果用户已经有外部 SGLang 集群，继续读 [[Slime-外部推理引擎]]；如果 Slime 要自己拉起 PD、多模型、placeholder 或 EPD，就留在本专题。
