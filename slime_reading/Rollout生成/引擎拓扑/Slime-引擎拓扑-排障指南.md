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
updated: 2026-07-10
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

## 1. 什么时候用 PD，什么时候用 regular？

先看 workload，而不是先看参数。上游文档把 PD 用在长上下文、多轮、decode 占主导、prefix cache locality 重要、prefill/decode 需要不同 TP 或内存设置的场景。

```markdown
# 来源：slime/docs/en/advanced/pd-disaggregation.md L5-L15
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

## 2. `--prefill-num-servers` 和 `--sglang-config` 有什么边界？

`--prefill-num-servers` 是单模型、简单 PD 的快捷入口；`--sglang-config` 是完整拓扑声明入口。两者不能同时使用。

```python
# 来源：slime/backends/sglang_utils/arguments.py L162-L173
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
# 来源：slime/ray/rollout.py L1231-L1239
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
# 来源：slime/ray/rollout.py L1152-L1169
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
# 来源：slime/ray/rollout.py L137-L152
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
# 来源：slime/rollout/sglang_rollout.py L153-L203
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
# 来源：slime/rollout/sglang_rollout.py L65-L81
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

## 6. 为什么 ref/reward 没有收到或不该收到权重？

`update_weights` 是模型级开关。`ModelConfig.resolve` 会根据 `model_path` 和 `args.hf_checkpoint` 自动推断默认值；后续 `RolloutManager` 只拿第一个可更新 server。

```python
# 来源：slime/backends/sglang_utils/sglang_config.py L90-L100
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
# 来源：slime/ray/rollout.py L511-L540
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

## 7. EPD 为什么必须等 encoder 先 ready？

EPD 的非 encoder worker 需要 `encoder_urls`。这不是性能优化，而是启动依赖：没有 encoder URL，regular/prefill worker 不知道图像或 encoder-only 服务在哪里。

```python
# 来源：slime/ray/rollout.py L1171-L1205
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

## 8. external engines 和 `--sglang-config` 为什么互斥？

两者的 ownership 不同。`--sglang-config` 表示 Slime 负责在本地 Ray PG 内启动 ServerGroup；external engines 表示外部系统已经启动 SGLang server，Slime 只发现、注册 Router 和控制权重更新。

上游 external 文档也明确说，这两条路径拥有不同边界：

```markdown
# 来源：slime/docs/en/advanced/external-rollout-engines.md L41-L48
`--rollout-external-engine-addrs` and `--sglang-config` are mutually exclusive because they own different boundaries:

- `--sglang-config`: slime owns the engine lifecycle.
- `--rollout-external-engine-addrs`: an external system owns the engine lifecycle.

If your main requirement is multi-model serving, frozen reference/reward models, PD disaggregation, or heterogeneous group configuration, prefer `--sglang-config`.
```

如果用户已经有外部 SGLang 集群，继续读 [[Slime-外部推理引擎]]；如果 Slime 要自己拉起 PD、多模型、placeholder 或 EPD，就留在本专题。
