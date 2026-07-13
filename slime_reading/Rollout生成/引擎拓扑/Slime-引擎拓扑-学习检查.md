---
title: "引擎拓扑 · 学习检查"
type: exercise
framework: slime
topic: "引擎拓扑"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 引擎拓扑 · 学习检查

这份清单不是检查“看过几段源码”，而是检查你能否把 rollout 拓扑从声明画到运行时，并能定位常见失败。

## 读者能做什么

- [ ] 能画出 `SglangConfig → ModelConfig → ServerGroupConfig → RolloutServer → ServerGroup → SGLangEngine` 的对象链。
- [ ] 能说明 `ServerGroupConfig` 和 `ServerGroup` 的区别：一个是声明，一个是运行时 actor 池。
- [ ] 能沿一份 YAML 复述它如何进入 `_resolve_sglang_config`、`ModelConfig.resolve`、`_start_router`、`_make_group`、`start_engines`。
- [ ] 能解释 `gpu_offset`、`rank_offset`、`port_cursors` 三个游标分别推进什么。
- [ ] 能判断 regular、prefill、decode、encoder、placeholder 五种 `worker_type` 的运行含义。
- [ ] 能说明 PD 为什么只改变 Router 和 worker pool，不改变训练侧默认 generate URL。
- [ ] 能说明 EPD 为什么 encoder group 必须先 `ray.get` ready，再把 `encoder_urls` 注入 regular/prefill group。
- [ ] 能解释 zero GPU rollout 为什么仍可创建默认 Router 映射但没有 local engines。
- [ ] 能说明 `update_weights` 是模型级开关，`needs_offload` 是 group 级资源开关。
- [ ] 能区分 `all_engines` 的 node actor 数与 `engines` 的逻辑 HTTP engine 数，并能在跨节点 TP 中手算两者。
- [ ] 能指出模型名、PD 两侧完整性、Router HTTP readiness 都没有由当前配置解析自动保证。

## 排障验收

- [ ] 看到 `sglang_config total GPUs` 与 `rollout_num_gpus` 不一致的报错时，能回到 YAML 逐组求和。
- [ ] 看到某组没有 engine 时，能区分 `placeholder`、zero GPU、debug train-only 和 actor 创建失败。
- [ ] 看到多模型请求打错模型时，能检查默认 generate 与 `get_model_url(args, model_name)` 的区别。
- [ ] 看到权重没有更新时，能检查 `ModelConfig.resolve` 推断出的 `update_weights` 和 `get_updatable_engines_and_lock` 返回值。
- [ ] 看到端口或 init hang 时，能查 `Ports for engine` 日志和 `disaggregation_bootstrap_port`。
- [ ] 看到 colocate 显存冲突时，能检查 `_make_group` 日志里的 `needs_offload`。
- [ ] 看到 Router 进程存活但请求失败时，能用 HTTP readiness 区分“活着”和“可服务”。
- [ ] 看到两个同名模型只有后者可寻址时，能定位 `servers[model_cfg.name]` 覆盖并检查遗留 Router/actor。
- [ ] 看到等价 checkpoint 路径导致错误更新策略时，能解释字符串比较的边界并改为显式 `update_weights`。
- [ ] 看到 EPD phase 1 收集到 0 个 URL 时，不会把“继续启动”误判为 EPD 已正确接线。
- [ ] 看到 unknown model name 静默打到 actor 时，能识别 `get_model_url` 的默认 Router 回退。
- [ ] 看到 group 只部分重叠训练 GPU 却整组 offload 时，能用 group 起点判据解释结果。

## 可执行验证

在依赖完整的环境中，优先跑轻量单测：

```powershell
Set-Location 'F:\源码阅读\slime'
python -m pytest tests/utils/test_sglang_config.py -q
```

预期覆盖：

- 多模型 GPU 总数累加。
- zero GPU 配置没有 ServerGroup，但有默认 Router 映射。
- `start_rollout_servers` 返回 init handle，不在内部等待普通 engine。
- EPD encoder 先等待，再把 URL 注入 regular/prefill。
- `get_model_url` 能按模型名选择 Router，找不到时回退默认 Router。

还应补做以下静态验收；它们是在确认当前实现边界，不是在宣称这些边界已经被单测防住：

- YAML 中重复 `name` 不会在 `from_yaml` 报错，最终字典写入会覆盖同名键。
- `update_weights` 自动推断是 `effective_model_path != args.hf_checkpoint` 的字符串比较。
- PD 判据对 prefill/decode 使用 `any`，没有成对校验。
- EPD 注入前带有 `if encoder_urls`，空 URL 会跳过注入并继续。
- `_start_router` 只 sleep 后检查 `process.is_alive()`；复用用户 Router 时直接返回。
- `needs_offload` 只用 group 起点判断是否落在 Megatron GPU 范围内。

更重的集成验证需要 GPU、Ray、SGLang 和模型数据：

```powershell
Set-Location 'F:\源码阅读\slime'
python -m pytest tests/test_qwen2.5_0.5B_sglang_config.py -q
```

这类测试会真正启动训练与 rollout，适合验证 placeholder、混合 `num_gpus_per_engine`、colocate/offload 等路径。本地缺少 `ray`、`sglang_router`、`httpx`、GPU 或模型数据时，不要把 collection/import 失败误判成拓扑逻辑失败。

## 源码复述题

完成本专题后，尝试不看笔记回答：

- [ ] 为什么 `_resolve_sglang_config` 要在 YAML 路径检查 GPU 总数？
- [ ] 为什么 `_start_router` 的 `force_new=True` 只在多模型后续模型中出现？
- [ ] 为什么 `ServerGroup.start_engines` 返回的是 init handle，而不是直接等待？
- [ ] 为什么 `RolloutServer.engines` 只包含 node-0 engine，而 `all_engines` 还要保留？
- [ ] 为什么 external engines 不应该和 `--sglang-config` 同时使用？
- [ ] 为什么同名模型不仅会覆盖索引，还可能留下已经启动却不可寻址的资源？
- [ ] 为什么 `num_gpus // min(num_gpus_per_engine, num_gpus_per_node)` 算出的不是逻辑 engine 数？
- [ ] 为什么只配 prefill 也能让 Router 进入 PD，而这不代表拓扑可用？

## 下一步

如果你关注单个 SGLang server 的生命周期，继续读 [[Slime-SGLang-Engine]]。如果你关注已经外部启动的 SGLang 集群如何接入 Slime，继续读 [[Slime-外部推理引擎]]。如果你要看拓扑 ready 后如何生产样本，继续读 [[Slime-SGLang-Rollout]]。
