---
title: "SGLang-Engine · 学习检查"
type: exercise
framework: slime
topic: "SGLang-Engine"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# SGLang-Engine · 学习检查

这份清单用于确认你真的能用本专题排障和改代码，而不是只记住几个函数名。

---

## 读者能做什么

- [ ] 能画出 `RolloutManager`、`ServerGroup`、`SGLangEngine`、SGLang HTTP server、`sglang_router`、Megatron updater 的关系图。
- [ ] 能沿一次启动复述：`start_rollout_servers` 解析配置、启动 router、创建 `ServerGroup`、分配端口、创建 Ray actor、调用 `engine.init.remote`、本地启动或外部校验、注册 router。
- [ ] 能说明 `engines` 和 `all_engines` 的区别，以及为什么 node 0 actor 是 HTTP 控制面代表。
- [ ] 能区分 `port`、`nccl_port`、`dist_init_addr` 端口、权重 update `master_port` 四种端口。
- [ ] 能沿 distributed 权重更新复述：取 updatable engines、pause、flush、init update group、HTTP metadata、NCCL broadcast、continue。
- [ ] 能解释 tensor、disk、disk delta 三条权重路径各自适合什么场景。
- [ ] 能说出 external engine 模式保留了哪些 Slime 控制面，移走了哪些本地生命周期职责。

---

## 排障验收

| 给定症状 | 你应该能定位到 |
|----------|----------------|
| router 没有 worker | `_register_to_router`、router 版本、`worker_type`、prefill bootstrap port |
| `flush_cache` 超时 | pending request、`pause_generation` 是否先执行、`/v1/loads?include=core` |
| distributed update hang | `engine_gpu_counts`、`world_size`、`rank_offset`、`rollout_engine_lock` |
| 某些 engine 用旧权重 | `num_new_engines` 是否触发 reconnect、`get_weight_version` 是否匹配 |
| external 启动失败 | external address 格式、`/server_info`、`_init_external` sanity check |
| GPU 占错或 OOM | PG `reordered_gpu_ids`、`gpu_offset`、`base_gpu_id`、`CUDA_VISIBLE_DEVICES` |

---

## 最小口试

1. 为什么 `SGLangEngine` 不是 generate 数据面的核心？
2. 为什么 Ray actor 只申请 `0.2` GPU，SGLang 却能使用完整 TP GPU？
3. 多节点 engine 中，为什么非 node 0 actor 的 `_make_request` 可以直接返回？
4. 为什么 distributed 权重更新要先发 HTTP metadata，再做 NCCL broadcast？
5. disk delta 为什么需要 `all_engine_actors`，而不是只需要 `rollout_engines`？
6. external engine 下 `shutdown` 为什么不能 kill 进程？

---

## 可执行验证

| 验证 | 入口 | 预期 |
|------|------|------|
| engine 初始化 | 日志 `Launch HttpServerEngineAdapter` 和 Ray actor 状态 | `ray.get(init_handles)` 返回，node 0 `/health_generate` 可用 |
| router 注册 | router `/workers` | regular/prefill/decode worker 出现，encoder 不作为普通 worker 注册 |
| update 前清流量 | SGLang `/v1/loads?include=core` | pause/abort 后 request 数降到 0 |
| 权重版本 | `engine.get_weight_version.remote()` | 与 updater `weight_version` 字符串一致 |
| distributed rank | 打印 `engine_gpu_counts` 和 `world_size` | `world_size = sum(engine_gpu_counts) + 1` |
| external sanity | 外部 server `/server_info` | worker type、GPU 数、PD bootstrap 信息与 Slime 推导一致 |

---

## 改代码前的不变量

- [ ] 新增 worker type 时，同时检查 `_compute_server_args`、端口分配、router 注册和 `SglangConfig` 解析。
- [ ] 改权重同步路径时，先说明 metadata 走哪里、tensor 数据走哪里、版本号在哪里更新。
- [ ] 改多节点逻辑时，明确使用 `engines` 还是 `all_engines`。
- [ ] 改 GPU 分配时，验证 PG GPU 槽位和 SGLang `base_gpu_id` 坐标一致。
- [ ] 改 external engine 时，确认不会引入本地进程 kill、offload 或 recover 假设。
- [ ] 改 flush/abort 逻辑时，提供 request 数降为 0 的可观测证据。

---

## 下一篇

如果已经掌握本专题，继续读 [[Slime-分布式权重同步]]，把 Megatron 侧 bucket、TP/EP gather、PP source rank 与 NCCL broadcast 细节补齐。
