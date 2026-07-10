---
title: "CheckpointEngine · 学习检查"
type: exercise
framework: sglang
topic: "CheckpointEngine"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# CheckpointEngine · 学习检查

## 读者能做什么

- [ ] 能画出四道门：启动等待门、HTTP 控制门、Scheduler 执行门、ModelRunner 适配门。
- [ ] 能解释 `/ping` 可达、`initial_weights_loaded=True`、warmup 完成、业务可服务的区别。
- [ ] 能沿 `update.py -> /update_weights_from_ipc -> TokenizerManager -> WeightUpdater -> ModelRunner -> checkpoint-engine worker` 复述主线。
- [ ] 能说明 HTTP body 为什么只有 `zmq_handles`，真实权重为什么不走 HTTP。
- [ ] 能解释 `inference_parallel_size`、TP、GPU UUID、ZMQ handle 的对齐关系。
- [ ] 能说明 `dp_size == 1 or enable_dp_attention` 的 IPC update 约束。
- [ ] 能解释主 TP worker 成功、draft worker 失败时最终 success 和 cache flush 的语义。
- [ ] 能说明为什么 base weights 热更新后默认要 flush prefix cache。
- [ ] 能指出 `weight_load_duration_seconds{source="ipc"}` 为什么是边沿触发。
- [ ] 能解释 post hook warning 对量化模型的风险。

## 源码入口自检

| 你要解释的现象 | 应该能指出的源码入口 |
|----------------|----------------------|
| 等待权重开关 | `server_args.py` 的 `checkpoint_engine_wait_weights_before_ready` |
| 初始权重状态 | `TokenizerManager.init_weight_update` |
| warmup 前等待 | `http_server._wait_and_warmup`、`_wait_weights_ready` |
| HTTP IPC endpoint | `http_server.update_weights_from_ipc` |
| IPC 请求体 | `UpdateWeightsFromIPCReqInput` |
| 控制面锁与 DP 约束 | `TokenizerManager.update_weights_from_ipc` |
| Scheduler 路由 | `scheduler.py` 的请求类型映射 |
| pause/flush/barrier/metrics | `SchedulerWeightUpdaterManager.update_weights_from_ipc` |
| ModelRunner 适配 | `ModelRunner.update_weights_from_ipc` |
| GPU UUID 与 ZMQ handle | `SGLangCheckpointEngineWorkerExtension.update_weights_from_ipc` |
| 外部脚本回调 | `update.py` 的 `check_sglang_ready`、`req_inference`、`update_weights` |

## 排障演练

不打开正文，试着回答这些问题：

- [ ] `/ping` 成功但 `_wait_weights_ready` 仍超时，最可能缺哪一步？
- [ ] POST 返回 400，response message 里应该优先找哪三类错误？
- [ ] 为什么 `--inference-parallel-size` 配错会表现为 GPU UUID mismatch？
- [ ] 为什么默认 `flush_cache=True` 是正确性保护？
- [ ] 为什么 `weight_load_duration_seconds` 不依赖 Scheduler stats tick？
- [ ] 为什么缺 checkpoint-engine 包不会影响普通冷启动？
- [ ] 为什么 post hook 失败可能不导致 HTTP update 失败？
- [ ] draft worker 更新失败时，主模型是否可能已经更新？
- [ ] `join` 路径和普通 `update_weights` 路径在 SGLang 侧是否使用同一个 HTTP endpoint？
- [ ] CheckpointEngine IPC 和 LoRA 热加载的 API、对象、cache 语义分别有什么不同？

## 最小运行验证

| 验证目标 | 操作 | 预期现象 |
|----------|------|----------|
| HTTP 可达 | 外部脚本轮询 `/ping` | 只证明 HTTP listen，不证明权重 ready |
| 权重 ready | 成功 POST `/update_weights_from_ipc` | `initial_weights_loaded` 被置为 True |
| DP 约束 | 多 DP 且未开 DP attention 调 IPC update | 返回包含 DP 约束的失败消息 |
| UUID 对齐 | 打印 `zmq_handles` keys 和 CUDA UUID | 当前 worker UUID 在 handles 中 |
| cache flush | 默认 update 后观察 cache hit | `cache_hit_rate` 可能下降 |
| IPC duration | 开启 metrics 后热更新 | 出现或更新 `weight_load_duration_seconds{source="ipc"}` |
| 缺包路径 | 未安装 checkpoint-engine 调用 IPC update | 返回 ImportError 风格失败消息 |
| post hook | 量化模型热更新后查日志 | 不应出现 `Post-hook processing failed` warning |
| draft worker | speculative 模型热更新 | draft runner 更新失败会让最终 success=false |

## 迁移复盘

如果你能完成以上自检，这个专题的核心模型就算建立起来：

1. CheckpointEngine 在 SGLang 侧是 serving 适配通道，不是 ParameterServer 本体。
2. 状态要拆成 HTTP 可达、权重 ready、warmup ready。
3. IPC HTTP request 传 handles；权重数据走外部 checkpoint-engine worker 的 ZMQ 通道。
4. WeightUpdater 把权重替换纳入 serving 正确性：metrics、flush、barrier、draft。
5. 大多数线上问题不是“权重文件坏”，而是 topology、UUID、依赖、post hook 或 cache 语义没对齐。

下一步建议把本专题和 [[SGLang-可观测性]]、[[SGLang-LoRA]]、[[Slime-分布式权重同步]] 连起来读。
