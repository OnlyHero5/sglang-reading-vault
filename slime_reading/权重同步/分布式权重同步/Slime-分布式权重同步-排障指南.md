---
title: "分布式权重同步 · 排障指南"
type: troubleshooting
framework: slime
topic: "分布式权重同步"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# 分布式权重同步 · 排障指南

## 排障总表

| 症状 | 优先看哪里 | 常见原因 |
|------|------------|----------|
| engine 仍用旧权重 | `weight_version`、`actor.update_weights` | update 未触发、engine 未完成 recv、CI 抽查失败 |
| NCCL broadcast hang | `rollout_engine_lock`、group rank 布局 | metadata/broadcast 顺序错、engine_gpu_counts 错 |
| 只有部分 PP stage 有进度条 | `_is_pp_src_rank` | 只有 DP=0 且 TP=0 的 rank 是 source，这是正常现象 |
| MoE 同步慢或 OOM | `_iter_expert_chunks` | EP 放大后 bucket 过大 |
| offload 后更新失败 | actor `wake_up/reconnect` | process group 被 sleep/destroy 后没重连 |
| 量化模型加载后异常 | `post_process_weights` | compressed-tensors 缺少前后处理 |
| 路径和预期不同 | actor updater 选型 | colocate/delta/disk 配置走了其他 updater |
| 一个大参数仍让 source OOM | buffer 是 flush 阈值，不会拆单个 conversion chunk | 比较最大 chunk 字节与配置值 |
| 一次异常后所有 PP source 卡在 acquire | lock 无 finally/timeout | 检查 lock owner，必要时重建 lock/worker |
| CI version 通过但仍有旧 engine | 只随机抽查一个 engine | 枚举全部 engine 的版本与请求日志 |

## 1. 我配置了 NCCL，为什么没有走本专题路径

NCCL 路径需要同时满足：非 colocate、`update_weight_mode=full`、`update_weight_transport=nccl`。任一条件不满足都会走其他 updater。

源码入口：来源：slime/backends/megatron_utils/actor.py L139-L168

源码入口：来源：slime/utils/arguments.py L1978-L2002

验证方法：

- 在 actor init 后打印 `type(self.weight_updater).__name__`。
- 若 `--colocate` 打开，预期是 `UpdateWeightFromTensor`。
- 若 `--update-weight-mode=delta`，必须是 disk 路径。
- 若 `--update-weight-transport=disk`，必须配置共享目录。

## 2. 为什么只有 DP-with-CP=0、TP=0 的 rank 发权重

PP stage 内，TP ranks 共同持有同一层的分片；DP/CP ranks 持有参数副本。源码查询的是 `get_data_parallel_rank(with_context_parallel=True)`，所以只有 combined DP×CP rank 0 且 TP rank 0 的进程代表该 PP stage 发 metadata 和 NCCL broadcast。

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L72-L92

验证方法：

- 看 `[slime-pp_i] Update weights` 进度条，只应出现在 PP source rank。
- 非 source rank 仍会进入 `all_gather_param`，不要误判为“没参与同步”。
- 多 PP stage 会有多个 `slime-pp_i` group。

## 3. NCCL hang 时先查什么

优先查顺序，而不是先怀疑 tensor 内容。

| 检查项 | 预期 |
|--------|------|
| `rollout_engine_lock` | metadata、broadcast、Ray refs 等待都在 lock 内 |
| `group_name` | 训练侧和 engine 侧相同 |
| `world_size` | `1 + sum(engine_gpu_counts)` |
| metadata 顺序 | `names/dtypes/shapes` 与 broadcast tensor 顺序一致 |
| engine 状态 | 已 pause，未被 continue 或重启打断 |

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L240-L265

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L326-L355

## 4. `update_weight_buffer_size` 太大或太小会怎样

`update_weight_buffer_size` 控制单个 bucket 的 tensor 字节上限。

| 设置 | 结果 |
|------|------|
| 太小 | bucket 数增加，lock、HTTP、NCCL launch 次数增加，同步变慢 |
| 太大 | PP source 和 engine 侧 recv buffer 峰值变高，可能 OOM |
| MoE 未留余量 | expert pass 乘 EP size 后超过预期 |
| 单参数大于阈值 | 仍形成超限 bucket，配置不是硬上限 |

源码入口：来源：slime/utils/arguments.py L513-L528

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L183-L202

验证方法：

- 调小 buffer，看 tqdm bucket 数是否增加。
- MoE 模型优先观察 expert pass 的 OOM 或 hang。
- 静态模型同步慢时，不要盲目放大 buffer，先看单 bucket 峰值显存。
- 记录 `convert_to_hf` 单次返回的总字节；若它已大于阈值，继续调小 buffer 不会拆开该 chunk。

## 5. 为什么 expert 权重单独同步

Expert 参数需要 EP all_gather。把 expert 和非 expert 混在同一 pass，会让普通参数同步被 EP 逻辑拖慢，也会让 bucket size 估算失真。

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L136-L146

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L204-L238

验证方法：

- 非 MoE 模型 expert pass 应为空或很快结束。
- MoE 模型如果卡在 expert pass，检查 EP group、expert names 对齐和 buffer。

## 6. 为什么需要 pause_generation 和 flush_cache

权重更新期间继续生成会产生两个风险：请求读取半更新参数，或者旧 KV 与新权重混用。Slime 在发送权重前暂停 generation 并 flush cache，发送完再继续。

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L107-L134

验证方法：

- engine 日志中应能看到 pause 和 continue 对应请求。
- 如果 update 异常后 engine 不再生成，检查是否停在 pause 之后、continue 之前。

源码没有 `finally` 保证 continue。恢复时还要检查 compressed-tensors 是否停在 restore-before-load 与 post-process-quantization 之间，不能只补发 continue。

## 7. offload_train + critic 为什么要 reconnect

PPO + critic 会强制 `offload_train=True`。actor sleep 时可能 disconnect rollout engines 并 destroy process groups；下一次 update 前要 wake up 并重建 NCCL group。

源码入口：来源：slime/backends/megatron_utils/actor.py L190-L220

源码入口：来源：slime/backends/megatron_utils/actor.py L601-L652

验证方法：

- `reconnect_rollout_engines = offload_train and use_critic and not colocate`。
- 该条件为真时，预期 update 前 `wake_up()`，之后 `sleep()`。
- 只开 offload_train 但不用 critic 时，预期 reload/destroy process groups，而不是完整 reconnect。

## 8. compressed-tensors 量化为什么多两次 post_process

compressed-tensors 量化模型的运行时 layout 和 Megatron 直传 tensor layout 不完全等价。Slime 在加载前后各调用一次 engine `post_process_weights`。

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L107-L134

源码入口：来源：slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py L358-L374

验证方法：

- `quantization_config["quant_method"] == "compressed-tensors"` 时，应出现 pre/post 两轮 engine post-process。
- 非量化模型不应走这两轮。

## 9. `weight_version` 不一致说明什么

CI 模式下，actor 更新后随机抽一个 rollout engine，比对 engine 版本和 updater 版本。如果失败，至少说明某个 engine 没进入新版本。

源码入口：来源：slime/backends/megatron_utils/actor.py L625-L636

源码入口：来源：slime/backends/sglang_utils/sglang_engine.py L363-L378

验证方法：

- 打开 `--ci-test`。
- 失败时先查 engine 是否收到 metadata，再查 NCCL broadcast 是否完成。
- fault tolerance 后如果新 engine 未同步，检查 `num_new_engines` 是否触发 connect。
- 不要只抽一个 engine；逐个调用 `get_weight_version` 才能证明全体一致。

## 10. fault tolerance 重连为什么可能残留旧 group

updater 在 reconnect 开头先覆盖 `self.rollout_engines`，随后若发现旧 process group 存在，disconnect RPC 使用的已经是新 engine 列表。若旧 actor 被真正替换而不是保留或扩展，旧 actor 上的 group 可能没有收到 destroy。

验证方法：记录重连前后 actor id、group name 与 destroy 请求接收者；替换拓扑时确认旧 actor 已退出，或显式向旧列表销毁 group。

## 11. Direct iterator 要不要画进 NCCL 主线

不要。`HfWeightIteratorDirect` 是 raw 模式下的 HF 权重迭代器，主要服务 checkpoint 保存等消费者。它和 NCCL 路径共享命名、转换、bucket 思想，但不是 `UpdateWeightFromDistributed.update_weights()` 的直接调用。

源码入口：来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_base.py L4-L15

源码入口：来源：slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py L19-L41

读者抓手：要理解本专题主线，看 `UpdateWeightFromDistributed`；要理解 raw HF 导出、PP/EP 跨 rank 补齐和 bucket metadata，再看 Direct iterator。

## 12. SGLang 侧出错该从哪个入口看

Slime 侧的 `SGLangEngine` 只是 HTTP 包装层。SGLang 侧真实入口是 `/init_weights_update_group`、`/update_weights_from_distributed`、`/pause_generation`、`/continue_generation`。

源码入口：来源：slime/backends/sglang_utils/sglang_engine.py L439-L517

交叉阅读：[[Slime-SGLang-Engine-数据流]]。

## 13. `update_weights_interval` 会改变什么

`update_weights_interval` 控制异步主循环中多久推一次权重；同步主循环当前每轮 train 后都会调用 `actor_model.update_weights()`。如果 keep-old-actor 打开，interval 还影响 `old_actor/rollout_actor` 备份队列。

源码入口：来源：slime/utils/arguments.py L523-L528

源码入口：来源：slime/backends/megatron_utils/actor.py L638-L648

验证方法：

- 同步主循环看 `train.py` 每轮 update。
- async 主循环看 `train_async.py` 中 `(rollout_id + 1) % args.update_weights_interval == 0`。
