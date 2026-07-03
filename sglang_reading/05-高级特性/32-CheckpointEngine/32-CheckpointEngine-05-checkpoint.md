---
type: batch-doc
module: 32-CheckpointEngine
batch: "32"
doc_type: checkpoint
title: "CheckpointEngine 验收清单"
tags:
 - sglang/batch/32
 - sglang/module/checkpoint-engine
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# CheckpointEngine 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 CheckpointEngine 运行时热更新的职责
- [ ] 能画出 ParameterServer → HTTP → Scheduler → ModelRunner 的数据流
- [ ] 能说出 3 个核心类/函数及其职责（FlattenedTensorBucket、SGLangCheckpointEngineWorkerExtensionImpl、update_weights_from_ipc）
- [ ] 能解释 wait_weights_before_ready 与 initial_weights_loaded 的关系
- [ ] 能说明热更新为何必须 flush_cache
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. checkpoint-engine 经 ZMQ IPC + `/update_weights_from_ipc` 实现运行时权重热更新，无需重启 Scheduler。
2. FlattenedTensorBucket 扁平化多 tensor 传输，metadata 保留重建信息；与 12-ModelLoader weight_sync 共用。
3. wait_weights_before_ready 允许 dummy 启动后外部灌权重，再 warmup serving；metrics 见 31-Observability。

## CheckpointEngine 核心函数/类清单

| 符号 | 职责 |
|------|------|
| `FlattenedTensorBucket` | 多 tensor 扁平化为单 buffer |
| `FlattenedTensorMetadata` | 重建 name/shape/dtype/offset |
| `SGLangCheckpointEngineWorkerExtension` | checkpoint-engine IPC 接口 |
| `SGLangCheckpointEngineWorkerExtensionImpl` | ModelRunner 具体集成 |
| `ModelRunner.update_weights_from_ipc` | Scheduler 侧 load 入口 |
| `weight_updater.update_weights_from_ipc` | pause/flush/metrics 包装 |
| `update_weights_from_ipc` (HTTP) | TokenizerManager 入口 |
| `_wait_weights_ready` | 启动等待权重屏障 |
| `check_sglang_ready` | update.py 轮询 /ping |
| `req_inference` / `update_weights` | 外部 ParameterServer 脚本 |
| `UpdateWeightsFromIPCReqInput` | HTTP body 结构 |

## 遗留问题

- checkpoint-engine 外部 ParameterServer 协议细节见 MoonshotAI/checkpoint-engine 仓库
- join 模式（load 预存 metas）生产用法较少，可按需补充示例
