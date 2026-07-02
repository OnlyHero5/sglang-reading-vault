---
type: module-moc
module: 32-CheckpointEngine
batch: "32"
doc_type: moc
title: "CheckpointEngine 运行时权重热更新"
tags:
 - sglang/batch/32
 - sglang/module/checkpoint-engine
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# CheckpointEngine 运行时权重热更新

> **阶段 V · 高级特性** | Git：`70df09b` 
> **源码范围：** `srt/checkpoint_engine/`、`srt/weight_sync/tensor_bucket.py`、`entrypoints/http_server.py`（`/update_weights_from_ipc`、`_wait_weights_ready`） 
> **前置专题：** [[12-ModelLoader-00-MOC]] §weight_sync · [[31-Observability-00-MOC]]（weight_load metrics）

---

## 1. 本模块目标

**Explain：** 生产环境需要在**不重启服务**的情况下更新 base 权重：训练侧 checkpoint-engine 在独立 torchrun 进程加载新权重，经 ZMQ IPC 推送到已运行的 SGLang server。本模块覆盖 `FlattenedTensorBucket` 扁平化传输、`SGLangCheckpointEngineWorkerExtension` IPC 桥接、`wait_weights_before_ready` 启动屏障，以及外部 `update.py` ParameterServer 脚本。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1306-L1320
@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(
    obj: Annotated[UpdateWeightsFromIPCReqInput, Body()], request: Request
):
    """Update the weights from IPC (Inter-Process Communication) for checkpoint-engine integration."""
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
```

**Comment：**

- 首次成功更新设置 `initial_weights_loaded=True`，解除 wait_weights 屏障。
- 需 admin auth（可选）；flush_cache 默认 true，热更新后 prefix cache 清空。
- metrics 见 [[31-Observability-04-关键问题|31-Observability]] Q4。

---

## 2. 与 12-ModelLoader 的关系

**Explain：** `weight_sync/tensor_bucket.py` 被 ModelLoader 与 checkpoint-engine 共用；IPC 路径在 `weight_updater.update_weights_from_ipc` 调用 `model_runner.update_weights_from_ipc`。冷启动 disk load 见 12-ModelLoader，本模块聚焦**运行时热更新**语义。

| 能力 | 12-ModelLoader | 32-CheckpointEngine |
|------|----------------|---------------------|
| 冷启动 load | ✓ disk/HF | dummy + wait |
| 热更新 disk | update_weights_from_disk | |
| 热更新 IPC | weight_sync 共用 bucket | ✓ checkpoint-engine 协议 |
| tensor_bucket | ✓ | ✓ 本模块详解 |

---

## 3. 文档导航

| 文件 | 内容 |
|------|------|
| [[32-CheckpointEngine-01-核心概念]] | 热更新术语、wait_weights_before_ready、FlattenedTensorBucket |
| [[32-CheckpointEngine-02-源码走读]] | checkpoint_engine_worker、tensor_bucket、update.py |
| [[32-CheckpointEngine-03-数据流与交互]] | ParameterServer → HTTP → Scheduler → ModelRunner |
| [[32-CheckpointEngine-04-关键问题]] | broadcast vs p2p、ready 探针、flush_cache |
| [[32-CheckpointEngine-05-checkpoint]] | 验收清单 |

---

→ 关联：[[31-Observability-00-MOC|31-Observability]] · [[07-总结与索引-00-MOC|07-总结与索引]]
