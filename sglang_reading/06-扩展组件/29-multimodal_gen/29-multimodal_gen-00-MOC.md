---
type: module-moc
module: 29-multimodal_gen
batch: "29"
doc_type: moc
title: "multimodal_gen（扩散 / 多模态生成）"
tags:
 - sglang/batch/29
 - sglang/module/multimodal-gen
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# multimodal_gen（扩散 / 多模态生成）

> **源码范围：** `python/sglang/multimodal_gen/runtime/` — launch、gpu_worker、http_server、pipeline_executor、scheduler 
> **Git 基线：** `70df09b` 
> **前置专题：** [[28-Frontend-lang-00-MOC|28-Frontend-lang]]

---

## 1. 本模块目标

**Explain：** `multimodal_gen` 是 SGLang 的**扩散模型推理子系统**（图像/视频生成），架构借鉴 FastVideo 与 srt：多 GPU worker 进程 + FastAPI HTTP 入口 + ZMQ 调度 + PipelineStage 编排。与文本 LLM 的 srt 并行存在，共享部分工具（logging、network utils），但模型 forward 走 DiT/VAE/TextEncoder pipeline 而非 transformer decode loop。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/launch_server.py L17-L26
from sglang.multimodal_gen.runtime.entrypoints.http_server import create_app
from sglang.multimodal_gen.runtime.managers.gpu_worker import run_scheduler_process
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    prepare_server_args,
    set_global_server_args,
)
from sglang.multimodal_gen.runtime.utils.common import is_port_available
from sglang.multimodal_gen.runtime.utils.logging_utils import configure_logger, logger
from sglang.multimodal_gen.runtime.utils.trace_wrapper import init_diffusion_tracing
```

**Comment：**

- CLI 入口通常为 `sglang multimodal` 或模块 `launch_server`。
- `ServerArgs` 承载 pipeline 配置、并行度、disagg 角色等。

---

## 2. 架构位置

```
Client (OpenAI Images/Videos API)
 │ HTTP
 ▼
FastAPI http_server.py
 │ ZMQ SchedulerClient
 ▼
GPUWorker rank0 Scheduler.event_loop
 │ mp.Pipe / torch.distributed
 ▼
GPUWorker rank1..N + PipelineExecutor + ComposedPipeline
 │
 ▼
输出 tensor → 保存文件 / base64 响应
```

| 组件 | 职责 |
|------|------|
| `launch_server.py` | spawn worker、起 HTTP |
| `gpu_worker.py` | 单 GPU 模型加载与 forward |
| `managers/scheduler.py` | 收 ZMQ 请求、派 batch |
| `pipelines_core/` | Stage 编排（text encode → denoise → decode） |
| `entrypoints/openai/` | OpenAI 兼容 image/video API |

---

## 3. 验收标准

- [ ] 能说明 multimodal_gen 与 srt LLM 推理的进程模型差异
- [ ] 能追踪 HTTP 请求 → ZMQ → Scheduler → PipelineExecutor 路径
- [ ] 能解释多 GPU spawn 与 rank0 master/slave pipe 协调
- [ ] 五篇正文 ≥ 15 段 ETC，合计 ≥ 200 行内嵌源码

→ [[29-multimodal_gen-01-核心概念]] · [[29-multimodal_gen-02-源码走读]]
