---
title: "多模态生成"
type: map
framework: sglang
topic: "多模态生成"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# 多模态生成

> **源码范围：** `python/sglang/multimodal_gen/runtime/` — launch、gpu_worker、http_server、pipeline_executor、scheduler 
> **Git 基线：** `70df09b` 
> **前置专题：** [[SGLang-前端语言]]

---

## 1. 本模块目标

专题读法：`multimodal_gen` 是 SGLang 的**扩散模型推理子系统**（图像/视频生成），架构借鉴 FastVideo 与 srt：多 GPU worker 进程 + FastAPI HTTP 入口 + ZMQ 调度 + PipelineStage 编排。与文本 LLM 的 srt 并行存在，共享部分工具（logging、network utils），但模型 forward 走 DiT/VAE/TextEncoder pipeline 而非 transformer decode loop。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L17-L26
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

读法：

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

## 3. 自测与验收标准

- [ ] 能说明 multimodal_gen 与 srt LLM 推理的进程模型差异
- [ ] 能追踪 HTTP 请求 → ZMQ → Scheduler → PipelineExecutor 路径
- [ ] 能解释多 GPU spawn 与 rank0 master/slave pipe 协调
- [ ] 能为 HTTP、Scheduler 和 PipelineExecutor 各指出一个源码入口，并用一次静态或运行验证确认对象确实沿图中路径流动

→ [[SGLang-多模态生成-核心概念]] · [[SGLang-多模态生成-源码走读]]
