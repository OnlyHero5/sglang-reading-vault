---
title: "多模态生成 · 排障指南"
type: troubleshooting
framework: sglang
topic: "多模态生成"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 多模态生成 · 排障指南

---

## 你为什么要读

多模态生成与文本 `srt` 共享仓库，却有独立的 server args、worker、Scheduler 和 pipeline。本文先确认你启动的是哪套 runtime，再沿 stage、rank 与 latent 生命周期排查，避免把扩散 pipeline 的错误送去文本 Scheduler 寻医。

## 1. multimodal_gen 与 srt 能否共进程？

**读法：** **不能默认共进程**。二者各有独立 `launch_server`、Scheduler、GPU worker 模型。同一机器可不同 port 各起一套服务，但共享 GPU 需手动分配 `CUDA_VISIBLE_DEVICES`。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L86-L90
def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """
```

**要点：**

- srt 用 `python -m sglang.launch_server`；扩散用 multimodal 入口。
- 仅共享 utilities（logging、orjson_response 等）。

---

## 2. 为什么 HTTP 与 Scheduler 分进程？

**读法：** FastAPI asyncio 进程与 CUDA worker 分离，避免 GIL 与 blocking forward 卡住 event loop。ZMQ 跨进程 REQ/REP 序列化 `Req`/`OutputBatch`。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L110-L112
    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    async_scheduler_client.initialize(server_args)
```

**要点：**

- 与 srt TokenizerManager ↔ Scheduler ZMQ 模式 analogous。
- 长生成（视频）需要大 ZMQ timeout（6000s RCVTIMEO）。

---

## 3. Rank0 master 挂了会怎样？

**读法：** Rank0 持有 ZMQ socket 与 HTTP client 连接。Master OOM/崩溃后 HTTP 全部失败；slave 无独立服务端口。K8s 应 liveness 整个 worker group。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L179-L185
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise
```

**要点：**

- 启动阶段 EOF 即 abort；运行期需外部 supervisor 重启。
- `kill_process_tree` 用于优雅关闭全部 worker。

---

## 4. CPU Offload 如何影响数据流？

**读法：** `PipelineExecutor.before_stage` 根据 `dit_cpu_offload` 等在 stage 前将 component 权重迁到 GPU，stage 后迁回 CPU，拉长单次请求 wall time 但降低峰值显存。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L171-L176
            if server_args.dit_cpu_offload and component_name in (
                "transformer",
                "transformer_2",
                "video_dit",
                "audio_dit",
            ):
```

**要点：**

- FSDP inference 路径需特殊 `inference_mode(False)` context。
- `OFFLOAD_DISABLE_RECOMMENDATION_ORDER` 建议 OOM 时关闭 offload 顺序。

---

## 5. broker 与 HTTP 能否同时用？

**读法：** **可以**。lifespan 始终启动 broker task；HTTP 与 offline client 并发连同一 Scheduler，Scheduler 需串行或内部 queue 化请求（实现于 `managers/scheduler.py`）。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L116-L117
    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))
```

**要点：**

- broker 用 pickle，注意版本兼容与安全（应仅 bind localhost）。
- 高 QPS 场景以 HTTP 为主，broker 适合 batch benchmark。

---

## 6. warmup 失败为什么要 SIGTERM？

**读法：** 未 warmup 的首请求可能极慢或 OOM，不如快速失败让 orchestrator 重启。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L98-L100
    except Exception as e:
        logger.error("Server warmup failed; aborting startup: %s", e, exc_info=True)
        os.kill(os.getpid(), signal.SIGTERM)
```

**要点：**

- `fail_open=True` 时分辨率列表未配置可跳过 fatal。
- 生产建议显式配置 `warmup_resolutions`。

---

## 7. disagg 与 multi-GPU TP 如何组合？

**读法：** 每个 role pool 内仍可 `num_gpus>1` TP；orchestrator 在 role 边界传递 intermediate tensor（embeddings/latents），而非完整像素。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L220-L223
    """Launch a pool-based disaggregated server with N:M:K independent role instances.

    DiffusionServer orchestrates the full pipeline, dispatching at every
    role transition (Encoder → Denoiser → Decoder).
```

**要点：**

- encoder 与 decoder 可共享物理 GPU（列表可重叠），靠 queue 分时。
- 类比 srt PD：compute 阶段拆分，非简单 replica。

---

## 8. 与 FastVideo 上游关系

**读法：** 多文件 header 标注 Copied from FastVideo；SGLang 在此基础上集成 OpenAI API、disagg、与 srt 生态工具。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L1-L4
# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
```

**要点：**

- 读 FastVideo 文档可辅助理解原始 pipeline 设计。
- SGLang 特有扩展：realtime、rollout_api、layerwise offload 等。

---

## 验证抓手

`multimodal_gen` 同时包含 HTTP 入口、scheduler 进程、broker、warmup、offload 和 disagg pool。排障时先用静态检索把问题归到正确层。

```powershell
rg -n "server_warmup_done|run_zeromq_broker|SIGTERM|broker_task|dit_cpu_offload|pool-based disaggregated|kill_process_tree|launch_scheduler_process" `
  sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py `
  sglang/python/sglang/multimodal_gen/runtime/launch_server.py `
  sglang/python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py
```

预期现象：

- `http_server.py` 命中 `server_warmup_done`、`SIGTERM`、`run_zeromq_broker` 和 `broker_task`，证明 warmup fatal、HTTP lifespan 与 offline broker 都在入口层。
- `launch_server.py` 命中 `kill_process_tree` 和 `pool-based disaggregated`，证明多进程生命周期和 pool disagg 是 launch 层职责。
- `pipeline_executor.py` 命中 `dit_cpu_offload`，证明 CPU offload 的权重迁移发生在 pipeline stage 执行前后。

运行期看到启动失败时，先看是 warmup 触发 `SIGTERM`，还是 scheduler 子进程 EOF；看到单次请求变慢时，先查 offload；看到 HTTP 与离线请求互相影响时，再回到 broker 与 scheduler queue 边界。
