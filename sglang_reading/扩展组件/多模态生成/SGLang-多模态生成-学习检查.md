---
title: "多模态生成 · 学习检查"
type: exercise
framework: sglang
topic: "多模态生成"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---

# 多模态生成 · 学习检查

## 你为什么要做这组检查

目标不是记住文件名，而是能在一次真实故障中分清四类容易混淆的边界：HTTP 与 worker 进程、ZMQ 与 distributed broadcast、普通生成与 disagg、pipeline 计算与输出 transport。

## 一、闭卷主线

- [ ] 能说明 `multimodal_gen` 与文本 SRT 是并列运行时，而不是 SRT 的一个 ModelRunner backend。
- [ ] 能画出 HTTP/Vertex/broker → SchedulerClient → rank0 ROUTER → waiting queue → GPUWorker → OutputBatch。
- [ ] 能解释每个 rank 都构造 Scheduler/GPUWorker，但只有 rank0 创建 ROUTER receiver。
- [ ] 能指出普通生成请求按 SP/CFG/TP group 走 `broadcast_pyobj`，不是走父进程的 task Pipe。
- [ ] 能解释 ready pipe、控制 task Pipe、ZMQ REQ/ROUTER 与 torch distributed group 各自解决什么问题。
- [ ] 能说明动态合批的等待窗口、兼容性签名、admission、merge、split 与错误返回。
- [ ] 能区分 request warmup 与 server warmup，并说明合成 warmup为什么不是完整 HTTP 生成链。
- [ ] 能说明 `dp_size` 字段存在但当前 `dp_size > 1` 会被校验拒绝。
- [ ] 能解释 `return_raw_frames`、`return_frames`、`return_file_paths_only` 如何改变 OutputBatch transport。
- [ ] 能判断故障属于入口、队列、并行广播、pipeline stage、component residency 还是输出物化。

## 二、静态证据链

操作：

```powershell
rg -n "reader.recv|status.*ready|launch_http_server_only|launch_pool_disagg_server|dispatch_launch" sglang/python/sglang/multimodal_gen/runtime/launch_server.py
rg -n "receiver = None|waiting_queue|broadcast_pyobj|_try_merge_generation_reqs|_split_batched_output|return_result|_broadcast_task" sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py
rg -n "class AsyncSchedulerClient|temporary REQ socket|materialize_file_refs|run_zeromq_broker" sglang/python/sglang/multimodal_gen/runtime/scheduler_client.py
rg -n "return_raw_frames|return_file_paths_only|return_frames|execute_forward|build_pipeline|RealtimeSessionCache" sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
rg -n "dp_size > 1|server_warmup = False|batching_max_size|sp_degree.*ring_degree.*ulysses_degree" sglang/python/sglang/multimodal_gen/runtime/server_args.py
```

预期：能够证明 ready barrier 先于 HTTP；非 rank0 没有 ROUTER；普通请求的多 rank 同步使用 distributed broadcast；Pipe 有独立控制方法；异步 client 每请求创建 socket；DP 当前被拒绝。字符串命中只负责定位，最终判断仍需阅读函数上下文。

## 三、故障推演

### 推演 A：开启 `num_gpus=2` 后，slave 没收到普通请求

合格答案不能只检查 `mp.Pipe`。应先确认实际并行分解出的 `tp_size/sp_degree/cfg_parallel_degree`，再检查 `recv_reqs()` 对应的 `broadcast_pyobj` group、source rank 和 CPU group。Pipe 主要检查控制类任务，而非普通生成主线。

### 推演 B：两个看似相同的请求没有合批

检查 warmup/realtime、prompt 是否为字符串、是否带 image、`return_file_paths_only`、SamplingParams 中未标记 `batch_sig_exclude` 的字段，以及 `extra.diffusers_kwargs`。即使签名兼容，admission 仍可能因模型/分辨率规则提前判满。

### 推演 C：`/health` 已经 200，但生成接口一直等待

检查 `server_warmup_done`。控制面路径绕过 gate，所以 `/health` 成功不等于 synthetic warmup 已完成；warmup task 会直接调用 scheduler client，失败时当前进程收到 SIGTERM。

## 四、无依赖静态实验

在仓库根目录运行：

```powershell
@'
import ast
from pathlib import Path

root = Path("sglang/python/sglang/multimodal_gen/runtime")
scheduler = ast.parse((root / "managers/scheduler.py").read_text(encoding="utf-8"))
server_args = ast.parse((root / "server_args.py").read_text(encoding="utf-8"))

recv = next(n for n in ast.walk(scheduler) if isinstance(n, ast.FunctionDef) and n.name == "recv_reqs")
calls = [n.func.id for n in ast.walk(recv) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
print("broadcast_pyobj calls:", calls.count("broadcast_pyobj"))

validate = next(n for n in ast.walk(server_args) if isinstance(n, ast.FunctionDef) and n.name == "_validate_parallelism")
text = ast.get_source_segment((root / "server_args.py").read_text(encoding="utf-8"), validate)
print("DP rejected:", 'DP is not yet supported' in text)
'@ | python -
```

预期：当前基线打印 `broadcast_pyobj calls: 3` 与 `DP rejected: True`。该实验只证明静态控制流形状，不证明 NCCL/Gloo、GPU pipeline 或真实模型可运行。

## 复盘

主链见 [[SGLang-多模态生成-源码走读]]，跨进程对象见 [[SGLang-多模态生成-数据流]]。
