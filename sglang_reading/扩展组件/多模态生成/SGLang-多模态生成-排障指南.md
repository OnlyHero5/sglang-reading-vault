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
updated: 2026-07-12
---

# 多模态生成 · 排障指南

## 你为什么要读

扩散服务一次请求跨过父进程、HTTP、ZMQ、Scheduler queue、distributed group、pipeline stage 与输出 transport。最快的排障方法不是从 traceback 里猜模型，而是先回答：请求最后一次被谁确认接收、是否已 dispatch、是否已形成 OutputBatch、是否只在返回阶段失败。

## 快速分层

| 现象 | 第一责任层 | 先查 |
|---|---|---|
| HTTP 端口未监听 | 启动 barrier | 各 rank ready/EOF、模型加载、distributed init |
| `/health` 200，生成接口等待 | server warmup gate | `warmup_done`、synthetic scheduler request |
| rank0 收到请求，其他 rank不动 | distributed broadcast | TP/SP/CFG group、source rank、CPU group |
| 延迟出现固定小台阶 | Scheduler queue | batching delay、admission、兼容拒绝原因 |
| 合批后每个请求都报相同错误 | merge/forward/split | merged error、第一维、output paths 数量 |
| GPU forward 已完成但 client timeout | reply transport | pickle、local spill、send、client materialize |
| 文件生成了但 HTTP 返回 error | HTTP 后处理 | 文件读取/base64、路径可见性、编码 |
| disagg role 无 HTTP | 角色设计 | role 类型与 head server 地址，而非 HTTP port |

## 1. 服务没有监听 HTTP

**症状：** 父进程日志停在启动阶段，或某个 rank 报 EOF/exit code。

**可能原因：** 任一 Scheduler 构造失败。由于 Scheduler 构造会加载 GPUWorker/pipeline，模型路径、OOM、distributed init、port 或配置校验都可能在 ready 前失败。

**源码入口：** `launch_server.py::launch_server`、`gpu_worker.py::run_scheduler_process`。

**操作：** 按 rank 找最后一条日志；区分 `pipe_writer.send(status=ready)` 之前还是之后退出。检查最小空闲 GPU，而非只看所有卡总显存。

**预期：** 所有 rank 都返回 ready 后才出现 “All workers are ready” 并进入 HTTP 启动。只看到进程 PID 不算 ready。

## 2. `/health` 正常，但用户路由一直等待

**症状：** `/health`、`/model_info` 可访问，image/video generation 卡住。

**可能原因：** server warmup middleware 正在等待 `server_warmup_done`。控制面路径刻意 bypass，所以 health 200 不代表 warmup完成。

**源码入口：** `http_server.py::_run_server_warmup_after_http_ready` 与 `create_app` middleware。

**操作：** 查 warmup task 是否已经调用 `async_scheduler_client.forward`，再看 Scheduler 是否收到 `is_warmup` 请求。不要重复测试 `/health_generate`：当前实现尚未执行真实生成，只固定返回 ok。

**预期：** synthetic request成功后 event set，普通路由立即放行；失败则当前 HTTP 进程收到 SIGTERM，而不是永久留在半 warm 状态。

## 3. 多 GPU 时 slave 没执行请求

**症状：** rank0 ROUTER 收到请求，其他 rank无对应 forward，随后 collective hang。

**可能原因：** TP/SP/CFG group 分解或 `broadcast_pyobj` source/CPU group异常。普通生成不是经 task Pipe 分发。

**源码入口：** `scheduler.py::recv_reqs`。

**操作：** 打印最终 `tp_size/sp_degree/ulysses_degree/ring_degree/cfg_parallel_degree`；确认每个条件分支在所有相关 rank 上一致进入。检查 `sp_degree == ring_degree * ulysses_degree` 与 GPU 整除约束。

**预期：** rank0 从 ROUTER获得非空对象，其他 rank通过对应 broadcast得到同一逻辑请求。task Pipe 正常不能证明这条主线正常。

## 4. 两个请求为什么没有合批

**症状：** queue 同时存在多个请求，但 dispatch size 始终为 1。

**可能原因：** warmup、realtime、image condition、prompt 非字符串、path-only 不同、SamplingParams 签名不同、`diffusers_kwargs` 不同，或 admission 提前判满。

**源码入口：** `_can_dynamic_batch`、`_get_dynamic_batch_reject_reason`、`BatchAdmissionController`。

**操作：** 开启 batching metrics；比较首个 reject reason，而不是只比较 width/height。确认 `batching_max_size > 1` 且 pipeline config 宣称支持 dynamic batching。

**预期：** compatible 请求在 delay 到期、达到 max size 或 admission 满时合并；不兼容队列项可以被跳过并留待后续，不要求物理相邻。

## 5. batching delay 为什么放大尾延迟

**症状：** 低 QPS 时每个请求都多出接近 `batching_delay_ms` 的等待。

**可能原因：** 队首兼容且尚未到 user/admission 上限，Scheduler 有意等待更多候选。

**源码入口：** `get_next_batch_to_run` 中 `should_wait_for_more` 与 event loop poll/sleep。

**操作：** 对照 queue wait metrics、dispatch size 和 stop_reason；用 `batching_delay_ms=0` 做控制组。

**预期：** 零 delay 基本立即派发；增加 delay 后吞吐可能提高，但低流量 queue wait 会可见增加。没有 workload 和硬件时不要给固定推荐阈值。

## 6. 合批失败后为什么没有顺序重试

**症状：** 一次 merged forward error 后，所有原请求立即收到同类错误。

**可能原因：** 当前策略只在 merge 前不兼容时顺序执行；merged forward 已运行后，error 或 split 失败直接构造逐请求错误。

**源码入口：** `_handle_generation`、`_build_dynamic_batch_error_outputs`、`_split_batched_output`。

**操作：** 区分日志 “merge returned None” 与 “Dynamic batch execution returned error / could not split”。检查 output/output_file_paths 第一维是否等于 `sum(num_outputs_per_prompt)`。

**预期：** 前者可顺序执行；后者不二次生成。不要把没有重试误判成 Scheduler crash。

## 7. CPU/layerwise offload 后延迟异常

**症状：** 显存下降但每个 stage 前后出现明显搬运或同步开销。

**可能原因：** component residency、layerwise prefetch、FSDP version counter context 或相互覆盖的 offload配置。

**源码入口：** `PipelineExecutor.before_stage/_stage_execution_context`、ServerArgs `_adjust/_validate_offload`。

**操作：** 记录具体 active component、residency manager动作和 H2D/D2H 时间；检查 layerwise selection 是否自动关闭同 component 的普通 CPU offload，DiT layerwise 是否禁用了 FSDP。

**预期：** 每个选项的实际 owner清楚可见。`before_stage` 命中只证明委托发生，不证明某个 component 必然迁移。

## 8. forward 完成但 client 仍超时

**症状：** GPU metrics 显示 pipeline结束，client 在 REQ receive超时。

**可能原因：** raw/frame物化、文件保存、local array spill、pickle、ZMQ send或client端 `materialize_file_refs` 失败。100 分钟 RCVTIMEO 不会取消后端任务。

**源码入口：** GPUWorker `_materialize_output_transport`、Scheduler `return_result`、scheduler_client `_materialize_output_batch_file_refs`。

**操作：** 分别观察 `Scheduler.return_result.spill_arrays/pickle/send` metrics；检查临时目录容量和权限；确认 reply identity仍对应活着的 REQ socket。

**预期：** 模型完成时间与响应完成时间可被拆开。若文件已产生而 client断开，系统不会自动删除资产或回滚计算。

## 9. broker 与 HTTP 互相影响

**症状：** 离线 job 加入后，HTTP queue wait或吞吐变化。

**可能原因：** broker 与 HTTP 都通过同一个 async client/context接到同一 Scheduler queue；它们不是独立 backend。broker自身是单 REP loop，一次 recv→forward→reply 后才处理下一离线 job。

**源码入口：** `http_server.py::lifespan`、`scheduler_client.py::run_zeromq_broker`。

**操作：** 以 request id区分入口，观察统一 waiting queue；不要把 localhost bind理解成资源隔离。

**预期：** 两类流量共享调度容量。关闭 HTTP lifespan 后 broker也随之不存在；仅 `launch_http_server=False` 不会提供 broker服务。

## 10. Disagg role 启动后没有 HTTP

**症状：** encoder/denoiser/decoder role worker ready，但指定 HTTP port不可访问。

**可能原因：** standalone role 本来就不服务 HTTP；只有 monolithic 或 server role需要 HTTP。role 使用 work/result endpoint 与 head DiffusionServer通信。

**源码入口：** `launch_server.py::dispatch_launch/launch_disagg_role`、ServerArgs `_adjust_network_ports/_adjust_warmup`。

**操作：** 核对 `disagg_role`、`disagg_server_addr`、derived work/result endpoint、内部 scheduler port与物理 GPU映射。确认 head server已启动并绑定 frontend/result sockets。

**预期：** role进程进入 `_disagg_event_loop`；server warmup被关闭；客户端只访问 head server，而不是直接访问 role HTTP。

## 11. 配置里有 `dp_size`，为什么启动仍失败

**症状：** 设置 `dp_size=2` 后参数校验抛出 “DP is not yet supported”。

**原因：** 当前基线保留 DP字段和部分派生公式，但明确拒绝大于 1。接口表面与可运行能力不同步。

**操作：** 先保持 `dp_size=1`，用当前支持的 TP/SP/CFG 或多个外部 replica扩展；不要绕过校验后宣称 DP可用。

**预期：** 配置回到合法组合后通过 `_validate_parallelism`；真实水平扩容由外部路由与多个服务实例承担。

## 验证抓手

```powershell
rg -n 'status.*ready|reader.recv|launch_http_server_only' sglang/python/sglang/multimodal_gen/runtime/launch_server.py sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
rg -n 'receiver = None|broadcast_pyobj|waiting_queue|reject_reason|should_wait_for_more|Dynamic batch|return_result|spill_arrays' sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py
rg -n 'server_warmup_done|SIGTERM|health_generate|run_zeromq_broker' sglang/python/sglang/multimodal_gen/runtime/entrypoints/http_server.py
rg -n 'DP is not yet supported|server_warmup = False|sp_degree.*ring_degree.*ulysses_degree' sglang/python/sglang/multimodal_gen/runtime/server_args.py
```

静态预期是能把每个症状定位到唯一所有者。真实数值、NCCL collective、模型兼容、文件编码和 disagg transfer 仍需目标 Linux/GPU 环境故障注入。
