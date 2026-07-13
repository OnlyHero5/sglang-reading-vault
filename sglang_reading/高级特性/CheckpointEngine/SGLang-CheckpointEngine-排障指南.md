---
title: "CheckpointEngine · 排障指南"
type: troubleshooting
framework: sglang
topic: "CheckpointEngine"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# CheckpointEngine · 排障指南

## 读者任务

这篇按症状排障。先把现象归到启动等待、外部脚本、HTTP 控制面、Scheduler 执行面、worker extension、cache/metrics，再看源码入口。

## 快速症状表

| 症状 | 优先归类 | 第一入口 | 直接验证 |
|------|----------|----------|----------|
| `/ping` 一直失败 | 外部脚本可达性 | `check_sglang_ready` | endpoint、port、UDS |
| `/ping` 成功但权重仍未到 | 启动等待 | `_wait_weights_ready` | 状态位、剩余超时与随后是否继续 warmup |
| POST 返回 400 | 控制面或执行面 | HTTP endpoint、TokenizerManager、ModelRunner | response message |
| 报 `dp_size` 限制 | TokenizerManager 控制面 | `update_weights_from_ipc` | 单 DP 或启用 DP attention |
| DP-Attention 局部失败但 HTTP 200 | 控制面结果 | IPC communicator 的 `[0]` | 逐 scheduler 查日志，不只看首 response |
| 缺 checkpoint-engine 包 | ModelRunner integration | delayed import | 安装 `sglang[checkpoint-engine]` |
| `Device UUID not found` | worker extension | GPU UUID lookup | 对比 handles keys 与 CUDA UUID |
| 更新后 cache hit 下降 | cache flush | WeightUpdater flush | 默认行为，不一定是错误 |
| HTTP 失败但 duration 仍变化 | metrics | `_observe_weight_load` | 正常，gauge 记录尝试耗时 |
| draft worker 失败 | Scheduler 执行面 | WeightUpdater draft 分支 | 查看最终 message |
| post hook warning | 模型加载后处理 | worker post hook | 量化模块是否处理成功 |

## Q1：什么时候该用 wait-before-ready？

当你希望 SGLang 先占住 GPU 和 HTTP 端口，再由外部脚本灌入真实权重时使用。源码默认是关闭的。

```python
# 来源：python/sglang/srt/server_args.py L2505-L2508
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
    ] = False
```

适合：

- RL 在线 rollout，需要训练侧周期性推新 base weights。
- dummy load 启动，占 GPU 后等待 checkpoint-engine。
- 外部系统希望控制初始权重版本。

不适合：

- 普通模型冷启动。
- 只加载 LoRA adapter。
- 没有外部 update 脚本或 ParameterServer。

## Q2：`/ping` 成功为什么还不能推理？

`/ping` 只证明 HTTP 可达。等待模式下，warmup 前会轮询 `initial_weights_loaded=True`，但这不是无限等待或 fail-closed gate。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2178-L2184
    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)
```

验证动作：

- 看外部 POST `/update_weights_from_ipc` 是否成功。
- 看 HTTP endpoint 是否把 `initial_weights_loaded` 从 False 改成 True。
- 看是否出现 `Weights are not ready after waiting` 日志。
- 一旦出现超时日志，要立刻确认进程是否已经继续 warmup；当前函数不会抛错，不能假定服务仍被安全阻塞。
- 必要时调 `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT`，但更重要的是让 update 在超时前成功，并用真实生成验证权重不是 dummy/旧版本。

## Q3：POST `/update_weights_from_ipc` 返回 400，先看什么？

HTTP endpoint 失败时会返回 `success=False` 的 message。真正原因通常在 TokenizerManager、Scheduler、ModelRunner 或 worker extension。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1316-L1322
    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)
```

排查顺序：

- message 是否包含 `dp_size must be 1 or dp attention`。
- message 是否包含 ImportError。
- message 是否包含 `Device UUID not found` 这一类 UUID mismatch 信息。
- server log 是否有 post hook warning 或 draft worker failure。

## Q4：为什么 `dp_size > 1` 会失败？

TokenizerManager 明确限制：当前 IPC update 只支持单 DP，或开启 DP attention。

```python
# 来源：python/sglang/srt/managers/tokenizer_control_mixin.py L493-L498
        try:
            # For now, we only support single data parallel instance
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            logger.info("Starting IPC weight update")
```

这不是 Prometheus 问题，也不是外部 checkpoint 文件问题。先确认部署拓扑是否在支持范围内。

入口允许 DP-Attention 不等于控制面已经做全结果归并。communicator 会等齐所有 DP response，但 IPC 路径只取 `[0]`；如果后续 scheduler 失败，HTTP 仍可能依据第一个结果返回 success。生产验收必须逐 scheduler 查日志或补独立权重校验。

## Q5：缺 checkpoint-engine 包会怎样？

worker 模块导入第三方 `checkpoint_engine.worker.update_weights_from_ipc`。缺包时抛出带安装提示的 ImportError；ModelRunner 延迟 import，所以普通 serving 不受影响。

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L25-L31
try:
    from checkpoint_engine.worker import update_weights_from_ipc
except ImportError:
    raise ImportError(
        "checkpoint-engine is not installed. "
        "Please install it with: pip install sglang[checkpoint-engine]"
    )
```

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L3253-L3261
            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)
```

验证动作：

- 在运行环境安装 `sglang[checkpoint-engine]`。
- 确认外部 `update.py` 的 ParameterServer 依赖也存在。
- 重试 POST，观察 response message 是否仍是 ImportError。

## Q6：GPU UUID mismatch 怎么排？

worker extension 用当前 CUDA device 的 UUID 找 handle。请求里的 `zmq_handles` 必须包含这个 key。

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L77-L89
        device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        if device_uuid not in zmq_handles:
            raise ValueError(
                f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}"
            )
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=device_id,
            run=self.get_model_loader(),
            post_hook=self.get_post_hook(),
        )
```

常见原因：

- `--inference-parallel-size` 切出的 handle 集合没有覆盖该 endpoint 触达的全部 worker UUID；单机纯 TP 时常表现为与 TP 不一致。
- torchrun rank 与可见 GPU 顺序不一致。
- 多机部署里 endpoint 和 rank group 对不上。
- socket path 切片覆盖了别的 group。

验证动作：打印外部 `socket_paths` 的 keys，再在 SGLang worker 侧打印当前 CUDA UUID，必须完全匹配 `GPU-<uuid>` 字符串。

## Q7：为什么默认 `flush_cache=True`？

KV cache 是旧权重产生的中间状态。base weights 变了以后，继续复用旧 prefix cache 会造成错误输出。外部脚本默认传 `flush_cache=True`。

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L121-L130
                resp = client.post(
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
                    },
                    timeout=timeout,
```

SGLang 侧只在请求要求 flush 时执行。

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L101-L106
    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"
```

更新后 `cache_hit_rate` 下降是预期现象。真正要排查的是 flush 失败、更新前后输出不一致，或不该关闭 flush 却关闭了。

## Q8：热更新 metrics 应该看什么？

WeightUpdater 在 `_observe_weight_load("ipc")` 的 finally 里记录耗时。这个指标是边沿触发，不依赖周期性 stats tick，也不以 success 为条件。

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L86-L99
    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )
```

建议同时看：

- `weight_load_duration_seconds{source="ipc"}`：更新耗时。
- 不要用 `num_paused_reqs` 估算 IPC 影响：当前基线未找到它的递增生产者，series 可能长期为零。
- `cache_hit_rate`：flush 后下降是否符合预期。
- HTTP 400 rate 或 request log：控制面是否失败。

因此“duration 更新了”只能证明 update 尝试离开了 context manager，不能证明 target、draft、post hook 或所有 DP scheduler 都成功。

## Q9：draft worker 失败会怎样？

WeightUpdater 先更新 target TP worker。target 成功且存在 draft worker 时，再更新 draft。最终 success 可能被 draft worker 失败改成 false；target success 决定是否调用 flush helper，是否真正 flush 仍由请求开关决定。

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L169-L176
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
```

如果使用 EAGLE 或 multi-layer EAGLE，要看 draft runner 的 IPC update 路径。

```python
# 来源：python/sglang/srt/speculative/eagle_worker_v2.py L1666-L1672
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."
```

## Q10：post hook warning 是否可以忽略？

不要默认忽略。post hook 失败只写 warning，不会直接让 HTTP update 失败。

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L127-L141
                # Process quantization methods after loading weights
                for _, module in self.model_runner.model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        # Move parameters to device if needed for quantization processing
                        target_device = torch.device(
                            "cuda", torch.cuda.current_device()
                        )
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)
                # Call model-specific post-loading hook if available
                if hasattr(self.model_runner.model, "post_load_weights"):
                    self.model_runner.model.post_load_weights()
            except Exception as e:
                logger.warning(f"Post-hook processing failed: {e}")
```

对量化模型，post hook 失败可能意味着权重值更新了，但加载后格式、packing 或模型级 hook 没跑完整。验证要包含一次真实生成和日志检查。

## Q11：CheckpointEngine 和 LoRA 热加载有什么区别？

CheckpointEngine IPC 把 checkpoint-engine 提供的参数交给 base model `load_weights`；LoRA 热加载是 adapter 管理。二者不是同一个 API，也不是同一个 cache 语义。

| 维度 | CheckpointEngine IPC | LoRA |
|------|----------------------|------|
| 更新对象 | 进入 base model `load_weights` 的参数 | adapter weights |
| 控制入口 | `/update_weights_from_ipc` | `/load_lora_adapter` 等 |
| 请求体 | `zmq_handles` | adapter path / name |
| cache 影响 | base 变更后应 flush | adapter namespace 依赖 extra key |
| 典型场景 | RL rollout 权重同步 | 多租户 adapter serving |

两者都可能涉及 model update lock，但排障入口不同。LoRA 读 [[SGLang-LoRA]]。

## Q12：`update.py --update-method` 有哪些脚本级陷阱？

当前 argparse 没有给 `--update-method` 设置 `choices`。`broadcast` 触发一次无 ranks 的 `ps.update`，`p2p` 触发一次显式 ranks 更新，`all` 会先执行 broadcast，再等待两秒执行 p2p，也就是两次更新。未知字符串则两个分支都不进入，脚本可能没有发出任何 update 就结束。

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L161-L172
    if update_method == "broadcast" or update_method == "all":
        with timer("Update weights without setting ranks"):
            ps.update(checkpoint_name, req_func)

    if update_method == "p2p" or update_method == "all":
        if update_method:
            # sleep 2s to wait destroy process group
            time.sleep(2)
        with timer("Update weights with setting ranks"):
            ps.update(
                checkpoint_name, req_func, ranks=list(range(inference_parallel_size))
            )
```

验证动作：

- 明确只传 `broadcast`、`p2p` 或 `all`，不要依赖自由字符串被校验。
- 使用 `all` 时预期看到两段 timer 和两次 update，不要把第二次当重试异常。
- 脚本“正常退出”但 SGLang 没收到 POST 时，检查 `--update-method` 拼写和 server access log。

## Q13：`check_sglang_ready` 会不会永远等？

对连接拒绝和 HTTP 非成功状态，它没有最大重试次数，会持续轮询；但当前 `except` 只列出 `httpx.ConnectError` 与 `httpx.HTTPStatusError`，并不等于所有 timeout/transport 异常都会被吞掉重试。排障时既要防“永远等”，也要防未覆盖异常直接让脚本退出。

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L49-L71
def check_sglang_ready(
    endpoint: str, inference_parallel_size: int, uds: str | None = None
):
    rank = int(os.getenv("RANK", 0))
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    retry_num = 0
    transport = None
    if uds is not None:
        transport = httpx.HTTPTransport(uds=uds)
    with httpx.Client(transport=transport) as client:
        while True:
            try:
                response = client.get(f"{endpoint}/ping", timeout=10)
                response.raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                if retry_num % 10 == 0:
                    logger.warning(
                        f"fail to check sglang ready, retry {retry_num} times, error: {e}"
                    )
                retry_num += 1
                time.sleep(0.1)
```

验证动作：

- 给外层作业系统设置总超时与重试预算，不要只依赖脚本内部循环。
- 区分日志持续出现 retry、Python traceback 退出、以及已经进入 `ps.update` 三种状态。

## 复盘

- `/ping` 可达不是权重 ready；等待超时后也不会自动 fail closed。
- HTTP IPC update 失败要先读 response message。
- `inference_parallel_size`、endpoint 覆盖的 inference processes、GPU UUID 是一组一致性约束，不能只按单节点 TP 猜。
- 默认 flush cache 是正确性保护，不是性能 bug。
- checkpoint-engine 包是可选依赖，只在 IPC update 路径变成硬要求。
- post hook warning 对量化模型尤其重要。
- duration gauge、HTTP success、各 DP scheduler success 和真实生成正确性是四层不同证据。

下一篇 [[SGLang-CheckpointEngine-学习检查]] 用清单验收。
