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
updated: 2026-07-10
---
# CheckpointEngine · 排障指南

## 读者任务

这篇按症状排障。先把现象归到启动等待、外部脚本、HTTP 控制面、Scheduler 执行面、worker extension、cache/metrics，再看源码入口。

## 快速症状表

| 症状 | 优先归类 | 第一入口 | 直接验证 |
|------|----------|----------|----------|
| `/ping` 一直失败 | 外部脚本可达性 | `check_sglang_ready` | endpoint、port、UDS |
| `/ping` 成功但服务一直 not ready | 启动等待 | `_wait_weights_ready` | `initial_weights_loaded` 是否变 True |
| POST 返回 400 | 控制面或执行面 | HTTP endpoint、TokenizerManager、ModelRunner | response message |
| 报 `dp_size` 限制 | TokenizerManager 控制面 | `update_weights_from_ipc` | 单 DP 或启用 DP attention |
| 缺 checkpoint-engine 包 | ModelRunner integration | delayed import | 安装 `sglang[checkpoint-engine]` |
| `Device UUID not found` | worker extension | GPU UUID lookup | 对比 handles keys 与 CUDA UUID |
| 更新后 cache hit 下降 | cache flush | WeightUpdater flush | 默认行为，不一定是错误 |
| `weight_load_duration_seconds` 没变 | metrics | `_observe_weight_load` | 是否进入 WeightUpdater |
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

`/ping` 只证明 HTTP 可达。等待模式下，warmup 前还要等 `initial_weights_loaded=True`。

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
- 必要时调 `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT`，但更重要的是确认 update 真的成功。

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

- `--inference-parallel-size` 和 SGLang TP 不一致。
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

WeightUpdater 在 `_observe_weight_load("ipc")` 的 finally 里记录耗时。这个指标是边沿触发，不依赖周期性 stats tick。

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
- `num_paused_reqs`：热更新影响的请求数量。
- `cache_hit_rate`：flush 后下降是否符合预期。
- HTTP 400 rate 或 request log：控制面是否失败。

## Q9：draft worker 失败会怎样？

WeightUpdater 先更新主 TP worker。主 TP 成功且存在 draft worker 时，再更新 draft。最终 success 可能被 draft worker 失败改成 false，但 cache flush 由主 TP 成功决定。

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

CheckpointEngine IPC 替换 base model 全量权重；LoRA 热加载是 adapter 管理。二者不是同一个 API，也不是同一个 cache 语义。

| 维度 | CheckpointEngine IPC | LoRA |
|------|----------------------|------|
| 更新对象 | base model weights | adapter weights |
| 控制入口 | `/update_weights_from_ipc` | `/load_lora_adapter` 等 |
| 请求体 | `zmq_handles` | adapter path / name |
| cache 影响 | base 变更后应 flush | adapter namespace 依赖 extra key |
| 典型场景 | RL rollout 权重同步 | 多租户 adapter serving |

两者都可能涉及 model update lock，但排障入口不同。LoRA 读 [[SGLang-LoRA]]。

## 复盘

- `/ping` 可达不是权重 ready。
- HTTP IPC update 失败要先读 response message。
- `inference_parallel_size`、TP、GPU UUID 是一组一致性约束。
- 默认 flush cache 是正确性保护，不是性能 bug。
- checkpoint-engine 包是可选依赖，只在 IPC update 路径变成硬要求。
- post hook warning 对量化模型尤其重要。

下一篇 [[SGLang-CheckpointEngine-学习检查]] 用清单验收。
