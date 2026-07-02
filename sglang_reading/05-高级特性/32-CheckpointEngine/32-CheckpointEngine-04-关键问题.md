---
type: batch-doc
module: 32-CheckpointEngine
batch: "32"
doc_type: faq
title: "CheckpointEngine：关键问题"
tags:
 - sglang/batch/32
 - sglang/module/checkpoint-engine
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# CheckpointEngine：关键问题

## Q1：何时用 wait_weights_before_ready？

**Explain：** 需要**先起 engine、后灌权重**时用：例如 `--load-format dummy` 占位内存，训练 checkpoint 就绪后由 update.py 推送。不用此 flag 则 launch 后立即 load disk 权重并 warmup。RLHF online rollout 是典型场景。

**Code：**

```python
# 来源：python/sglang/srt/server_args.py L2505-L2508
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
    ] = False
```

**Comment：**

- 必须配套外部 update 脚本。
- 超时未灌权重会 error 但仍可能部分 listen（HTTP 已 up，warmup 未执行）。

---

## Q2：checkpoint-engine 包未安装会怎样？

**Explain：** import checkpoint_engine.worker 失败时 checkpoint_engine_worker 模块 raise ImportError；ModelRunner.update_weights_from_ipc 捕获后返回 success=False。普通推理不受影响，仅热更新路径需要该依赖。

**Code：**

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

**Comment：**

- update.py 同样依赖 ParameterServer。
- pip install sglang[checkpoint-engine] 安装 MoonshotAI 包。

---

## Q3：热更新为何 flush_cache？

**Explain：** 权重变化后 radix prefix KV 与旧权重不一致，必须 flush；否则 cache hit 返回错误 token。POST body `flush_cache: true` 默认开启。flush 后 cache_hit_rate metrics 骤降属预期（见 15-RadixAttention）。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L122-L128
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
```

**Comment：**

- flush 在 weight_updater.flush_cache_after_weight_update 执行。
- 高 QPS 热更新需评估 flush 对 latency 的冲击。

---

## Q4：broadcast vs p2p update_method？

**Explain：** update.py 支持 broadcast（默认）、p2p、all；broadcast 不设置 ranks，p2p 指定 inference_parallel rank 列表；all 两种都跑用于测试。p2p 前 sleep 2s 等待 destroy process group。

**Code：**

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

**Comment：**

- inference_parallel_size 必须与 server TP 一致。
- 生产默认 broadcast。

---

## Q5：与 31-Observability 的 metrics 对照

**Explain：** 监控热更新应同时看：`sglang:num_paused_reqs`、`sglang:weight_load_duration_seconds{source="ipc"}`、`sglang:cache_hit_rate`（flush 后）。冷启动另有 `sglang:engine_load_weights_time`。详见 [[31-Observability-00-MOC|31-Observability]]。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1143-L1149
    def observe_weight_load(self, duration_seconds: float, source: str) -> None:
        # Edge-triggered: engine is paused during the update, so log_stats
        # won't fire — write the gauge inline at end of update_weights_from_*.
        # `source` is "disk" | "distributed" | "tensor" | "ipc".
        self.weight_load_duration_seconds.labels(**self.labels, source=source).set(
            duration_seconds
        )
```

**Comment：**

- weight_load 是 edge-triggered，不依赖 log_stats tick。
- initial_weights_loaded 无直接 metric，需靠 /ping + 业务 probe。

---

## Q6：GPU UUID 不匹配怎么排障？

**Explain：** zmq_handles key 必须包含每张卡的 `GPU-{cuda uuid}`；update.py 的 inference_parallel_size 切片必须与 server TP 一致。错误日志 `Device UUID ... not found in zmq_handles` 指向配置 mismatch。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L102-L108
    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # Get device UUID for current device
        device_id = torch.cuda.current_device()
        try:
            return f"GPU-{torch.cuda.get_device_properties(device_id).uuid!s}"
        except AssertionError as e:
```

**Comment：**

- 多机场景需确保 ParameterServer rank 与 GPU 映射正确。
- 可先 log zmq_handles keys 与 server 侧 UUID 对比。

---

## Q7：与 LoRA 热加载的区别？

**Explain：** LoRA 通过 `/load_lora_adapter` 等 API 加载 adapter 权重，不替换 base model；checkpoint-engine IPC 替换 **全量 base weights**。两者可共存但操作互斥时需 model_update_lock。LoRA 见 12-ModelLoader LoRA 章节。

**Comment：**

- flush_cache 对两者均适用（KV 与权重绑定）。
- EAGLE draft model 在 update_weights_from_ipc 中单独更新。

---

## 附录：12-ModelLoader 交叉引用

`FlattenedTensorBucket`、`weight_updater`、distributed/tensor 热更新路径详见 **[[12-ModelLoader-00-MOC]]** §weight_sync；Observability metrics 详见 **[[31-Observability-04-关键问题|31-Observability]] Q4**。
