---
type: batch-doc
module: 11-ModelRunner
batch: "11"
doc_type: faq
title: "ModelRunner：关键问题"
tags:
 - sglang/batch/11
 - sglang/module/model-runner
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# ModelRunner：关键问题

## Q1：为什么要有 TpModelWorker，不直接让 Scheduler 调 ModelRunner？

**Explain：** Worker 层封装 tokenizer、NCCL group、多 ModelRunner（EAGLE）、权重热更新 RPC 等，Scheduler 只关心 generation 语义。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L225-L227
# 提交版本：70df09b
class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

```

**易错：** 在 Scheduler 里直接 new ModelRunner 会漏掉 random_seed broadcast、pp_group 初始化等。

---

## Q2：prefill 和 decode 在 ModelRunner 里走同一条 forward 吗？

**Explain：** 都进 `ModelRunner.forward`，但 `ForwardMode` 不同：prefill 用 EXTEND/MIXED，走 Eager 或 prefill graph；decode 优先 CUDA Graph。

**Code（模式判断）：**

```python
# 来源：python/sglang/srt/model_executor/forward_batch_info.py L128-L161
# 提交版本：70df09b
    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_decode_or_idle(self):
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE

    def is_target_verify(self):
        return self == ForwardMode.TARGET_VERIFY

    def is_draft_extend_v2(self):
        # For fixed shape logits output in eagle v2 worker
        return self == ForwardMode.DRAFT_EXTEND_V2

    def is_extend_or_draft_extend_or_mixed(self, include_draft_extend_v2: bool = False):
        return (
            self == ForwardMode.EXTEND
            or self == ForwardMode.MIXED
            or self == ForwardMode.SPLIT_PREFILL
            or (include_draft_extend_v2 and self == ForwardMode.DRAFT_EXTEND_V2)
        )

    def is_cuda_graph(self):
        return (
            self == ForwardMode.DECODE
            or self == ForwardMode.TARGET_VERIFY
            or self == ForwardMode.IDLE
            or self == ForwardMode.DLLM_EXTEND
        )
```

**正确做法：** 调度层设置正确的 `batch.forward_mode`；错误设为 DECODE 做长 prefill 会导致 KV 索引错误。

---

## Q3：CUDA Graph replay 失败会怎样？

**Explain：** `_forward_raw` 回退 EagerRunner；`can_run_graph=False` 通知 Scheduler。常见原因：batch size 未 capture、dynamic shape、权重更新后未 recapture。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L335-L338
# 提交版本：70df09b
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
```

**易错 vs 正确：**

```python
# ❌ 权重热更新后忘记 recapture
update_weights_from_disk(..., recapture_cuda_graph=False) # decode 可能 silent wrong

# ✅ 生产环境热更新应 recapture
update_weights_from_disk(..., recapture_cuda_graph=True)
```

---

## Q4：ForwardBatch 和 ScheduleBatch 能否混用？

**Explain：** 不能。ScheduleBatch 在 CPU，含 Python Req 对象；ForwardBatch 是 GPU tensor 视图。必须 `init_new` 转换。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/forward_batch_info.py L19-L25
# 提交版本：70df09b
ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
```

---

## Q5：PP 下 logits 在哪产生？

**Explain：** 仅 `pp_group.is_last_rank` 的 ModelRunner 输出真实 logits；其他 rank 传递 `PPProxyTensors`。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L506-L507
# 提交版本：70df09b
        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
```

---

## Q6：overlap schedule 与 forward_stream

**Explain：** ModelRunner 维护专用 CUDA stream；overlap 时 Scheduler 在 forward 未完成时准备下一 batch，依赖 WAR barrier。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L73-L78
# 提交版本：70df09b
    @property
    def war_fastpath_runner(self):
        # The runner that runs the step's LAST shared-buffer-reading phase --
        # it owns the read-done event the scheduler's WAR barrier waits on.
        # For a plain worker that's its own runner.
        return self.model_runner
```

**Comment：** 禁用 overlap 时 `disable_overlap_schedule=True`，逻辑更简单但吞吐下降。

---

## Q7：embedding 请求走哪？

**Explain：** 不走 `forward_batch_generation`，而走 `forward_batch_embedding`，同样 `ForwardBatch.init_new` 但输出 embedding 而非 token。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L219-L222
# 提交版本：70df09b
    def forward_batch_embedding(self, batch: ScheduleBatch):
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        output = self.model_runner.forward(forward_batch).logits_output
        return output  # Returns EmbeddingPoolerOutput
```
