---
title: "LoRA · 学习检查"
type: exercise
framework: sglang
topic: "LoRA"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# LoRA · 学习检查

## 读者能做什么

- [ ] 能画出 `LoRARegistry → Req.extra_key → Scheduler → ForwardBatch → LoRAMemoryPool → LoRABatchInfo → LoRA layer` 这条主线。
- [ ] 能说清 `LoRARegistry`、`LoRAManager`、`LoRAMemoryPool`、`LoRABackend`、LoRA 包装层的边界。
- [ ] 能解释为什么 `lora_id` 必须进入 `Req.extra_key`，以及它如何隔离 prefix cache。
- [ ] 能区分 `max_loaded_loras` 和 `max_loras_per_batch`：前者主要是 registry 上限且 overlap 时还约束 CPU pinned 集合，后者是 unique uid/GPU slot 上限。
- [ ] 能说明动态 unload 为什么要先 unregister、再等 counter 归零、最后通知后端。
- [ ] 能解释 overlap loading 和 drainer 都不改变 slot 容量；overlap 使用 CUDA event，drainer 对短请求仍有 1.2 tolerance。
- [ ] 能说明显式 unload 后 `lora_ref_cache` 为何仍可触发隐式 reload，以及动态 `lora_id` 为什么会变化。
- [ ] 能列出动态更新的四份非事务状态：registry/counter、历史引用、worker CPU 字典、GPU pool。

## 源码定位自测

- [ ] 请求带 adapter 但报未启用时，能定位到 `TokenizerManager._validate_and_resolve_lora`。
- [ ] adapter 动态加载失败时，能定位到 `TokenizerControlMixin.load_lora_adapter` 和 `LoRAManager.validate_new_adapter`。
- [ ] batch 里 adapter 太多时，能定位到 `Scheduler._can_schedule_lora_req` 与 `LoRAManager.validate_lora_batch`。
- [ ] 输出疑似串 adapter 时，能定位到 `Req.__init__` 的 `extra_key` 拼接。
- [ ] GPU slot 换出异常时，能定位到 `LoRAMemoryPool.prepare_lora_batch` 的 candidate 选择。
- [ ] MoE LoRA shape mismatch 时，能定位到 `LoRAMemoryPool` 的 `moe_tp_size` / `moe_ep_size` 初始化逻辑。
- [ ] lm_head + input logprobs 分块崩溃时，能定位 Triton backend 的局部/成员 pass list 所有权错误。

## 可观测验证

**操作：** 依次执行下表实验，记录请求的 adapter 身份、scheduler 准入结果与 GPU slot 变化。

**预期：** 观察结果应符合表中现象；若不一致，先判断问题发生在 registry、batch 准入还是 memory pool，而不是直接归咎于 LoRA kernel。

| 实验 | 预期现象 |
|------|----------|
| 不带 `lora_path` 请求 | `ForwardBatch.lora_ids` 对应项为 `None`，走 base slot |
| 带已加载 adapter 请求 | `LoRARegistry.acquire` 返回内部 `lora_id`，`Req.extra_key` 追加该 ID |
| 带从未加载 adapter 请求 | 请求在控制面失败，不进入 scheduler |
| 本轮需要的 unique uid（含 base `None`）超过 `max_loras_per_batch` | 部分请求留在 waiting queue，或 overlap loading 等待后续轮次 |
| 调用 unload 时仍有长请求在跑 | unload 等待 counter 归零后才通知后端 |
| strict loading 加载 target module 不匹配 adapter | strict 模式更早失败；关闭 strict 可能先 warning，但当前实现后续仍可能因同一未知权重名失败 |
| backend load 成功后注入 registry register 失败 | API 失败，但 backend 状态可能已改变；不存在自动补偿回滚 |
| lm_head LoRA + 长 prompt input logprobs 分块 | 当前 Triton 基线可能在 per-pass list append 处触发 `NoneType` 异常 |

## 复盘问题

1. 如果只改 `LoRAMemoryPool` 的 eviction policy，是否会影响 `LoRARegistry` 的 `max_loaded_loras` LRU？为什么？
2. 如果一个 adapter 已经 register，但 `LoRAMemoryPool.uid_to_buffer_id` 里没有它，下一次 forward 前应该由谁把它装进 slot？
3. 如果两个请求 token 完全相同但 adapter 不同，哪一行源码保证它们不会共享同一个 prefix cache key？
4. 如果 `enable_lora_overlap_loading=True`，为什么第一次遇到新 adapter 的请求可能不会立刻进入 batch？
5. 如果 MoE LoRA 在 `tp=4, ep=4` 下 shape mismatch，为什么不能只按外层 `tp_size` 推断 expert buffer 宽度？
6. 如果 `forward_batch.lora_ids` 有一个 uid 不在 pool，为什么当前实现可能读 slot 0 而不是立即报错？
7. 为什么 `/unload_lora_adapter` 返回后仍在 GPU pool 看到旧 uid，不足以证明 unload 失败？

## 通过标准

能不看笔记复述一条请求的 adapter 身份流：`lora_path → lora_id → extra_key → running_loras → ForwardBatch.lora_ids → buffer_id → weight_indices → LoRABatchInfo → layer delta`，并能为其中任意三步指出源码入口和一个失败模式。
