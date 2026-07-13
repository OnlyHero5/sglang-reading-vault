---
title: "多模态 · 学习检查"
type: exercise
framework: sglang
topic: "多模态"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 多模态 · 学习检查

## 你为什么要做

看懂名词不等于能定位跨进程、跨模型组件的错位。本页要求你用源码和小实验证明判断；没有 GPU 时也给出静态替代。

## 完成标准

- [ ] 能从架构名追到具体 Processor，并解释 Transformers backend 闸门。
- [ ] 能区分 raw media、processor feature、precomputed embedding。
- [ ] 能解释内部 modality 分组为何不破坏 prompt 顺序。
- [ ] 能画出 TokenizerManager→Scheduler 的共享内存/IPC 边界。
- [ ] 能证明 CUDA IPC 当前实现不是零复制。
- [ ] 能说明 hash/pad 的缓存身份语义。
- [ ] 能复现或静态证明 auto truncate 与 ViT graph 的两个风险。
- [ ] 能解释 encoder-only、language-only 和三种 transfer backend。

## 练习 1：画一项媒体的生命周期

不看笔记，补全：

```text
URL/Base64
→ __________
→ pixel/audio __________
→ MultimodalDataItem
→ __________ proxy 或 CPU tensor
→ Scheduler __________
→ hash + __________
→ ViT/Audio tower
→ __________ embedding
→ placeholder span
→ LLM prefill
```

**预期答案**：media loader/processor、feature、CUDA IPC、reconstruct、pad value、encoder/visual。

**验收追问**：哪条路径会跳过 ViT？答：`PRECOMPUTED_EMBEDDING`。

## 练习 2：验证 Processor 选择

```powershell
rg -n "PROCESSOR_MAPPING|supports_transformers_backend|get_mm_processor" sglang/python/sglang/srt/managers/multimodal_processor.py
rg -n "models =|supports_transformers_backend" sglang/python/sglang/srt/multimodal/processors
```

**操作**：任选一个 Qwen-VL processor，写出注册模型类名、是否支持 Transformers backend、最终构造函数入口。

**预期**：不能只凭文件名判断；必须能把 `hf_config.architectures` 的字符串命中到注册类名。

## 练习 3：证明 prompt 顺序与内部分组可同时成立

给定：

```text
prompt: <audio> ... <image> ... <video> ... <image>
images: [A, B]
videos: [V]
audios: [X]
```

写出占位符消费序列。

**预期**：X → A → V → B。`organize_results()` 即使返回 `[A, B, V, X]`，`build_input_ids()` 仍按 prompt 扫描，并使用三种独立游标。

## 练习 4：检查 item 的统一校验是否真实存在

```powershell
rg -n -A 4 "def validate\(self\)" sglang/python/sglang/srt/managers/schedule_batch.py
```

**预期**：看到 `...` 与“待实现”注释。

**结论题**：能否声称所有 offset/grid/feature mismatch 都在 `MultimodalDataItem` 层拒绝？不能。

## 练习 5：证明 CUDA IPC 不是零复制

```powershell
rg -n -A 22 "def _copy_slice_tensor_to_target" sglang/python/sglang/srt/utils/cuda_ipc_transport_utils.py
```

**预期**：看到 consumer `torch.empty(...)`，随后 `copy_(slice_tensor)`，最后递增共享计数。

**口述**：它省掉什么？省掉 producer→CPU→consumer 的往返。它保留什么？consumer-side device copy 与目标 tensor 分配。

## 练习 6：算 IPC pool 的实际下界

已知总预算 256 MiB，`tokenizer_worker_num=8`。

```text
名义均分：256 / 8 = 32 MiB
代码下限：max(32, 128) = 128 MiB / worker
实际池合计下界：8 × 128 = 1024 MiB
```

**验收**：解释为什么“配置预算 256 MiB”不能当成硬总上限。

## 练习 7：静态证明 auto truncate 风险

```powershell
rg -n -A 16 "if input_token_num >= self.context_len" sglang/python/sglang/srt/managers/tokenizer_manager.py
```

回答：

1. 哪个对象被裁？`input_ids`。
2. `mm_inputs.offsets` 是否同步裁？这段代码没有。
3. 截断点落入媒体 span 会怎样？token 与 item 对齐可能被破坏。
4. 安全替代？processor 前做预算或拒绝超长请求。

## 练习 8：构造 ViT graph 反例

设计两个请求：

- 请求 A：单图，视觉 token 总数 1024；
- 请求 B：两图，各 512，总数也为 1024。

静态检查：

```powershell
rg -n -A 5 "def _get_graph_key" sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py
rg -n "cu_seqlens|cu_window_seqlens" sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py
```

**预期**：两请求 key 都是 1024，但分段 metadata 不同。

**有 GPU 时**：分别跑 eager 与 graph，比较 encoder embedding 和最终 logits。若 B replay A 的 graph 后不一致，记录为布局 key 缺失，而不是笼统归因于“动态分辨率”。

## 练习 9：追踪 hash 时机

```powershell
rg -n "mm_hashes|SGLANG_MM_PRECOMPUTE_HASH|set_pad_value" sglang/python/sglang/srt/managers/tokenizer_manager.py sglang/python/sglang/srt/managers/schedule_batch.py
```

**预期**：

- caller hash 可覆盖 item.hash；
- 是否立即计算 pad 由环境变量决定；
- Scheduler 无论如何会在构造 `MultimodalInputs` 时保证 pad。

## 练习 10：画 encoder disaggregation

为三种 backend 各画一条：

```text
zmq_to_tokenizer: encoder → ______ → Scheduler
zmq_to_scheduler: encoder → ______
mooncake: /encode 返回 ______ → /send 等待 ______ → RDMA transfer
```

**预期**：TokenizerManager；Scheduler endpoint；embedding metadata；ready event。

再回答：为什么 DP dispatcher 要保存 `req_id → rank`？因为后续 `/send` 必须回到持有该 embedding 的同一 worker。

## 练习 11：运行静态门禁

```powershell
node maintenance/audit_wikilinks.mjs
node maintenance/audit_source_evidence.mjs
node maintenance/audit_markdown_quality.mjs
git diff --check
```

**预期**：三项审计零错误零警告；`git diff --check` exit 0。已有换行风格 warning 可记录，但不能掩盖新 whitespace error。

## 最终口述验收

请在三分钟内回答：

> 一张图片进入 SGLang 后，为什么 Processor、TokenizerManager、Scheduler 和 ViT 都不能单独保证正确？CUDA IPC、hash/pad、ViT CUDA Graph、encoder disaggregation 分别优化了什么，又各自不能改变什么？

合格答案必须出现：placeholder span、模型专用 metadata、Scheduler 二次重建、词表外缓存身份、IPC device copy、graph key 仅 `S`、远端 encoder 的所有权与回退边界。
