---
type: batch-doc
module: 25-LoRA
batch: "25"
doc_type: checkpoint
title: "LoRA 验收清单"
tags:
 - sglang/batch/25
 - sglang/module/lora
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# LoRA 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能说明 LoRAManager、LoRAMemoryPool、BaseLayerWithLoRA 的分工
- [ ] 能追踪请求 lora_id 从 API 到 CSGMV kernel 的路径
- [ ] 能解释 max_loras_per_batch 与 LRU eviction 的关系
- [ ] 能说出 embedding LoRA 与 linear LoRA 计算路径差异
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论

1. LoRAManager 替换 target 层为 LoRA 包装层，LoRAMemoryPool 管理 GPU 槽位，支持 batch 内多 adapter 并行。
2. Triton CSGMV backend 按 token→adapter 映射批处理 A/B 乘法，实现 S-LoRA/Punica 风格吞吐。
3. LRU eviction + lora_drainer 在容量有限时换出冷 adapter，与动态 load API 配合支持多租户。

## 遗留问题

- trtllm_lora_temp 实验路径与主 backend 收敛时间表以 upstream 为准。
