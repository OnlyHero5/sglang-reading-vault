---
type: batch-doc
module: 14-Models-专用
batch: "14"
doc_type: checkpoint
title: "Models 专用 验收清单"
tags:
 - sglang/batch/14
 - sglang/module/models-specialized
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# Models 专用 验收清单

## 读者自测（不打开 sglang/）

- [ ] 能说明 DeepSeek 相对 Llama 的三大差异：MLA、MoE 稀疏层、DSA/CP
- [ ] 能画出 DecoderLayer：communicator → AttentionMLA → MoE/MLP
- [ ] 能说出 3 个核心组件：`dispatch_attn_forward_method`、`DeepseekV2MoE`、`determine_num_fused_shared_experts`
- [ ] 能解释 EntryClass 三版本共存的注册方式
- [ ] 五篇正文 ≥ 15 段内嵌源码

## 核心结论（3 句话）

1. **DeepSeek 专用逻辑集中在 `deepseek_v2.py`**，通过 Mixin dispatch MLA/MHA/DSA 多路径，对外仍保持标准 `forward` 接口。
2. **稀疏层 = DeepseekV2MoE**（gate + FusedMoE + 可选 HashTopK/DeepEP fusion），dense 层用 `DeepseekV2MLP`。
3. **Shared experts fusion 与 CP metadata 在 init/forward 入口自动决策**，错误配置会 disable fusion 并打 rank0 日志而非 silent 错载。

## 遗留问题

- MoE kernel / DeepEP dispatch 细节 → `layers/moe/` 未单列专题
- DSA Indexer 算法 → `layers/attention/dsa/`
- EPLB expert 迁移 → `eplb/` 模块

## 后续可补充主题

- `DeepseekV2AttentionMLA`（file: deepseek_v2.py，layer:model）
- `DeepseekV2MoE` / `MoEGate` / `HashTopK`
- `AttnForwardMethod` / `deepseek_common/attention_forward_methods.py`
- `DSACPLayerCommunicator`
- `determine_num_fused_shared_experts`
