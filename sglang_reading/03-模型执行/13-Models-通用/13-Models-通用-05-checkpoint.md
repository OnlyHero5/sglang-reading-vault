---
type: batch-doc
module: 13-Models-通用
batch: "13"
doc_type: checkpoint
title: "Models 通用 验收清单"
tags:
 - sglang/batch/13
 - sglang/module/models-common
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# Models 通用 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 ModelRegistry 如何通过 EntryClass 注册 architecture
- [ ] 能画出 ModelLoader → Registry → ModelRunner → model.forward → RadixAttention 的位置
- [ ] 能说出 3 个核心组件：`resolve_model_cls`、`LlamaAttention`、`Qwen3Attention`（文档中均有内嵌代码）
- [ ] 能追踪一条权重加载路径：HF `q_proj` → `qkv_proj` stacked mapping → GPU param
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. **ModelRegistry 在 import 时扫描 `sglang.srt.models` 下所有 `EntryClass`**，`resolve_model_cls` 将 HF `architectures` 映射到 native `*ForCausalLM` 类，不支持时 fallback Transformers。
2. **Llama / Qwen3 共享 Pre-Norm Decoder + RadixAttention 模式**；Qwen3 额外 QK-Norm、`LayerCommunicator` 与 attn_tp 切分。
3. **`load_weights` 通过 stacked_params_mapping 合并 QKV/gate-up 分片**，PP rank 只加载本 stage 层参数。

## 遗留问题（后续专题）

- DeepSeek MLA/MoE / DSA → Models 专用
- RadixAttention.forward 与 prefix cache → RadixAttention
- Transformers fallback 内部实现 → 未单列专题
