---
type: batch-doc
module: 17-Attention
batch: "17"
doc_type: checkpoint
title: "Attention 验收清单"
tags:
 - sglang/batch/17
 - sglang/module/attention
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# Attention 验收清单

## 内容更新（2026-07-02）

- [x] §1 后端分层：HybridAttnBackend 实码 + flashinfer/triton/trtllm_mla 选型表 + vLLM 对比
- [x] §2 Extend vs Decode：类比 + ForwardMode 实码 + Mermaid 数据流 + Radix prefix 衔接
- [x] §6 设计追问：hybrid backend / CUDA Graph out_graph·in_graph / FlashInfer fallback 三问
- [x] 04-关键问题：新增 SGLang vs vLLM PagedAttention 对比（memory_pool + kv_indptr 实码）

## 读者自测（不打开 sglang/）

- [x] 仅读本模块 sglang_reading，能口头说明 AttentionBackend 三方法 metadata 契约
- [x] 能画出 RadixAttention.forward → get_attn_backend().forward → forward_extend/decode 路径
- [x] 能说出 FlashInfer 与 Triton 的选型差异（文档中均有内嵌代码）
- [x] 能追踪 extend 阶段 KV 写回与 paged prefill wrapper 的调用关系
- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [x] 2026-07-02 扩写：§1-2 后端分层与 Extend/Decode、§6 设计追问、vLLM 对比 Q
- [x] 2026-07-02 质量：Explain 2-4 句、消除「（概念）」、实码走读
- [x] 来源注释路径/行号与 git `70df09b` 一致
- [x] §9 RadixAttention 调用链已实码化（radix_attention / base_attn_backend / flashinfer_backend）

## 核心结论（3 句话）

1. AttentionBackend 三方法契约分离 eager/capture/replay 的 metadata 准备。
2. FlashInfer 与 Triton 共享 paged KV 元数据模型。
3. Extend 与 Decode 使用不同 kernel/wrapper 路径；RadixAttention.forward 统一分派到 backend。

## 内嵌源码统计（维护者）

| 文档 | 代码块（约） | 说明 |
|------|-------------|------|
| README.md | 1 | AttentionBackend 契约入口 |
| 01-核心概念.md | 4 | 后端分层 HybridAttn、ForwardMode、FlashInfer 模块、§6 设计追问 |
| 02-源码走读.md | 12 | 含 §9 RadixAttention 四段实码走读 |
| 03-数据流与交互.md | 2 | Extend / CUDA Graph 数据流 |
| 04-关键问题.md | 3 | FlashInfer vs Triton、merge_state fallback、SGLang vs vLLM PagedAttention |
| **合计** | **21** | **≥220 行** |

## 遗留问题

- TRT-LLM MLA 后端细节 → Models 专用 / 18
- RadixAttention 与 prefix cache 语义 → RadixAttention
