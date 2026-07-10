---
title: "MoE"
type: map
framework: sglang
topic: "MoE"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# MoE

> **SGLang 内存与 Attention** | 源码基线：`70df09b83363e0127b43c83a6007d3938f815b2d`
> **核心范围：** `models/*moe.py`、`layers/moe/topk.py`、`layers/moe/fused_moe_triton/layer.py`、`layers/moe/token_dispatcher/`、`eplb/`

## 读者为什么要读

MoE 层的难点不是“多了几个专家”，而是同一个 token 在一层里会变成多份路由任务：先由 gate 选择 top-k 专家，再按 expert id 被 dispatch 到本地或远端 rank，经过专家 GEMM 后再 combine 回原 token 顺序。吞吐下降、all-to-all 变慢、某个 rank 成为 straggler、CUDA Graph 断开，通常都发生在这条链上。

读完本专题后，你应该能做三件事：

- 解释一个 token 在 MoE 层内如何经历 `gate → topk → dispatch → expert GEMM → combine`。
- 判断瓶颈在 router、expert GEMM、EP all-to-all、EPLB 迁移，还是 TP all-reduce。
- 按源码入口排查 expert 不均、DeepEP 状态错误、top-k 格式不匹配、量化 runner 选错和 piecewise CUDA Graph fallback。

## 主线地图

把 MoE 想成“分诊和转运系统”：gate 是分诊台，top-k 输出是转诊单，dispatcher 是转运车，expert GEMM 是专科诊室，combine 是把多份处方按权重合并回原病历。这个类比只覆盖 token 到专家的生命周期，不解释每个 GEMM kernel 内部 tile。

```mermaid
flowchart LR
  H["hidden_states<br/>[num_tokens, hidden]"]
  G["gate<br/>router_logits"]
  K["TopK<br/>topk_ids / topk_weights"]
  D["Dispatcher<br/>local permute or A2A"]
  E["MoeRunner<br/>expert GEMM"]
  C["Combine<br/>weighted sum + unpermute"]
  O["output<br/>[num_tokens, hidden]"]
  L["EPLB<br/>logical to physical map"]

  H --> G --> K
  L -. "改写 expert id" .-> K
  K --> D --> E --> C --> O
```

## 一条 token 穿过它

场景：DeepSeek 或 BailingMoE 这类模型的一个 decode step 里，batch 中每个 token 已经有 hidden state。MoE 层先通过 `gate(hidden_states)` 得到每个 token 对所有 logical expert 的 logits；`TopK` 选择 top-k expert，并在 EPLB/EP 场景把 logical id 转为 physical id；`FusedMoE` 的 `forward_impl` 固定执行 dispatch、`run_moe_core`、combine；DeepEP 只是 dispatcher 的一种实现，把 dispatch/combine 拆成两段以管理 A2A 状态；EPLB 周期性根据 expert 分布统计重排 logical expert 到 physical expert 的映射。

| 阶段 | 输入 | 输出 | 常见瓶颈 |
|------|------|------|----------|
| gate | `hidden_states` | `router_logits` | 轻量矩阵乘或 fused router |
| top-k | `router_logits` | `topk_ids/topk_weights` | grouped top-k、bias、padding mask |
| dispatch | hidden + top-k | expert 分组后的 hidden | EP all-to-all |
| GEMM | expert 分组 hidden | expert 输出 | 量化 runner、local expert GEMM |
| combine | expert 输出 + weights | 原 token 顺序 hidden | A2A 回收、weighted sum |
| rebalance | expert 计数 | 新 physical map | 权重迁移与短暂停顿 |

## 五篇怎么读

| 文件 | 读完能解决什么 |
| ------ | ---------------- |
| [[SGLang-MoE-核心概念]] | 建立 gate、top-k、dispatcher、expert GEMM、combine、EPLB 的边界 |
| [[SGLang-MoE-源码走读]] | 沿一个 token 的 MoE 层生命周期读源码证据 |
| [[SGLang-MoE-数据流]] | 追踪 `topk_ids`、`topk_weights`、logical/physical expert id、dispatch state |
| [[SGLang-MoE-排障指南]] | 按症状排查 A2A、EPLB、top-k、量化和 CUDA Graph |
| [[SGLang-MoE-学习检查]] | 验收自己是否能画图、复述、排障、改配置 |

## 源码范围

| 文件 | 重点范围 | 用途 |
|------|----------|------|
| `sglang/python/sglang/srt/models/bailing_moe.py` | `_forward_router_experts`、`forward_deepep` | 模型侧如何调用 gate/top-k/experts |
| `sglang/python/sglang/srt/layers/moe/topk.py` | `TopKOutput`、`select_experts`、post process | top-k 输出契约、logical to physical 映射、统计记录 |
| `sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | `FusedMoE.forward`、`forward_impl`、`run_moe_core` | MoE 层执行骨架和量化 runner 注入点 |
| `sglang/python/sglang/srt/layers/moe/token_dispatcher/base.py` | `BaseDispatcher` hook | dispatch/combine 扩展协议 |
| `sglang/python/sglang/srt/layers/moe/token_dispatcher/deepep.py` | `DeepEPDispatcher` | 跨 rank A2A 的阶段状态机 |
| `sglang/python/sglang/srt/eplb/` | `EPLBManager`、expert location dispatch | 负载统计、重排、logical/physical expert 映射 |

## 不变量

- `topk_ids` 是 token 到 expert 的路由决策；`topk_weights` 是 combine 时的权重。
- `FusedMoE.forward_impl` 的骨架是 dispatch → `run_moe_core` → combine，量化只改变 core 的 runner，不改变这条生命周期。
- EP 场景下，logical expert id 可能在 dispatch 前被映射成 physical expert id。
- DeepEP 的 dispatch/combine 有内部阶段状态；阶段顺序错了就是逻辑错误，不是单纯性能下降。
- EPLB 通过统计 expert 分布再更新映射，不应该被理解为每个 token 动态搜索最快 rank。

## 验证入口

- 查看模型层断点：`gate(hidden_states)` 后 `router_logits.shape`，`topk` 后 `topk_ids.shape`。
- 查看 MoE 执行骨架：`FusedMoE.forward_impl` 中 dispatch、`run_moe_core`、combine 是否按顺序执行。
- 查看 EP 瓶颈：在 DeepEP dispatcher 的 `dispatch_a/dispatch_b/combine_a/combine_b` 处断点，确认 A2A 发生在哪一段。
- 查看负载不均：开启 expert distribution recorder 或 EPLB 日志，观察 `logical_count` 与 rebalance 日志。
- 图编译问题：检查 `TopKOutput` 格式是否 standard 或 bypassed，否则会 fallback 到 eager `forward_impl`。

## 阅读路径

← [[SGLang-Attention|Attention]]
→ [[SGLang-Quantization|Quantization：量化]]
