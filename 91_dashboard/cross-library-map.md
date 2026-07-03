---
type: dashboard
title: "跨库专题对照"
tags:
  - dashboard
  - index
  - sglang/index-layer
  - slime/index-layer
updated: 2026-07-03
---

# 跨库专题对照

> Slime 用 SGLang 作 Rollout 推理引擎。本页是跨库专题映射；按模块名与主题跳转。

---

## 架构对照

| 维度 | Slime | SGLang |
|------|-------|--------|
| 主循环 | `generate → train → update_weights` | 请求 → batch → forward → 响应 |
| 运行时 | Ray + Megatron + SGLang | Tokenizer + Scheduler + Detokenizer |
| 权重更新 | Train→Rollout NCCL/disk | CheckpointEngine 热更新 |
| 总入口 | [[Slime源码阅读指南]] | [[SGLang源码阅读指南]] |

---

## 专题对照表

| Slime 专题 | SGLang 专题 | 原因 |
|------------|-------------|------|
| [[12-SGLang-Rollout-00-MOC]] | [[04-OpenAI-API-00-MOC]] · [[06-TokenizerManager-00-MOC]] · [[07-Scheduler-00-MOC]] | HTTP generate 进入推理栈 |
| [[15-SGLang-Engine-00-MOC]] | [[02-启动链路-00-MOC]] · [[03-HTTP-Server-00-MOC]] | engine 启动与 server 生命周期 |
| [[09-EngineTopology-00-MOC]] · [[16-External-Engines-00-MOC]] | [[22-Disaggregation-00-MOC]] · [[23-Distributed-00-MOC]] | PD 拓扑与多节点 |
| [[24-WeightSync-Dist-00-MOC]] · [[25-WeightSync-Disk-00-MOC]] · [[26-Checkpoint-M2HF-00-MOC]] | [[12-ModelLoader-00-MOC]] · [[32-CheckpointEngine-00-MOC]] | 权重格式与热更新 |
| [[12-SGLang-Rollout-00-MOC]] | [[20-Sampling-00-MOC]] | sampling_params 透传 |
| — | [[15-RadixAttention-00-MOC]] · [[16-KV-Cache-00-MOC]] · [[17-Attention-00-MOC]] | Slime 不实现 KV；纯推理深潜 |

---

## 全链路对照

### Slime Rollout ↔ SGLang 推理 Hop

```mermaid
flowchart LR
    subgraph Slime
        SR["sglang_rollout.py<br/>HTTP POST"]
    end
    subgraph SGLang
        HTTP["http_server /generate"]
        TM["TokenizerManager"]
        SCH["Scheduler"]
        MR["ModelRunner"]
    end
    SR --> HTTP --> TM --> SCH --> MR
```

| Slime | SGLang | 文档 |
|-------|--------|------|
| `generate_and_rm_group` | `/generate` | [[12-SGLang-Rollout-02-源码走读]] · [[04-OpenAI-API-02-源码走读]] |
| sampling_params | `SamplingParams` | [[20-Sampling-01-核心概念]] |
| rollout_log_probs | forward logprob | [[20-Sampling-03-数据流与交互]] |
| PD 拓扑 | DisaggregationMode | [[22-Disaggregation-01-核心概念]] |

双全链路：[[全链路RL训练追踪]]（Hop 4 嵌入 [[全链路请求追踪]]）

---

## 参数透传

Slime `--sglang-*` → `sglang_parse_args()` → SGLang `ServerArgs`

→ [[04-Arguments-TrainRollout-02-源码走读]] · [[03-HTTP-Server-01-核心概念]]

---

## 权重同步

| Slime | SGLang |
|-------|--------|
| `update_weight_from_distributed` | CheckpointEngine / weight sync API |
| `megatron_to_hf` | `ModelLoader` HF 加载 |
| `--colocate` tensor 直传 | 同进程权重共享 |

→ [[24-WeightSync-Dist-01-核心概念]] · [[32-CheckpointEngine-01-核心概念]]

---

## 阅读顺序

**已有 SGLang 基础 → 读 Slime：** [[08-总结与索引-01-项目总览]] → [[全链路RL训练追踪]] → [[08-总结与索引-04-导读路径]]

**从零双库：** [[91_dashboard/dual-library-path|双库联合路径]]

---

## 基线 commit

| 库 | commit |
|----|--------|
| sglang | `70df09b` |
| slime | `22cdc6e1` |
