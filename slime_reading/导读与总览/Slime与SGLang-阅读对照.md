---
title: "Slime 与 SGLang 阅读对照"
type: map
framework: slime
topic: "导读与总览"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/map
  - source-reading
updated: 2026-07-10
---
# Slime 与 SGLang 阅读对照

读 Slime Rollout 层时，底层推理走 SGLang：HTTP → TokenizerManager → Scheduler → ModelRunner。本页聚焦两套系统的交接对象；更多横向主题见 [[knowledge_maps/三框架知识地图|跨库专题对照]]。

---

## 架构对照

| 维度 | Slime（当前） | SGLang（推理栈） |
|------|---------------|------------------|
| 主循环 | `generate → train → update_weights` | 请求 → batch → forward → 响应 |
| 运行时 | Ray + Megatron + SGLang | Tokenizer + Scheduler + Detokenizer |
| 权重更新 | Train→Rollout NCCL / disk | CheckpointEngine 热更新 |
| 总入口 | [[Slime学习指南]] | [[SGLang学习指南]] |

---

## 专题对照（Slime → SGLang）

| Slime 专题 | SGLang 专题 | 衔接说明 |
|------------|-------------|----------|
| [[Slime-SGLang-Rollout]] | [[SGLang-OpenAI-API]] · [[SGLang-TokenizerManager]] · [[SGLang-Scheduler]] | `sglang_rollout.py` 发 HTTP generate，进入推理三进程 |
| [[Slime-SGLang-Engine]] | [[SGLang-启动链路]] · [[SGLang-HTTP-Server]] | engine 子进程启动与 server 生命周期 |
| [[Slime-引擎拓扑]] · [[Slime-外部推理引擎]] | [[SGLang-PD分离]] · [[SGLang-分布式]] | PD 分离与多节点拓扑 |
| [[Slime-分布式权重同步]] · [[Slime-磁盘权重同步]] · [[Slime-Megatron到HF转换]] | [[SGLang-ModelLoader]] · [[SGLang-CheckpointEngine]] | Megatron→HF 转换与权重热更新 |
| [[Slime-SGLang-Rollout]] | [[SGLang-Sampling]] | sampling_params 透传至 `SamplingParams` |
| — | [[SGLang-RadixAttention]] · [[SGLang-KV-Cache]] · [[SGLang-Attention]] | Slime 不实现 KV；Attention 深潜需回 SGLang |

---

## 生命周期对照

| Slime 边界 | SGLang 对应 | 文档 |
|-----------|-------------|------|
| `generate_and_rm_group` | HTTP `/generate` | [[Slime-RL训练全链路]] · [[SGLang-OpenAI-API-源码走读]] |
| RolloutManager 调度 | TokenizerManager 收请求 | [[Slime-RolloutManager-源码走读]] · [[SGLang-TokenizerManager-源码走读]] |
| SGLang 子进程推理 | Scheduler + ModelRunner | [[SGLang-HTTP请求全链路]] 的调度与执行部分 |
| `rollout_log_probs` | forward logprob | [[SGLang-Sampling-数据流]] |
| `train` + `update_weights` | —（训练侧 Slime 独有） | [[Slime-训练步骤-源码走读]] · [[Slime-分布式权重同步-源码走读]] |

对照阅读：**[[Slime-RL训练全链路]]**（Rollout 嵌入）↔ **[[SGLang-HTTP请求全链路]]**。

---

## 推荐阅读路径

**读 Rollout 前补 SGLang：**

1. [[SGLang-零基础先修]] — Prefill/Decode、KV Cache、三进程
2. [[SGLang-HTTP请求全链路]] — HTTP 请求、调度、执行与输出回程
3. [[SGLang-学习路径]] — 按系统职责组织的导读（按需）

**已有 Slime、补推理栈：** [[SGLang-HTTP请求全链路]] → [[SGLang-学习路径]]

**从零双库：** [[knowledge_maps/AI-Infra联合学习路径|双库联合路径]]

**专题级跳转：** [[knowledge_maps/三框架知识地图|跨库专题对照]]

---

## 参数与权重

Slime CLI 中 `--sglang-*` 经 `sglang_parse_args()` 映射为 SGLang `ServerArgs` → [[Slime-训练与Rollout参数-源码走读]] · [[SGLang-HTTP-Server-核心概念]]

`update_weight_from_distributed` 触发 SGLang CheckpointEngine / weight sync API → [[Slime-分布式权重同步-核心概念]] · [[SGLang-CheckpointEngine-核心概念]]

`--colocate` 模式下 tensor 直传，与 SGLang 同进程权重共享场景相关 → [[Slime-引擎拓扑-核心概念]]
