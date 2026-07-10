---
title: "SGLang 与 Slime 阅读对照"
type: map
framework: sglang
topic: "导读与总览"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# SGLang 与 Slime 阅读对照

读完 SGLang 推理栈后，若继续 RL 后训练闭环，Slime 会把 SGLang 当作 Rollout 推理引擎。本页只解释两套系统如何在请求、logprob 和权重更新处接上；更多横向主题见 [[knowledge_maps/三框架知识地图|跨库专题对照]]。

---

## 架构对照

| 维度 | SGLang（你已读） | Slime（续读） |
|------|------------------|---------------|
| 主循环 | 请求 → batch → forward → 响应 | `generate → train → update_weights` |
| 运行时 | Tokenizer + Scheduler + Detokenizer | Ray + Megatron + SGLang |
| 权重更新 | CheckpointEngine 热更新 | Train→Rollout NCCL / disk |
| 总入口 | [[SGLang学习指南]] | [[Slime学习指南]] |

---

## 专题对照（SGLang → Slime）

| SGLang 专题 | Slime 专题 | 衔接说明 |
|-------------|------------|----------|
| [[SGLang-OpenAI-API]] · [[SGLang-TokenizerManager]] · [[SGLang-Scheduler]] | [[Slime-SGLang-Rollout]] | HTTP `/generate` 进入推理栈；Slime 通过 `sglang_rollout.py` 调用 |
| [[SGLang-启动链路]] · [[SGLang-HTTP-Server]] | [[Slime-SGLang-Engine]] | engine 启动与 server 生命周期 |
| [[SGLang-PD分离]] · [[SGLang-分布式]] | [[Slime-引擎拓扑]] · [[Slime-外部推理引擎]] | PD 拓扑与多节点部署 |
| [[SGLang-ModelLoader]] · [[SGLang-CheckpointEngine]] | [[Slime-分布式权重同步]] · [[Slime-磁盘权重同步]] · [[Slime-Megatron到HF转换]] | 权重格式转换与热更新 |
| [[SGLang-Sampling]] | [[Slime-SGLang-Rollout]] | `SamplingParams` 与 sampling_params 透传 |
| [[SGLang-RadixAttention]] · [[SGLang-KV-Cache]] · [[SGLang-Attention]] | — | Slime 不实现 KV；纯推理深潜留在 SGLang |

---

## 生命周期对照

| SGLang 边界 | Slime 对应 | 文档 |
|------------|------------|------|
| HTTP `/generate` | `generate_and_rm_group` | [[SGLang-HTTP请求全链路]] · [[Slime-SGLang-Rollout-源码走读]] |
| TokenizerManager | rollout 请求组装 | [[SGLang-TokenizerManager-源码走读]] |
| Scheduler + ModelRunner | SGLang 子进程（Slime 不介入） | [[SGLang-Scheduler-源码走读]] |
| 响应 / logprob | `rollout_log_probs` | [[SGLang-Sampling-数据流]] |
| — | `train` + `update_weights` | [[Slime-RL训练全链路]] |

Slime RL 全链路的 Rollout 生成嵌入 SGLang 推理栈；对照阅读：**[[SGLang-HTTP请求全链路]]** ↔ **[[Slime-RL训练全链路]]**。

---

## 推荐阅读路径

**已有 SGLang 基础：**

1. [[Slime-项目总览]] — Slime 三角架构
2. [[Slime-RL训练全链路]] — `parse_args` → generate → train → update_weights
3. [[Slime-学习路径]] — 按闭环职责组织的导读

**从零双库：** [[knowledge_maps/AI-Infra联合学习路径|双库联合路径]]

**专题级跳转：** [[knowledge_maps/三框架知识地图|跨库专题对照]]

---

## 参数与权重

Slime `--sglang-*` 参数经 `sglang_parse_args()` 注入 SGLang `ServerArgs` → [[SGLang-HTTP-Server-核心概念]] · [[Slime-训练与Rollout参数-源码走读]]

权重同步：`update_weight_from_distributed` 对接 CheckpointEngine → [[SGLang-CheckpointEngine-核心概念]] · [[Slime-分布式权重同步-核心概念]]
