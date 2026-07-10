---
title: "SGLang 补充主题"
type: reference
framework: sglang
topic: "总结复盘"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/reference
  - source-reading
updated: 2026-07-10
---
# SGLang 补充主题

> 索引层 · 对应 sglang `70df09b`  
> 下列 upstream 子系统暂无独立专题，但核心路径已在相关文档中内嵌讲解。本文给出“去哪读”，不重复已有深度内容。

---

## 1. 导读表

| Upstream 主题 | 为何未独立成专题 | 阅读路径 | 深度 |
|---------------|----------------|-------------------------|------|
| **Pipeline Parallelism (PP)** | 与 Scheduler / ModelRunner mixin 强耦合 | [[SGLang-Scheduler-源码走读]] PP mixin · [[SGLang-ModelRunner-数据流]] §5 · [[SGLang-分布式-核心概念]] | 中 |
| **HiCache / 分层 KV** | 跨 KV-Cache、Radix、Disaggregation | [[SGLang-KV-Cache-数据流]] · [[SGLang-RadixAttention-源码走读]] · [[SGLang-PD分离-核心概念]] | 中 |
| **Remote KV connectors** | 与 PD 分离、Radix 共享前缀 | [[SGLang-PD分离-数据流]] · [[SGLang-KV-Cache-排障指南]] | 浅–中 |
| **torch.compile cache** | 编译缓存属运维/调试 | [[SGLang-Attention-排障指南]] · [[SGLang-综合学习检查]] 已知局限 | 浅 |
| **平台后端（TPU/Ascend/CPU/Jetson）** | 非 NVIDIA CUDA 主路径 | upstream `docs/platforms/`；本库课程以 CUDA serving 为主 | 未覆盖 |
| **Benchmark 套件** | 非 runtime 主链路 | upstream `benchmark/`；性能方法见 [[性能指标与实验方法]] | 按需 |

---

## 2. Pipeline Parallelism — 30 分钟补课

补读重点：PP 将模型按层切到多 rank；非末 rank 不算 logits，只传 `PPProxyTensors`。

| 顺序 | 文档 | 关注点 |
|------|------|--------|
| 1 | [[SGLang-ModelRunner-数据流]] §5 | 非末 rank 返回 `pp_hidden_states_proxy_tensors` |
| 2 | [[SGLang-Scheduler-源码走读]] | 搜索 `pp_rank` / `PP` mixin 章节 |
| 3 | [[SGLang-分布式-核心概念]] | PP 与 TP/EP 组合时的进程拓扑 |

读法：若仅部署单卡/纯 TP，可先跳过；多节点大模型排障时再深读。

---

## 3. HiCache — 30 分钟补课

补读重点：HiCache 在 host/GPU 间分层存放 KV，与 Radix prefix 与 PD 传输交叉。

| 顺序 | 文档 | 关注点 |
|------|------|--------|
| 1 | [[SGLang-KV-Cache-核心概念]] | KV pool 与 consumer index |
| 2 | [[SGLang-ModelRunner-数据流]] §4 步骤 2 | `hicache_consumer_index` |
| 3 | [[SGLang-PD分离-源码走读]] | prefill/decode 分离时的 KV 迁移 |

---

## 4. 何时需要打开 upstream

- **平台移植 / 非 CUDA**：必须读 `sglang/docs/platforms/`
- **性能 benchmark 复现**：读 `sglang/benchmark/` README
- **行号漂移**：以函数名为锚在 `sglang/` 检索（基线 `70df09b`）

---

## 导航

- [[SGLang-导读与总览]]
- [[SGLang-综合学习检查]]
- [[SGLang-框架对比与设计决策]]
