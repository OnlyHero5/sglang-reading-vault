---
title: "AI Infra 联合学习路径"
type: guide
framework: cross-framework
topic: "AI Infra"
learning_role: core
tags:
  - framework/cross-framework
  - content/guide
  - source-reading
updated: 2026-07-13
---

# AI Infra 联合学习路径

## 学习目标

SGLang、Slime 和 FlashAttention 分别覆盖 serving runtime、RL 后训练闭环和 attention kernel。联合学习不是把三个目录依次读完，而是先建立公共基础，再沿对象生命周期进入深度专题。

三者对应三种观察尺度：Slime 看一轮策略如何更新，SGLang 看一条请求如何服务，FlashAttention 看一次算子如何搬运 tensor。联合阅读的目标，是能在尺度之间缩放：既看得见整条闭环，也能在需要时落到一个字段、一次 collective 或一个 register accumulator。

## 三种观察尺度与 GPU 底座

```mermaid
flowchart TB
    RL["Slime<br/>prompt sample reward train weight"]
    SV["SGLang<br/>request schedule KV forward response"]
    OP["实际 Attention backend<br/>QKV metadata kernel"]
    GPU["GPU<br/>HBM shared memory register Tensor Core"]
    RL --> SV
    SV -->|"按运行时 dispatch"| OP
    OP --> GPU
```

FlashAttention 提供 IO、online softmax 与 kernel 阅读主线，但 SGLang 实际运行可能选择 FlashInfer、Triton、FlashAttention 或平台专用 backend。联合阅读是尺度映射，不是静态依赖声明。

## 从零路径

### 基础层

[[LLM推理与Token]] → [[并发进程与背压]] → [[GPU内存与算子]] → [[分布式通信与并行]] → [[RL后训练数学基础]] → [[性能指标与实验方法]]

### 系统层

[[推理Serving主线]] → [[Attention算子主线]] → [[RL训练闭环主线]]

### 联合层

[[从Prompt到新权重]] → [[跨库一致性实验]] → [[课程完成标准]]

这条核心路径适合两到四周完成。三库深度专题是后续参考层，不把“读完全部文档”作为入门完成标准。

## 已有背景的路径

| 背景 | 推荐入口 |
|------|----------|
| 熟悉 serving | [[Attention算子主线]] → [[RL训练闭环主线]] |
| 熟悉训练 | [[推理Serving主线]] → [[从Prompt到新权重]] |
| 熟悉 CUDA | [[推理Serving主线]] → [[SGLang-Attention]] → [[FlashAttention-KV-Cache]] |
| 正在生产排障 | [[SGLang-生产排障]] → [[排障指南.base]] |
| 准备改框架源码 | [[源码走读.base]] → 对应数据流和学习检查 |

## 深度专题

- SGLang：[[SGLang-请求调度]] · [[SGLang-模型执行]] · [[SGLang-内存与Attention]] · [[SGLang-高级特性]]
- Slime：[[Slime-Ray编排]] · [[Slime-Rollout生成]] · [[Slime-训练后端]] · [[Slime-权重同步]]
- FlashAttention：[[FlashAttention-Attention-IO]] · [[FlashAttention-FA2-Forward]] · [[FlashAttention-Backward]] · [[FlashAttention-KV-Cache]]

## 实验路径

[[SGLang服务实验]] → [[FlashAttention性能实验]] → [[Slime闭环实验]] → [[跨库一致性实验]]

每次实验先记录模型、硬件、版本、workload 和单一变量，再记录预期与实际。没有 GPU 时完成静态定位；有 GPU 时补运行数据。

完成路径的验证不是“链接都点过”，而是能交付三张主线图、四本身份账、一份可复现实验和一次故障推演；详见 [[课程完成标准]]。

## 导航

[[index]] · [[AI-Infra入门课程]] · [[三框架知识地图]] · [[知识地图首页]]
