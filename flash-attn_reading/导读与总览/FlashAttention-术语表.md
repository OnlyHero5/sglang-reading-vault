---
title: "FlashAttention 术语表"
type: reference
framework: flash-attn
topic: "导读与总览"
learning_role: reference
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/reference
  - source-reading
updated: 2026-07-10
---
# FlashAttention 术语表

## 读者任务

这篇用于快速消歧术语。读完后你应该能做到：

- 把数学术语、GPU memory 术语、API 术语和后端术语分开。
- 看到源码字段时，知道它更接近算法状态、输入边界、cache 状态还是 dispatch 组合。
- 避免把 FA2、FA3、FA4 的后端术语混在同一条路径里。

## Memory 与 kernel

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| HBM | GPU 高带宽显存，大但相对慢，attention 中间矩阵落 HBM 是主要瓶颈 | [[FlashAttention-Attention-IO]] |
| SRAM / shared memory | GPU SM 内共享内存，小但快，FlashAttention 将 tile 放在这里复用 | [[FlashAttention-Attention-IO-核心概念]] |
| register | 线程私有寄存器，保存 score、softmax 状态、输出累积 | [[FlashAttention-FA2-Forward-数据流]] |
| tile/block | Q/K/V 的分块单位，决定并行度、共享内存占用和 occupancy | [[FlashAttention-前向全链路]] |
| CTA | CUDA thread block 级执行单元，常负责一块 query tile | [[FlashAttention-FA2-Forward-源码走读]] |
| kernel specialization | 针对 dtype、head_dim、mask、dropout 等生成不同 kernel | [[FlashAttention-架构分层]] |

## Softmax 与数值状态

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| online softmax | 流式更新 softmax 最大值与归一化分母的算法 | [[FlashAttention-Online-Softmax]] |
| row max | 当前 query 行已处理 key blocks 的最大 score | [[FlashAttention-Online-Softmax-核心概念]] |
| row sum | 当前 max 标尺下的 exp 累积和 | [[FlashAttention-Online-Softmax-数据流]] |
| `acc_o` | 已处理 K/V blocks 对输出的累计贡献 | [[FlashAttention-关键概念]] |
| LSE | log-sum-exp，forward 保存给 backward 重算 softmax | [[FlashAttention-Online-Softmax-数据流]] |
| `S_dmask` | 测试/调试用 attention probability 与 dropout mask 输出，不是主线长期状态 | [[FlashAttention-FA2-Forward-排障指南]] |

## Attention 语义

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| causal mask | 自回归遮罩，query 只能看见当前位置及之前的 key；FA2.1 后 `seqlen_q != seqlen_k` 时按右下角对齐 | [[FlashAttention-FA2版本演进]] |
| local attention | sliding window attention，只看窗口内 key | [[FlashAttention-FA2-Forward-排障指南]] |
| ALiBi | Attention with Linear Bias，对 attention score 加线性位置偏置 | [[FlashAttention-FA2-Forward-排障指南]] |
| softcap | 对 score 做 soft cap，减少极端 logits | [[FlashAttention-版本演进全景]] |
| dropout | 训练时对 attention probability 做随机丢弃 | [[FlashAttention-Backward]] |

## 输入形态

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| MHA | Multi-Head Attention，Q/K/V head 数相同 | [[FlashAttention-Python-API-核心概念]] |
| MQA | Multi-Query Attention，多个 Q head 共享一个 KV head | [[FlashAttention-KV-Cache-核心概念]] |
| GQA | Grouped-Query Attention，多个 Q head 分组共享 KV head | [[FlashAttention-KV-Cache-核心概念]] |
| packed QKV | Q/K/V 预先合并存储，backward 可减少显式 concat | [[FlashAttention-Python-API-核心概念]] |
| varlen | 变长序列 batch，用 `cu_seqlens` 表达每条样本边界 | [[FlashAttention-Python-API-数据流]] |
| `cu_seqlens` | cumulative sequence lengths，形状通常为 `(batch + 1,)` | [[FlashAttention-Python-API-数据流]] |

## 推理与后端

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| KV cache | 推理 decode 时缓存历史 K/V，避免重复计算 | [[FlashAttention-KV-Cache]] |
| paged KV | 将 KV cache 切成 page/block 管理，服务动态 batch 与长上下文 | [[FlashAttention-KV-Cache-数据流]] |
| SplitKV | 将长 K/V 维度拆成多个并行分片，再 combine | [[FlashAttention-KV-Cache-数据流]] |
| TMA | Tensor Memory Accelerator，Hopper 上高效异步内存搬运机制 | [[FlashAttention-Hopper与CuTe-核心概念]] |
| GMMA | Hopper warpgroup 级矩阵乘指令族 | [[FlashAttention-Hopper与CuTe-核心概念]] |
| CuTe | CUTLASS 的 layout/tensor algebra 抽象 | [[FlashAttention-FA4-CuTeDSL演进]] |
| CuTeDSL | 用 Python DSL 表达 CuTe kernel 并 JIT 编译 | [[FlashAttention-FA4-CuTeDSL演进]] |

## 复盘

术语表只负责消歧。需要建立主线时回到 [[FlashAttention-学习路径]]；需要看源码证据时进入对应专题的 `源码走读` 或 `数据流与交互`。
