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
updated: 2026-07-12
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
| HBM | GPU 板上高带宽显存；容量大于片上存储，但访问代价更高。`S/P` 的二次规模读写常是重要成本，是否主导仍取决于 workload 与 profile | [[FlashAttention-Attention-IO]] |
| SRAM / shared memory | 论文常用 SRAM 泛指片上快速存储；CUDA shared memory 是其中可编程、CTA 内共享的一类，不能把两词在所有语境下完全等同 | [[FlashAttention-Attention-IO-核心概念]] |
| register | 线程私有寄存器资源；具体 kernel 可在其中保存矩阵 fragment、softmax 行状态或输出累积，实际映射以实现和生成代码为准 | [[FlashAttention-FA2-Forward-数据流]] |
| tile | 算法或 kernel 一次处理的局部张量区域，例如 Q tile、K/V tile；tile shape 会影响并行度、片上资源和访存 | [[FlashAttention-前向全链路]] |
| block | 高度歧义词：可能指 CUDA thread block/CTA、tensor tile、KV page block 或逻辑分块；必须结合变量名和层级消歧 | [[FlashAttention-KV-Cache-核心概念]] |
| CTA | Cooperative Thread Array，CUDA 语境通常就是 thread block；FA2 forward 常让一个 CTA 处理一个 `(query tile, batch, head)` 工作单元 | [[FlashAttention-FA2-Forward-源码走读]] |
| kernel specialization | 把 head_dim、dtype、causal、dropout 等部分组合固化为编译期分支；这是 FA2 CUDA 等路径的策略，不代表 Triton/CuTe 使用相同生成方式 | [[FlashAttention-架构分层]] |

## Softmax 与数值状态

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| online softmax | 分块扫描 key 时，按共同最大值重标定旧状态并合并新块的算法；仍覆盖完整 dense attention | [[FlashAttention-Online-Softmax]] |
| row max | 当前 query 行已处理 key blocks 的最大 score | [[FlashAttention-Online-Softmax-核心概念]] |
| row sum | 当前 max 标尺下的 exp 累积和 | [[FlashAttention-Online-Softmax-数据流]] |
| `acc_o` | 已处理 K/V blocks 的重标定输出分子；epilogue 除以最终 `row_sum` 前还不是最终 `O` | [[FlashAttention-关键概念]] |
| `rP` | 当前 score tile 经 online update 后的低精度、未最终归一化指数权重，用完即乘 V；不是最终 probability | [[FlashAttention-关键概念]] |
| LSE / `softmax_lse` | 每个 query 行的 `log(sum(exp(score)))` 摘要；训练 forward/backward 用它恢复统一 softmax 标尺，具体 shape 随 dense/varlen 路径变化 | [[FlashAttention-Online-Softmax-数据流]] |
| `S_dmask` | testing-only 调试槽位，可能包含不同 scaling 与 dropout 符号编码；dropout 为 0 时公开三元组第三项为空，不能当稳定 attention map | [[FlashAttention-FA2-Forward-排障指南]] |

### 最危险的同名对象：不要看到 P 就说“概率矩阵”

| 名称 | 所在层 | 准确抓手 |
|------|--------|----------|
| 数学 `P = softmax(S)` | 算法公式 | 完整、最终归一化的 attention 权重概念 |
| `rP` | FA2 CUDA 主循环 | 当前 tile 的未最终归一化指数权重副本，马上参与 `V` 累积 |
| `p_ptr` | FA2 compiled 参数协议 | 条件调试缓冲指针；非空只说明请求了相应路径，不自动证明内容等于数学 `P` |
| `S_dmask` | Python/C++ 返回槽位 | testing-only、受 dropout 门禁影响的调试张量；不是稳定 attention map API |

判断规则：先问“它在哪一层、何时分配、shape 是什么、谁消费”，再解释名字。

## Attention 语义

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| causal mask | 自回归遮罩；FA2.1 后当 `seqlen_q != seqlen_k` 时按右下角对齐，因此不能总用方阵的 `j ≤ i` 直觉 | [[FlashAttention-FA2版本演进]] |
| local attention | sliding window attention，只看窗口内 key | [[FlashAttention-FA2-Forward-排障指南]] |
| ALiBi | Attention with Linear Bias，对 attention score 加线性位置偏置 | [[FlashAttention-FA2-Forward-排障指南]] |
| softcap | 用平滑饱和函数限制 score 幅度；`softcap=0` 表示关闭，不能与简单 hard clamp 混同 | [[FlashAttention-FA2-Forward-核心概念]] |
| dropout | 训练时对 softmax 权重施加随机 mask，并用保留概率倒数缩放；需要 RNG 状态支持 backward 重放 | [[FlashAttention-Backward]] |

## 输入形态

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| MHA | Multi-Head Attention，Q/K/V head 数相同 | [[FlashAttention-Python-API-核心概念]] |
| MQA | Multi-Query Attention，多个 Q head 共享一个 KV head | [[FlashAttention-KV-Cache-核心概念]] |
| GQA | Grouped-Query Attention，多个 Q head 分组共享 KV head | [[FlashAttention-KV-Cache-核心概念]] |
| packed QKV / KV | 在一个张量中增加轴来共同存放 Q/K/V 或 K/V；这不同于 varlen 把 batch 内有效 token 沿 token 轴拼接 | [[FlashAttention-Python-API-核心概念]] |
| varlen | 变长序列 batch：移除 padding 后把有效 token 拼接，并用累计边界恢复每条序列 | [[FlashAttention-Python-API-数据流]] |
| `cu_seqlens` | cumulative sequence lengths，长度为 `batch + 1` 的 int32 边界数组；相邻差是样本长度，末项是有效 token 总数 | [[FlashAttention-Python-API-数据流]] |

## 推理与后端

| 术语 | 中文解释 | 深入 |
|------|----------|------|
| KV cache | 推理 decode 时保存历史 token 的 K/V 投影，避免每步重新生成它们；attention 仍要读取被关注的历史 cache | [[FlashAttention-KV-Cache]] |
| paged KV | KV cache 按固定 page/block 存放，`block_table` 把逻辑块映射到物理页；页管理策略通常属于上层系统，kernel 消费映射 | [[FlashAttention-KV-Cache-数据流]] |
| SplitKV | 沿 K/V 工作量产生多个 partial O/LSE，再按 log-sum-exp 规则 combine；当前路径也可能选择 single-split，不是 decode 的必经步骤 | [[FlashAttention-KV-Cache-数据流]] |
| compiled extension | 预先编译并由 Python 加载的原生扩展；FA2 CUDA 与 ROCm CK 会经过此类路径，ROCm Triton 不应套用同一 pybind 链 | [[FlashAttention-Python-API]] |
| CK | Composable Kernel，当前 ROCm compiled backend 的实现体系之一 | [[FlashAttention-架构分层]] |
| Triton/Aiter | 当前 ROCm 可选 Python/Triton 路径及其 Aiter 依赖；CUDA 扩展 import 失败不会自动切到它 | [[FlashAttention-Python-API-排障指南]] |
| fake tensor / fake op | `torch.compile` 期间只构造输出 shape、dtype、device 等元数据的路径，不执行真实 attention 数值计算 | [[FlashAttention-Python-API-核心概念]] |
| TMA | Tensor Memory Accelerator，Hopper 的异步张量搬运能力；某条 FlashAttention 路径是否使用它必须由对应源码证明 | [[FlashAttention-Hopper与CuTe-核心概念]] |
| GMMA / WGMMA | Hopper warpgroup 级矩阵乘能力/指令语境；不要只凭 README 的 Hopper 支持声明推导具体调用 | [[FlashAttention-Hopper与CuTe-核心概念]] |
| CuTe | CUTLASS 生态中的 layout/tensor algebra 抽象；它描述布局与操作，不等于某个单独 kernel | [[FlashAttention-FA4-CuTeDSL演进]] |
| CuTeDSL | 用 Python DSL 表达 CuTe kernel 并生成/编译实现；当前 FA4 还包含按架构实现与 compile key/cache | [[FlashAttention-FA4-CuTeDSL演进]] |
| JIT/cache | 按输入与实现关键字段编译并缓存可执行 kernel；首次编译延迟与缓存命中后的稳态性能必须分开测 | [[FlashAttention-FA4-CuTeDSL演进]] |

## 复盘

术语表只负责消歧，不负责替源码证明行为。遇到 `block`、`P`、`LSE`、`backend` 这类跨层词，至少补齐“所在包、所在层、对象 shape/生命周期、当前分支”四项上下文。需要建立主线时回到 [[FlashAttention-学习路径]]；需要看证据时进入对应专题的源码走读或数据流文档。

快速自检：若看到 `block_table` 能想到 KV 页映射，看到 `blockIdx` 能先判断 CUDA grid 维度，看到 `rP` 不会说成最终概率，看到 import 失败会先检查包与 ABI，那么这张术语表已经完成任务。
