---
title: "SGLang 总结复盘"
type: map
framework: sglang
topic: "总结复盘"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-12
---
# SGLang 总结复盘

> 这里不是“读完后的附录”，而是把 200 余篇 SGLang 笔记压回一套可用于解释、比较、排障和实验的系统模型。

## 你为什么要读

源码专题容易让人掌握局部，却丢掉全局：会解释一个 attention backend，却说不清请求为什么仍在 waiting；会背 RadixAttention，却不能证明 prefix 真正命中；知道 PD、Spec、LoRA、Graph 都能优化，却不知道它们何时互相挤压资源或改变状态机。

本目录要求你最终同时维护五本账：

| 账本 | 核心问题 |
|---|---|
| 请求账 | 请求现在是什么对象，谁拥有它，下一次交接在哪里？ |
| 资源账 | 权重、KV、workspace、媒体 feature 和队列各占什么资源？ |
| 地址账 | token、request slot、KV loc、prefix node、adapter slot 如何互相映射？ |
| 执行账 | 当前 forward mode、backend、graph、并行组与 kernel 是什么？ |
| 回程账 | token id、logprob、文本 delta、stream chunk、取消和错误怎样返回？ |

## 文档地图

| 任务 | 入口 | 读完要得到什么 |
|---|---|---|
| 判断一段源码为何难改 | [[SGLang-复杂度热点]] | 复杂度来自状态交叉与所有权，不是文件行数 |
| 做 runtime 选型 | [[SGLang-框架对比与设计决策]] | 把“谁更快”改写成可复现实验与总成本决策 |
| 线上第一跳分诊 | [[SGLang-生产排障]] | 从症状落到对象、指标、源码入口、操作和预期 |
| 回答架构常见疑问 | [[SGLang-常见问题]] | 区分 CLI、服务形态、ready、缓存、调度与回程 |
| 补齐未独立成六篇的横切主题 | [[SGLang-补充主题]] | 为 PP、HiCache、connector、平台和 benchmark 建入口 |
| 验收是否真正学会 | [[SGLang-综合学习检查]] | 用口述、静态证据和对照实验完成可判分验收 |

## 三种使用方式

### 复盘

```text
复杂度热点
→ 任选一个跨层对象画完整生命周期
→ 综合学习检查
→ 回到薄弱专题补证据
```

### 生产排障

```text
生产排障第一跳
→ 对应专题排障指南
→ 记录最后一个可信边界
→ 单变量实验
→ 保留被否定的假设
```

### 架构决策

```text
框架对比与设计决策
→ 固定模型/版本/硬件/workload/SLA
→ 证明路径实际生效
→ 正确性基线
→ 延迟、goodput、显存、恢复成本
```

## 总结层禁止的四种说法

- “某功能存在，所以一定更快。”
- “配置写了某 backend，所以实际就用了它。”
- “HTTP 端口可达，所以生成链路 ready。”
- “总量指标正常，所以每个 rank、每个请求和每个阶段都正常。”

总结文档必须比专题更谨慎，因为它会被读者当成跨模块结论。任何跨层判断都应能回到已终审专题、源码卡或可执行实验。

## 通过标准

你不需要背完所有类名，但必须能：

1. 从一个线上症状找到最可能的所有者；
2. 说清它前后两个交接对象；
3. 找到源码入口并说明当前 baseline 的条件分支；
4. 设计只改变一个变量的验证；
5. 知道静态证据、单机实验和真实集群实验各自能证明到哪里。

完成后回到 [[SGLang学习指南]]，把 SGLang 与 [[Slime学习指南]]、[[FlashAttention学习指南]] 串成推理 runtime、训练闭环与 kernel 三层认知。
