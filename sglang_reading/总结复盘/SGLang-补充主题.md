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
updated: 2026-07-12
---
# SGLang 补充主题

## 你为什么要读

有些能力没有独立六篇专题，不代表它们不重要；它们往往横切多个已终审专题。本文不复制局部源码，而是给出“问题→对象→阅读路径→验证边界”，防止横切主题被塞进错误的单一模块。

## 导读表

| 主题 | 为什么横切 | 建议主线 | 最小验证 |
|---|---|---|---|
| Pipeline Parallelism | 同时改变进程拓扑、Scheduler loop、ModelRunner 输入输出和 sampling 所有权 | [[SGLang-分布式-核心概念]] → [[SGLang-Scheduler-源码走读]] → [[SGLang-ModelRunner-数据流]] | 记录每个 PP rank 的输入/输出对象与最后 rank 的 sampling |
| HiCache | 把 device KV、host pool、Radix node 与 storage backend 连成多层生命周期 | [[SGLang-KV-Cache-数据流]] → [[SGLang-RadixAttention-源码走读]] → [[SGLang-PD分离-核心概念]] | 记录 device hit、host hit、load-back、eviction 与 storage 状态 |
| Remote KV / connector | 涉及 prefix 身份、远端存储或传输、请求 metadata 与失败恢复 | [[SGLang-KV-Cache-排障指南]] → [[SGLang-PD分离-数据流]] | 注入 miss、timeout、partial failure，确认不会把未完成 transfer 当 success |
| CUDA Graph / compile cache | 既受 shape/metadata 约束，也受 backend、硬件和模型分支影响 | [[SGLang-ModelRunner-源码走读]] → [[SGLang-Attention-排障指南]] | 证明 capture/replay 实际命中，并与 eager 做数值对照 |
| 平台后端 | CUDA、ROCm、NPU、CPU、XPU/MPS 的默认 backend 与可用优化不同 | [[SGLang-分布式-排障指南]]、[[SGLang-Attention-排障指南]]，再读 upstream `docs/platforms/` | 在目标平台记录最终 ServerArgs、backend、kernel 和 fallback |
| Benchmark 套件 | benchmark 只产生观测，不自动构成选型结论 | [[性能指标与实验方法]]、[[SGLang服务实验]]，再读 upstream `benchmark/` | 固定版本、模型、硬件、到达过程和正确性基线 |
| Native gRPC / 外围组件 | 实现文件存在不等于默认 HTTP 已接线，gateway/legacy wrapper/native listener 所有权不同 | [[SGLang-启动链路]]、[[SGLang-HTTP-Server-源码走读]]、[[SGLang-model-gateway]] | 从真实启动调用点证明 listener、sidecar 与 downstream 协议 |

## 1. Pipeline Parallelism

PP 不是“把层平均切开”一句话能解释完。至少要分清：

- global rank、PP rank 与 TP/DP/CP 组合坐标；
- Scheduler 是走普通、overlap、PP 还是 PD+PP loop；
- 非末 rank 返回的是下一 stage 所需 proxy/hidden state，不负责最终 sampling；
- micro-batch、async depth、bubble 与禁用 overlap 的配置关系；
- abort、streaming 和统计最终由谁收口。

**环境限制**：单进程静态阅读只能证明分支和对象类型，不能证明跨节点 pipeline 时序、bubble 或通信性能。

## 2. HiCache

把 HiCache 理解成“KV 放到 CPU”会漏掉关键状态。一次 prefix 可能同时涉及：

```text
Radix 逻辑节点
↔ device KV page
↔ host backup / consumer index
↔ storage backend object
↔ load-back / eviction / write policy
```

不同 layout 与 I/O backend 会在 `ServerArgs` 中被自动归一化；ROCm、Mooncake storage、page-first/direct 等组合还会被改写。排障时记录最终值，不要只抄启动命令。

## 3. Remote KV 与 connector

远端 KV 的核心不是“能传 tensor”，而是三个协议：

1. **身份协议**：这份 KV 属于哪个 token prefix、权重版本、adapter 或请求 namespace？
2. **完成协议**：metadata、buffer、transfer 与消费分别何时 ready？
3. **失败协议**：超时、partial transfer、worker 重启后，谁释放本地和远端资源？

如果没有这三项，吞吐实验没有意义，因为错误命中和资源泄漏也可能看起来“很快”。

## 4. CUDA Graph 与 compile cache

Graph/compile 优化必须同时满足：

- 输入 shape 可归桶；
- 非 shape metadata 也适合复用；
- backend 与模型路径支持 capture；
- fallback 可观测；
- eager 与 replay 数值一致。

多模态 ViT graph 仅按总 `S` 取 key 的风险说明：shape 相同不等于语义布局相同。普通 LLM graph 同样应检查 position、KV metadata、spec info、DP padding 等动态对象。

## 5. 平台后端

本库以 CUDA serving 为主线，但总结不能把 CUDA 结论外推到所有平台。平台差异至少可能改变：

- attention/sampling/MoE 默认 backend；
- collective 与 all-reduce 实现；
- page size、graph、dtype 与量化支持；
- fallback 是启动时报错、自动改写还是运行时分支；
- profiler 中实际 kernel 名称。

因此非 CUDA 问题必须在目标平台验证；本库提供对象和证据方法，不替代平台兼容表。

## 6. Benchmark 的证据等级

| 证据 | 能证明 | 不能证明 |
|---|---|---|
| 静态源码 | 某分支、参数和对象存在 | 目标环境实际走到该分支 |
| 微基准 | 单 kernel/单组件在特定 shape 的性能 | 端到端队列、传输和 tail latency |
| 离线固定并发 | 稳态容量与资源曲线 | 真实突发到达、取消和恢复 |
| 在线回放 | 目标 workload 的系统表现 | 未来版本、另一模型或另一硬件 |
| 故障注入 | 超时、回收、恢复协议 | 未注入的故障组合 |

## 何时必须直接打开 upstream

- 修改或适配代码；
- 行号、函数签名或行为与笔记冲突；
- 使用非 CUDA 平台、实验性 backend 或新模型；
- 对性能、正确性或生产故障作最终判断；
- 笔记只给出横切地图，没有覆盖当前组合。

入口仍是当前 baseline 下的 `sglang/`，而不是网络上的另一个版本。完成补课后用 [[SGLang-综合学习检查]] 验收。
