---
title: "SGLang 综合学习检查"
type: exercise
framework: sglang
topic: "总结复盘"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# SGLang 综合学习检查

## 你为什么要做

“读过”不能证明你能处理真实系统。本页用对象图、反例、静态证据和对照实验验收五本账：请求、资源、地址、执行、回程。

## 第一关：请求账

不看笔记，画出普通生成请求的形态变化：

```text
外部协议对象
→ GenerateReqInput
→ TokenizedGenerateReqInput
→ Req
→ ScheduleBatch
→ ForwardBatch
→ LogitsProcessorOutput / SampleOutput
→ BatchTokenIDOut
→ text delta / API chunk
```

为每条箭头标注：函数调用、ZMQ/pickle、GPU forward、事件唤醒或 HTTP/gRPC serialization。

**通过标准**：能说明 `rid` 为什么不是 batch 下标、KV 地址或 DP rank；能指出 abort 和 streaming 回程在哪里分叉。

## 第二关：资源账与地址账

解释下面五组对象为什么不能混用：

| 逻辑对象 | 物理/索引对象 |
|---|---|
| request | req pool slot |
| token position | KV token/page loc |
| radix node | 节点引用的 device/host indices |
| LoRA identity | CPU cache entry / GPU slot |
| media item | hash/pad / feature / embedding |

**操作**：任选 KV、LoRA 或多模态专题，写出一次分配、命中、复制、淘汰和释放的所有者。

**预期**：不会用“缓存命中”代替物理 slot 已分配，也不会把 pad value、adapter id 或 request id 当成 tensor 地址。

## 第三关：执行账

回答：

1. 当前 batch 的 `ForwardMode` 是什么？
2. prefill/decode backend 是否可能不同？
3. 配置名、resolved backend、wrapper 与最终 kernel 分别在哪里观测？
4. graph capture/replay 使用了哪些 shape 与 metadata？
5. TP/PP/DP/CP/EP 的 group alias 如何决定 collective scope？

**静态操作**：

```powershell
rg -n 'class ForwardMode|def get_attention_backends|def init_forward_metadata|def initialize_model_parallel' sglang/python/sglang/srt
```

**预期**：能定位 mode、backend resolution、metadata owner 和并行组初始化；静态命中不冒充真实 kernel 已运行。

## 第四关：回程账

给出三个故障的第一跳：

- GPU 已产生 token id，但客户端没有文本；
- token 文本正确，但 streaming delta 重复；
- OpenAI finish reason 与内部 finish 状态不一致。

**通过标准**：分别检查 Scheduler 输出、Detokenizer decode window、TokenizerManager/API adapter，而不是一律回到 attention kernel。

## 第五关：条件式特性

分别写出 RadixCache、Speculative、PD、LoRA、CUDA Graph、多模态的：

```text
收益假设
→ 生效条件
→ 新增状态/资源
→ 失败面
→ 证明路径实际生效的观测
→ 单变量对照
```

若答案中出现固定收益百分比、跨 workload 的统一阈值或“开启即生效”，本关不通过。

## 第六关：生产分诊

任选一个症状：

- 长期 waiting；
- retract 激增；
- TTFT 尖刺；
- prefix 命中下降；
- Graph/backend 未生效；
- PD transfer 卡住；
- token 已出但无文本；
- 热更新失败。

按以下模板作答：

```text
症状与时间窗：
模型/版本/硬件/workload：
最后一个可信观测：
当前对象与所有者：
第一条假设：
源码入口：
操作：
预期：
实际结果：
结论或下一边界：
```

**通过标准**：操作只改变一个变量；假设被否定时保留证据，不靠继续堆参数“碰运气”。

## 第七关：完成一个对照实验

从 [[SGLang服务实验]] 选择一项，至少记录：

- commit、镜像、依赖和配置；
- 模型、权重、tokenizer、dtype/quant；
- GPU、驱动、CUDA、互联；
- 输入/输出长度、并发、到达过程、prefix 重复度；
- 正确性结果；
- P50/P95/P99 TTFT/TPOT、goodput、错误率和显存；
- 实际 backend/kernel/feature 生效证据；
- 环境限制与不可外推范围。

## 口述终验

在五分钟内回答：

> 一个请求为什么可能“HTTP 已接收、Scheduler 未准入、KV 有逻辑命中、GPU 尚未执行、token 已采样、客户端仍无文本”？请用五本账说明每个阶段的对象、资源与交接。

能完整回答并完成一项对照实验，才算真正通过；文档数量和阅读时长不作为标准。
