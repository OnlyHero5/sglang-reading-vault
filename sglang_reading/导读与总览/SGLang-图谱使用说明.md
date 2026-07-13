---
title: "SGLang 图谱使用说明"
type: reference
framework: sglang
topic: "导读与总览"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/reference
  - source-reading
updated: 2026-07-10
---
# SGLang 图谱使用说明

## 你为什么要读

SGLang 的难点不是“笔记在哪个文件夹”，而是同一个请求同时连接协议入口、Scheduler、KV、ModelRunner、Attention backend 和输出回程。图谱适合看这些语义邻居，但不负责替你安排课程顺序。

首次学习仍从 [[SGLang学习指南]] 开始；当你已经有一条主线，想回答“这个对象的上游和下游是谁”“这个故障还可能牵涉哪些专题”时，再打开 Backlinks 或 Local Graph。

## 三种任务，三种用法

| 读者任务 | 首选功能 | 为什么 |
|----------|----------|--------|
| 按顺序学习 | [[SGLang学习指南]] + Bookmarks | 顺序稳定，不会被高连接度节点带偏 |
| 追一个对象或请求 | Local Graph + Backlinks | 能看到直接生产者、消费者与专题解释 |
| 找故障邻域 | Search 定位错误文本，再开 Local Graph | Search 找精确入口，图谱补相关系统边界 |

不要一上来打开 Global Graph。全局图更适合发现孤岛和异常簇；初学者容易把“链接多”误认为“应先学习”。

## 推荐起点

### 请求主线

从 [[SGLang-HTTP请求全链路]] 打开 Local Graph，深度先设为 1。你应该看到协议入口、TokenizerManager、Scheduler、ScheduleBatch、ModelRunner、Detokenizer 等邻居。

阅读时沿这组问题移动：

1. 请求此刻由谁持有？
2. 下一跳传递的是协议对象、`Req`、batch、tensor 还是输出消息？
3. 哪个对象跨 step 存活，哪个只是本轮执行视图？
4. 出错时哪个日志、metric 或源码入口能证明消息停在这里？

### 资源主线

从 [[SGLang-Scheduler]]、[[SGLang-RadixAttention]] 或 [[SGLang-KV-Cache]] 起步。不要只看三者互相连接，还要区分：

| 节点 | 主要责任 | 不应混淆为 |
|------|----------|------------|
| Scheduler | 排序、准入、组 batch、结果消费 | KV 数据本身 |
| RadixAttention | prefix key、匹配、节点保护与驱逐 | attention kernel |
| KV Cache | 请求行、KV 地址、物理 pool 与释放 | prefix 命中语义 |
| Attention | backend、metadata 与 KV 消费 | 请求准入策略 |

当 Local Graph 同时出现 [[SGLang-ModelRunner]] 与 [[SGLang-Attention]] 时，沿 `ForwardBatch → runner view → backend metadata` 继续，而不是从“显存不足”直接跳到某个 kernel。

### 生产特性分支

| 想研究的分支 | Local Graph 起点 | 回到主线时要守住的边界 |
|--------------|------------------|------------------------|
| PD 分离 | [[SGLang-PD分离]] | 请求身份、bootstrap、KV transfer 与输出顺序 |
| 投机解码 | [[SGLang-Speculative]] | draft/verify/accept 与目标模型状态对齐 |
| LoRA | [[SGLang-LoRA]] | adapter 身份、CPU/GPU 驻留与 batch capacity |
| 多模态 | [[SGLang-多模态]] | 原始输入、processor output、embedding 与 token 对齐 |
| MoE | [[SGLang-MoE]] | token、expert id、dispatcher、placement 与 scaling owner |
| 量化 | [[SGLang-Quantization]] | 配置、layer method、loader、postprocess 与最终 kernel |

## SGLang 专用过滤式

只看 SGLang 学习内容：

```text
[framework:sglang] path:sglang_reading -path:模板
```

只看核心学习主线：

```text
[framework:sglang] [learning_role:core]
```

只看源码走读：

```text
[framework:sglang] [type:walkthrough]
```

只看排障与练习：

```text
[framework:sglang] ([type:troubleshooting] OR [type:exercise])
```

不要用裸 `-path:sglang` 排除 upstream；它可能误伤名称相近的 `sglang_reading`。需要隐藏源码目录时，应使用界面验证过的精确路径条件。

## Backlinks 应该怎样读

Backlinks 不只表示“有人提到本页”。对源码阅读笔记，它通常表达四类语义：

- 前置知识：读本页前需要理解什么；
- 上游生产者：谁创建当前对象或状态；
- 下游消费者：谁读取、改写或释放它；
- 验证入口：哪个实验、排障或学习检查会观察它。

如果一个核心文档只有目录页入链，没有任何上游、下游或验证页，应把它视为知识关系可能不足，而不是图谱“看起来很干净”。

## 一次完整使用示例

症状：流式请求连接仍在，但长时间没有文本。

1. 用 Search 查错误文本、`rid` 或相关日志，不用图谱猜字符串位置。
2. 从 [[SGLang-HTTP请求全链路]] 打开 Local Graph。
3. 依次查看 Scheduler 是否产出 token、Detokenizer 是否产出 text、TokenizerManager 是否唤醒等待事件。
4. 如果 token id 已产生，把故障域收缩到 Detokenizer、decode offset、stop trimming、worker IPC 或 event notify；不要先改 Attention kernel。
5. 跳到 [[SGLang-Detokenizer-排障指南]] 执行有操作和预期的检查。

图谱在这里负责提醒“还有哪些边界”，证据仍来自日志、metric、源码和实验。

## 使用边界

- Local Graph 深度默认 1；只有追跨专题关系时才升到 2。
- 图谱边的存在不证明运行时一定走该分支，配置和实际对象必须另行验证。
- 高连接度不等于高优先级；入口页和术语页天然会有更多链接。
- 文件夹表达物理归档，双链表达语义关系，Properties/Bases 表达可筛选结构。
- 不为“让图更漂亮”制造无意义链接。

跨框架用法见 [[关系图谱指南]]；动态入口见 [[知识地图首页]]。

## 静态验证

操作：在仓库根目录运行：

```powershell
$targets = @(
  'SGLang-HTTP请求全链路',
  'SGLang-Scheduler',
  'SGLang-RadixAttention',
  'SGLang-KV-Cache',
  'SGLang-ModelRunner',
  'SGLang-Detokenizer-排障指南'
)

foreach ($target in $targets) {
  rg -n --fixed-strings "[[$target]]" sglang_reading knowledge_maps AI-Infra课程
  if ($LASTEXITCODE -ne 0) { throw "missing wikilink: $target" }
}
```

预期：六个主线节点都至少有一处语义链接。这个检查只证明关系入口存在；是否链接得恰当仍需结合 Backlinks 和正文语义判断。

## 复盘

学习顺序交给课程与指南，精确定位交给 Search，直接依赖交给 Backlinks，局部关系交给 Local Graph。把工具分工守住，图谱才能帮助你在 SGLang 的宏观架构和微观对象之间来回切换，而不会变成另一张难以阅读的目录树。
