---
type: index
title: "SGLang 源码阅读指南"
tags:
 - sglang/index
updated: 2026-07-02
---

# SGLang 源码阅读指南

本目录是对 [SGLang](https://github.com/sgl-project/sglang) 源码的**自包含**中文讲解体系。 
**你只需阅读 `sglang_reading/` 目录，不必打开 `sglang/` 源码目录。**

## 核心原则：sglang_reading 即源码

| 原则 | 说明 |
|------|------|
| **自包含** | 每篇文档必须内嵌足够的源码片段 + 逐行/逐段中文讲解，读者无需跳转 |
| **讲解与源码交织** | 采用「先讲意图 → 贴代码 → 再讲细节 → 再贴代码」结构，禁止只有路径/行号引用 |
| **可追溯** | 每段代码标注 sglang 源码路径与行号，便于与 upstream 对照 |
| **概念索引** | 架构图与概念索引见 [[07-总结与索引-00-MOC]] |

> 详细规范见 [[PLAN|§ 六、文档写作规范]]

## 阅读原则

1. **由外向内**：先入口与请求链路，再调度与执行，最后内核与硬件
2. **由主到辅**：优先 `python/sglang/srt` 运行时，再扩展 kernel / gateway / frontend
3. **每模块闭环**：每个专题五篇标准文档 + 内嵌源码，读完后能回答本模块「是什么、怎么跑、和谁交互」
4. **渐进精进**：按 [[PLAN]] 与 [[04-导读路径]] 顺序推进，后篇引用前篇结论（均在 sglang_reading 内完成）

## 快速入口

| 用途 | 文档 |
|------|------|
| **零基础读者** | [[00-零基础先修|00-零基础先修]] — 无 LLM serving 经验先读 |
| 模块与目录对照 | [[10-批次编号对照|10-模块与目录对照]] — 文件夹名 vs 专题名 |
| Onboarding 六件套 | [[07-总结与索引-00-MOC]] → [[01-项目总览]] |
| HTTP 七 hop 全链路 | [[全链路请求追踪|全链路请求追踪]] |
| gRPC 七 hop 全链路 | [[全链路请求追踪-gRPC|全链路请求追踪-gRPC]] |
| 用户故事与场景 | [[07-用户故事与场景|07-用户故事与场景]] |
| 设计追问与框架对比 | [[08-设计追问与框架对比|08-设计追问与框架对比]] |
| 生产排障速查 | [[09-生产排障速查|09-生产排障速查]] |
| 阅读进度 | [[progress]] |

## 目录结构

```
sglang_reading/
├── SGLang源码阅读指南.md # 本文件：总索引
├── PLAN.md # 全专题计划 + 写作规范 + Skills 对齐（维护者）
├── progress.md # 阅读进度追踪（32/32 已完成，维护者）
├── 00-方法论/ # 阅读方法论
├── 01-启动与入口/ # 启动链路–gRPC/Proto · [[01-启动与入口-00-MOC|阶段 I]]
├── 02-请求调度/ # TokenizerManager–Detokenizer · [[02-请求调度-00-MOC|阶段 II]]
├── 03-模型执行/ # ModelRunner–Models 专用 · [[03-模型执行-00-MOC|阶段 III]]
├── 04-内存与Attention/ # RadixAttention–Quantization · [[04-内存与Attention-00-MOC|阶段 IV]]
├── 05-高级特性/ # Sampling–分布式并行、31–32 · [[05-高级特性-00-MOC|阶段 V]]
│ ├── 31-Observability/ # 可观测性 · Prometheus / SchedulerStats
│ └── 32-CheckpointEngine/ # CheckpointEngine · 权重热更新
├── 06-扩展组件/ # Multimodal–multimodal_gen · [[06-扩展组件-00-MOC|阶段 VI]]
└── 07-总结与索引/ # 总结与索引 · onboarding + 索引层
 ├── 00-零基础先修.md # 零基础入口
 ├── 01–06 onboard 六件套
 ├── 全链路请求追踪*.md
 ├── 07-用户故事与场景.md
 ├── 08-设计追问与框架对比.md
 ├── 09-生产排障速查.md
 └── 07-总结与索引-checkpoint.md
```

## 文档版本

内嵌代码对应 sglang Git commit **`70df09b`**。若 upstream 行号漂移，以代码块注释中的路径在 sglang 仓库内搜索函数名核对。

## Obsidian 阅读

本目录已适配 Obsidian vault（根目录 `F:\源码阅读`）。**关系图谱**默认过滤与颜色分组见 `.obsidian/graph.json`；[[obsidian-graph-presets]] 含手动预设。Dataview 仪表盘：[[91_dashboard/home|阅读仪表盘]]。

## 归档说明

[[07-总结与索引/_archive/_archive-MOC|`07-总结与索引/_archive/`]] 为过时草稿，请勿阅读；请使用 onboard 六件套与全链路追踪。

## 当前进度

**32 个专题全部完成** — 维护进度见 [[progress]]（维护者用）

| 模块 | 主题 | 状态 | 文档 |
|------|------|------|------|
| 01 | 项目总览与阅读方法论 | ✅ | [[00-方法论-00-MOC]] |
| 启动链路 | 启动链路与 CLI | ✅ | [[02-启动链路-00-MOC]] |
| HTTP Server | HTTP Server 入口 | ✅ | [[03-HTTP-Server-00-MOC]] |
| OpenAI API | OpenAI API 兼容层 | ✅ | [[04-OpenAI-API-00-MOC]] |
| gRPC/Proto | gRPC 与 Proto | ✅ | [[05-gRPC-Proto-00-MOC]] |
| 06 | TokenizerManager | ✅ | [[06-TokenizerManager-00-MOC]] |
| 07 | Scheduler 核心 | ✅ | [[07-Scheduler-00-MOC]] |
| 08 | 调度策略 | ✅ | [[08-SchedulePolicy-00-MOC]] |
| 09 | Batch 与 IO 结构 | ✅ | [[09-ScheduleBatch-IO-00-MOC]] |
| 10 | Detokenizer 与输出 | ✅ | [[10-Detokenizer-00-MOC]] |
| 11 | ModelRunner 与执行器 | ✅ | [[11-ModelRunner-00-MOC]] |
| 12 | 模型加载 | ✅ | [[12-ModelLoader-00-MOC]] |
| 13 | 通用模型实现 | ✅ | [[13-Models-通用-00-MOC]] |
| 14 | 专用模型实现 | ✅ | [[14-Models-专用-00-MOC]] |
| 15 | RadixAttention 与前缀缓存 | ✅ | [[15-RadixAttention-00-MOC]] |
| 16 | KV Cache 分配与存储 | ✅ | [[16-KV-Cache-00-MOC]] |
| 17 | Attention 后端 | ✅ | [[17-Attention-00-MOC]] |
| 18 | MoE 层 | ✅ | [[18-MoE-00-MOC]] |
| 19 | 量化 | ✅ | [[19-Quantization-00-MOC]] |
| 20 | Sampling 与约束解码 | ✅ | [[20-Sampling-00-MOC]] |
| 21 | 投机解码 | ✅ | [[21-Speculative-00-MOC]] |
| 22 | Prefill-Decode 分离 | ✅ | [[22-Disaggregation-00-MOC]] |
| 23 | 分布式并行 | ✅ | [[23-Distributed-00-MOC]] |
| 24 | 多模态 VLM | ✅ | [[24-Multimodal-00-MOC]] |
| 25 | LoRA | ✅ | [[25-LoRA-00-MOC]] |
| 26 | sgl-kernel | ✅ | [[26-sgl-kernel-00-MOC]] |
| 27 | sgl-model-gateway | ✅ | [[27-model-gateway-00-MOC]] |
| 28 | Frontend 编程接口 | ✅ | [[28-Frontend-lang-00-MOC]] |
| 29 | 扩散模型 runtime | ✅ | [[29-multimodal_gen-00-MOC]] |
| 30 | 全链路复盘与索引 | ✅ | [[07-总结与索引-00-MOC]] |
| 31 | 可观测性 | ✅ | [[31-Observability-00-MOC]] |
| 32 | CheckpointEngine 热更新 | ✅ | [[32-CheckpointEngine-00-MOC]] |

## 推荐阅读顺序

严格按 [[PLAN]] 与 [[04-导读路径]] 从阅读方法论起按专题顺序推进。**只读 `sglang_reading/` 即可**，无需对照 `sglang/` 目录。

新手捷径： [[00-零基础先修|00-零基础先修]] → [[01-项目总览|01-项目总览]] → [[全链路请求追踪|全链路请求追踪]] → [[04-导读路径|04-导读路径 Step 1–8]] → 按需深入各专题目录。
