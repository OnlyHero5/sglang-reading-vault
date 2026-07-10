---
title: "SGLang 导读与总览"
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
# SGLang 导读与总览

> SGLang 阅读第一站：零基础先修、项目总览、全链路追踪与查阅索引。代码基线 `70df09b`。

---

## 本目录定位

本目录解决“从哪里开始读”的问题。它不要求先读完专题，也不假设读者熟悉 LLM serving。读者应先在这里建立三件事：

1. SGLang 做什么，HTTP 请求如何进入推理栈。
2. Prefill / Decode / KV Cache / Scheduler 等核心词是什么意思。
3. 读完总览后，应该进入哪个专题目录继续深读。

**收尾复盘、排障和复杂度热点**已经移到 [[SGLang-总结复盘]]。

---

## 快速入口

| 用途 | 文档 |
|------|------|
| 零基础 | [[SGLang-零基础先修]] |
| 项目总览 | [[SGLang-项目总览]] |
| HTTP 全链路 | [[SGLang-HTTP请求全链路]] |
| gRPC 全链路 | [[SGLang-gRPC请求全链路]] |
| 语义学习路线 | [[SGLang-学习路径]] |
| 文件地图 | [[SGLang-源码地图]] |
| 术语 | [[SGLang-术语表]] |
| 双库续读 | [[SGLang与Slime-阅读对照]] |

---

## 文档地图

### Onboarding（建议顺序）

| 顺序 | 文档 | 内容 |
|------:|------|------|
| 0 | [[SGLang-零基础先修]] | Prefill / Decode、KV Cache、连续批处理 |
| 1 | [[SGLang-项目总览]] | 项目定位、启动链、三进程模型 |
| 2 | [[SGLang-架构分层]] | 分层架构与代表代码 |
| 3 | [[SGLang-关键概念]] | 核心概念与源码示例 |
| 4 | [[SGLang-HTTP请求全链路]] | HTTP 请求职责边界串讲 |
| 5 | [[SGLang-学习路径]] | 按系统职责组织的学习路线 |

### 查阅索引

| 文档 | 内容 |
|------|------|
| [[SGLang-源码地图]] | 按层定位源码文件 |
| [[SGLang-术语表]] | 术语与代码出处 |
| [[SGLang-模块依赖图]] | 模块关系图 |
| [[SGLang-业务流程]] | 业务域流程 |
| [[SGLang-用户场景]] | 生产场景叙事 |
| [[SGLang-图谱使用说明]] | 图谱预设跳转 |

---

## 推荐阅读路径

**完全新手：** [[SGLang-零基础先修]] → [[SGLang-项目总览]] → [[SGLang-HTTP请求全链路]] → [[SGLang-学习路径]]

**有 serving 基础：** [[SGLang-项目总览]] → [[SGLang-HTTP请求全链路]] → [[SGLang-请求调度]] 或 [[SGLang-内存与Attention]]

**双库阅读：** [[SGLang-HTTP请求全链路]] → [[Slime-零基础先修]] → [[Slime-RL训练全链路]]

---

## 下一站

| 目标 | 下一站 |
|------|--------|
| 看请求怎么跑 | [[SGLang-HTTP请求全链路]] |
| 看调度 | [[SGLang-请求调度]] |
| 看 KV / Attention | [[SGLang-内存与Attention]] |
| 看生产排障 | [[SGLang-生产排障]] |
| 看复盘材料 | [[SGLang-总结复盘]] |
