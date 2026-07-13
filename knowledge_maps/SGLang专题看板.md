---
title: "SGLang 专题看板"
type: dashboard
framework: sglang
topic: "SGLang"
learning_role: reference
tags:
  - framework/sglang
  - content/dashboard
  - source-reading
updated: 2026-07-13
---

# SGLang 专题看板

## 阅读路径

首次阅读只走 [[推理Serving主线]]，再按问题进入专题：

- 请求入口：[[SGLang-启动与入口]]
- 调度与连续批处理：[[SGLang-请求调度]]
- 模型执行：[[SGLang-模型执行]]
- KV 与 Attention：[[SGLang-内存与Attention]]
- 分布式与生产特性：[[SGLang-高级特性]]
- 生产排障：[[SGLang-生产排障]]

## 动态内容

![[SGLang内容.base]]

## 使用建议

按 `topic` 分组浏览；同一主题内优先读入口图和核心概念，需要改代码再进入源码走读与数据流，出现故障时直接打开排障指南。

看板中的模块关系是阅读路由，不表示固定进程数或唯一 worker loop；普通/overlap、PP、PD、DP 和平台路径应回到对应专题确认。
