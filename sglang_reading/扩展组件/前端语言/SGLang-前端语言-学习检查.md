---
title: "前端语言 · 学习检查"
type: exercise
framework: sglang
topic: "前端语言"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# 前端语言 · 学习检查

## 你为什么要做这组检查

目标是确认你能把 SGL 程序表达、解释执行和后端请求分开，并解释 prefix cache、batch、fork 如何落到 serving 调用。

## 能力检查

- [ ] 能说明 SGL 的 IR、解释器和 Backend 三层结构。
- [ ] 能追踪 `gen()` → `_execute_gen` → RuntimeEndpoint → `/generate`。
- [ ] 能说明 `SglFunction`、`StreamExecutor`、`RuntimeEndpoint` 的职责。
- [ ] 能解释 tracer、batch、fork 对请求形态和前缀复用的影响。
- [ ] 能判断问题来自前端程序语义、Backend 参数转换还是 SRT 服务。

## 最小验证

操作：

```powershell
rg -n "class SglFunction|class StreamExecutor|class RuntimeEndpoint|_execute_gen|to_srt_kwargs" sglang/python/sglang/lang
```

预期：能把用户函数包装、解释执行和 SRT 参数转换连成一条链。若 API 已生成正确 `/generate` 请求但返回异常，继续到 [[SGLang-HTTP-Server-排障指南]] 或 [[SGLang-TokenizerManager-排障指南]]。

## 复盘

完整调用链见 [[SGLang-前端语言-源码走读]]，对象变化见 [[SGLang-前端语言-数据流]]。
