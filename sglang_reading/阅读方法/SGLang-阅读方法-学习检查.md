---
title: "阅读方法 · 学习检查"
type: exercise
framework: sglang
topic: "阅读方法"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---

# 阅读方法 · 学习检查

## 你为什么要做这组检查

目标不是记住文档结构，而是确认你能从用户命令建立源码假设，并知道如何用调用链和证据修正它。

## 能力检查

- [ ] 能说明 SGLang 是 LLM/VLM 推理 serving 框架，`python/sglang/srt` 是核心运行时。
- [ ] 能画出配置与入口、请求调度、模型执行、内存与算子之间的边界。
- [ ] 能说明 `cli/main.py`、`cli/serve.py`、`launch_server.run_server` 分别承担什么职责。
- [ ] 能沿 `sglang serve --model-path M` 追到 HTTP 默认入口，并指出 gRPC、Ray、Encoder 模式在哪里分叉。
- [ ] 能说明 `sglang serve` 是推荐 CLI 而非唯一入口，并区分 diffusion parser 与 SRT `ServerArgs`。
- [ ] 能说明 `lang.api.Engine`、包级 `sglang.Engine` 最终指向同一个 SRT Engine 实现。
- [ ] 读到一个结论时，能区分“已有源码证据”“需要继续验证的假设”和“仅用于理解的类比”。

## 最小验证

操作：

```powershell
rg -n "def main|def serve|def run_server|Default mode: HTTP mode" sglang/python/sglang
```

预期：结果能覆盖 CLI 子命令路由、serve 参数转交、运行模式分派和默认 HTTP 分支。若只能找到其中一层，回到 [[SGLang-阅读方法-源码走读]] 补齐调用链。

## 复盘

完成后进入 [[SGLang-启动与入口]]。需要先补 serving 术语时，阅读 [[SGLang-零基础先修]]。
