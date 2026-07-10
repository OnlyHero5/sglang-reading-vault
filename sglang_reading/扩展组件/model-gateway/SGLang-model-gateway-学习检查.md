---
title: "model-gateway · 学习检查"
type: exercise
framework: sglang
topic: "model-gateway"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# model-gateway · 学习检查

## 你为什么要做这组检查

目标是确认你能把网关控制面与 SRT 推理数据面分开，并从一个 OpenAI 请求追到 worker 选择和反向代理。

## 能力检查

- [ ] 能说明 model-gateway 是 Rust Axum 代理与路由层，不执行 GPU forward。
- [ ] 能画出 Client → handler → Router → WorkerRegistry/Policy → SRT worker。
- [ ] 能说明 `AppState`、`RouterTrait::route_chat`、`select_worker_for_model` 的职责。
- [ ] 能解释 regular 与 PD 路由对 worker readiness 的不同要求。
- [ ] 能从超时、无可用 worker 或错误路由症状定位到网关还是 SRT。

## 最小验证

操作：

```powershell
rg -n "route_chat|select_worker_for_model|WorkerRegistry|readiness|RouterFactory" sglang/sgl-model-gateway
```

预期：能找到 handler 委托、worker 注册或选择，以及 readiness/路由构造证据。若请求已经被选到 worker 但生成失败，应继续进入 SRT 排障，而不是停留在 gateway。

## 复盘

主链见 [[SGLang-model-gateway-源码走读]]，跨服务数据流见 [[SGLang-model-gateway-数据流]]。
