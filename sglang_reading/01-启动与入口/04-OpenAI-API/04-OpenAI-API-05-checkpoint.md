---
type: batch-doc
module: 04-OpenAI-API
batch: "04"
doc_type: checkpoint
title: "OpenAI API 验收清单"
tags:
 - sglang/batch/04
 - sglang/module/openai-api
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# OpenAI API 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 OpenAI/Ollama 兼容层职责：协议转换 + 流式封装 + 统一错误
- [ ] 能画出 HTTP 路由 → Serving → TokenizerManager 三层位置
- [ ] 能说出 3 个核心类/函数及其职责（文档中均有内嵌代码）：
 - `OpenAIServingBase.handle_request` — 模板方法入口
 - `OpenAIServingCompletion._convert_to_internal_request` — OpenAI → GenerateReqInput
 - `OllamaServing.handle_chat` — Ollama → GenerateReqInput
- [ ] 能追踪 `POST /v1/chat/completions` 从路由到 SSE chunk 的完整路径
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [ ] 内嵌实码 + ETC 讲解（2026-07-02）

- [ ] 对照 knowledge-graph 无遗漏关键 file 节点（openai/*、ollama/*）— **待后续图谱更新 图谱增量补充节点**
- [ ] 来源注释路径/行号与当前 git 一致
- [ ] 已更新 [[progress]] OpenAI API → ✅

## 核心结论（3 句话）

1. **OpenAI 兼容层**采用 `OpenAIServingBase` 模板方法：校验 → `_convert_to_internal_request` → 流式/非流式分支，所有 `/v1/*` handler 共享错误处理与 header 解析。
2. **协议转换的终点**是 `GenerateReqInput` / `EmbeddingReqInput`；Serving 不直接接触 Scheduler，统一经 `TokenizerManager.generate_request` 下发。
3. **Ollama 为平行适配**：独立 protocol + `OllamaServing`，同样构造 `GenerateReqInput`，但流式用 NDJSON、默认 `stream=True` 且 `num_predict` 默认 2048。

## 遗留问题

- `OpenAIServingChat` 全文件（~2000 行）tool/reasoning/multimodal 细节可在Sampling（Sampling/Constrained）交叉阅读
- OpenAI Responses API（`serving_responses.py`）与 tool_server 集成本模块仅列目录，未逐步走读
- knowledge-graph 当前无 openai/ollama 专用节点，待后续图谱更新 图谱增量更新时补充

## 代码块统计（维护者）

| 文件 | 代码块数 | 约行数 |
|------|----------|--------|
| README.md | 1 | 18 |
| 01-核心概念.md | 4 | 95 |
| 02-源码走读.md | 14 | 280 |
| 03-数据流与交互.md | 5 | 75 |
| 04-关键问题.md | 4 | 55 |
| **合计** | **28** | **~523** |

满足 PLAN 要求：≥ 15 代码块、≥ 200 行。

## 3 个核心组件速查

| 组件 | 文件 | 一句话职责 |
|------|------|-----------|
| `OpenAIServingBase` | `openai/serving_base.py` | 所有 OpenAI endpoint 的抽象模板与公共工具 |
| `OpenAIServingCompletion` / `OpenAIServingChat` | `openai/serving_*.py` | 具体 API 的字段映射与响应组装 |
| `OllamaServing` | `ollama/serving.py` | Ollama JSON ↔ GenerateReqInput ↔ Ollama 响应 |
