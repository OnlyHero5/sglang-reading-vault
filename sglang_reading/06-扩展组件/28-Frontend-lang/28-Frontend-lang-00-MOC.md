---
type: module-moc
module: 28-Frontend-lang
batch: "28"
doc_type: moc
title: "Frontend Language（SGL）"
tags:
 - sglang/batch/28
 - sglang/module/frontend-lang
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# Frontend Language（SGL）

> **源码范围：** `python/sglang/lang/` — `api.py`、`ir.py`、`interpreter.py`、`tracer.py`、`backend/` 
> **Git 基线：** `70df09b` 
> **前置专题：** [[27-model-gateway-00-MOC|27-model-gateway]] · **下一专题：** [[29-multimodal_gen-00-MOC|29-multimodal_gen]]

---

## 1. 本模块目标

**Explain：** SGL（Structured Generation Language）是 SGLang 的**前端 DSL**：用户用 Python 函数编写 prompt 程序，通过 `@sgl.function` 装饰器定义 IR，由 `StreamExecutor` 解释执行并调用 Backend（RuntimeEndpoint / OpenAI / Anthropic 等）完成实际推理。本模块覆盖从 `gen()` API 到 HTTP 调 srt 的完整链路。

**Code：**

```python
# 来源：python/sglang/lang/api.py L23-L32
def function(
    func: Optional[Callable] = None, num_api_spec_tokens: Optional[int] = None
):
    if func:
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    def decorator(func):
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    return decorator
```

**Comment：**

- `@sgl.function` 将普通 Python 函数包装为 `SglFunction`，首参必须是 `s`（ProgramState）。
- `num_api_spec_tokens` 启用 API 级投机执行（lazy commit）。

---

## 2. 架构位置

```
用户程序 @sgl.function
 │
 ▼
api.py (gen, select, Runtime) → ir.py (SglGen, SglExpr)
 │
 ▼
interpreter.py (StreamExecutor, ProgramState)
 │
 ▼
backend/*.py (RuntimeEndpoint → HTTP /generate)
 │
 ▼
srt HTTP server（或 OpenAI 等外部 API）
```

| 模块 | 职责 |
|------|------|
| `api.py` | 公开 API：`gen`、`Runtime`、`Engine` |
| `ir.py` | 中间表示：`SglGen`、`SglFunction`、`SglSamplingParams` |
| `interpreter.py` | 解释执行、fork/join、batch |
| `tracer.py` | 静态 trace 提取 prefix |
| `backend/` | 后端适配层 |

---

## 3. 验收标准

- [ ] 能写出一个最小 `@sgl.function` 并说明 `s += gen()` 如何变成 HTTP 请求
- [ ] 能解释 `StreamExecutor.submit` → `_execute_gen` → `backend.generate` 调用链
- [ ] 能说明 `RuntimeEndpoint` 与 `Engine` 的区别
- [ ] 五篇正文 ≥ 15 段 ETC，合计 ≥ 200 行内嵌源码

→ [[28-Frontend-lang-01-核心概念]] · [[28-Frontend-lang-02-源码走读]]
