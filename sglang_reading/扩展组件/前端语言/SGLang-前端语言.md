---
title: "前端语言"
type: map
framework: sglang
topic: "前端语言"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-12
---
# 前端语言

> **源码范围：** `python/sglang/lang/` — `api.py`、`ir.py`、`interpreter.py`、`tracer.py`、`backend/` 
> **Git 基线：** `70df09b` 
> **前置专题：** [[SGLang-model-gateway]] · **下一专题：** [[SGLang-多模态生成]]

---

## 1. 本模块目标

专题读法：SGL（Structured Generation Language）是 SGLang 的前端 DSL。`@sgl.function` 只把 Python 函数包装成 `SglFunction` 并检查首参/记录参数名；真正运行函数时，`gen/select/role/image` 等 API 才创建 IR 节点，`ProgramState.__iadd__` 将节点提交给 `StreamExecutor` 立即或排队解释。后端可以是连接既有服务的 `RuntimeEndpoint`、会自行 spawn HTTP server 的 `Runtime` wrapper，或 OpenAI 等 `BaseBackend` 实现。

**源码锚点：**

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

读法：

- `@sgl.function` 不会在装饰时执行函数或生成完整 IR 图；完整依赖图只在 tracing 路径显式记录。
- `num_api_spec_tokens` 是 API 级投机参数：OpenAI chat backend 会积累格式后在 role end 合并请求；completion backend 走超前生成并在前端切片。它与 fork 的 `SglCommitLazy` 不是同一机制，也不能无条件用于 `RuntimeEndpoint`。

---

## 2. 架构位置

```
用户程序 @sgl.function
 │
 ▼
api.py (gen, select, Runtime/Engine 延迟入口) → ir.py (SglFunction / SglGen / SglExpr)
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
| `api.py` | 公开 DSL 与延迟构造入口；`Engine()` 本身不是 SGL `BaseBackend` |
| `ir.py` | 中间表示：`SglGen`、`SglFunction`、`SglSamplingParams` |
| `interpreter.py` | 解释执行、fork/join、batch |
| `tracer.py` | 静态 trace 提取 prefix |
| `backend/` | 后端适配层 |

---

## 3. 自测与验收标准

- [ ] 能写出一个最小 `@sgl.function` 并说明 `s += gen()` 如何变成 HTTP 请求
- [ ] 能解释 `StreamExecutor.submit` → `_execute_gen` → `backend.generate` 调用链
- [ ] 能区分 `RuntimeEndpoint`（连接既有 HTTP 服务）、`Runtime`（spawn HTTP server + endpoint）与 `Engine`（直接 IPC 驱动 SRT、非 SGL Backend）
- [ ] 能运行或静态追踪一个最小 `@sgl.function`，证明 `gen()` 如何经过解释器和 backend 形成真实请求

→ [[SGLang-前端语言-核心概念]] · [[SGLang-前端语言-源码走读]]
