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
updated: 2026-07-12
---

# 前端语言 · 学习检查

## 你为什么要做这组检查

目标是确认你能把 SGL 程序表达、解释执行和后端请求分开，并解释 prefix cache、batch、fork 如何落到 serving 调用。

## 一、闭卷主线

- [ ] 能说明 SGL 的 IR、解释器和 Backend 三层结构。
- [ ] 能追踪 `gen()` → `_execute_gen` → RuntimeEndpoint → `/generate`。
- [ ] 能说明 `SglFunction`、`StreamExecutor`、`RuntimeEndpoint` 的职责。
- [ ] 能解释 tracer、batch、fork 对请求形态和前缀复用的影响。
- [ ] 能判断问题来自前端程序语义、Backend 参数转换还是 SRT 服务。
- [ ] 能说明装饰器、普通 submit 执行与 tracing 建图的不同时间点。
- [ ] 能区分 RuntimeEndpoint、Runtime wrapper 与 Engine，并判断谁可直接作为 SGL backend。
- [ ] 能算出 RuntimeEndpoint select 的两次/三次请求，并说明 unconditional 分支。
- [ ] 能指出 RuntimeEndpoint 单图限制、fork(1) 不前置 commit、`position_ids_offset` 未生效三个边界。
- [ ] 能解释 `_thread_worker_func` 异常清理为什么可能多调用一次 `task_done()`，以及它会怎样遮蔽原始异常。
- [ ] 能指出 `.bind()` 与 fork 为什么会丢失 `num_api_spec_tokens`，以及 SglImage/SglVideo 为什么可能破坏 tracing 图节点假设。
- [ ] 能沿 executor `sid` → generate/commit payload → `concatenate_and_append(src_rids, dst_rid)` 检查 request identity，并指出当前缺失的 `rid` 绑定。

## 二、静态证据链

操作：

```powershell
rg -n "class SglFunction|def __iadd__|class StreamExecutor|class RuntimeEndpoint|_execute_gen|to_srt_kwargs" sglang/python/sglang/lang
rg -n "class Runtime:|self.endpoint = RuntimeEndpoint|class Engine\(|def get_chat_template|def select\(|def generate_stream" sglang/python/sglang/lang/backend/runtime_endpoint.py sglang/python/sglang/srt/entrypoints/engine.py
rg -n "requires_unconditional_logprobs|_generate_http_request|Only support one image|position_ids_offset|fork_program" sglang/python/sglang/lang
rg -n "Clean the queue and events|task_done\(\)|get_nowait\(\)|error_ = error" sglang/python/sglang/lang/interpreter.py
rg -n "def bind|num_api_spec_tokens|TODO.*API speculative|class SglImage|class SglVideo|super\(\).__init__" sglang/python/sglang/lang/ir.py sglang/python/sglang/lang/interpreter.py
rg -n "self.sid|src_rids|dst_rid|concatenate_and_append|\"rid\"" sglang/python/sglang/lang/interpreter.py sglang/python/sglang/lang/backend/runtime_endpoint.py
```

预期：能证明 `SglFunction` 只包装函数，普通 `ProgramState.__iadd__` 走 submit；Runtime wrapper 有 endpoint、Engine 没有 SGL backend 方法；select 存在条件第三请求；RuntimeEndpoint 明确断言单图。若 API 已生成正确 `/generate` 请求但返回异常，继续到 [[SGLang-HTTP-Server-排障指南]] 或 [[SGLang-TokenizerManager-排障指南]]。

## 三、故障推演

### 推演 A：把 `Engine()` 传给 `my_program.run(backend=...)`

合格答案不能只说“都是 SGLang 所以可用”。应指出 `StreamExecutor.__init__` 首先需要 `backend.get_chat_template()`，后续还会调用 `generate(self, StreamExecutor, sampling_params=...)`、`select` 或 `generate_stream`；当前 Engine 的 `generate(prompt, sampling_params, ...)` 是另一套离线接口，缺少其余方法，因此会在进入模型前发生接口错误。

### 推演 B：RuntimeEndpoint select 的延迟突然增加

先数请求：一次 prefix 零 token探测、一次 choices batch logprob；若 method 要 unconditional logprob，再加一次 input-id batch。然后分别观察三次请求的输入规模与后端 queue，不要把整个延迟归到“选择算法的一次请求”。

### 推演 C：某个后台 expr 失败后 `sync()` 不返回

先用 `use_thread=False` 重现原始异常，再检查 worker 清理是否出现 `task_done() called too many times`。前者用于定位原始 expr，后者是当前异常清理的二次故障；两者不能混成一个 SRT 超时。

## 四、无依赖静态实验

在仓库根目录运行：

```powershell
@'
import ast
from pathlib import Path

root = Path("sglang/python/sglang")

def methods(path, cls_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

required = {"get_chat_template", "generate", "generate_stream", "select", "commit_lazy_operations"}
engine = methods(root / "srt/entrypoints/engine.py", "Engine")
endpoint = methods(root / "lang/backend/runtime_endpoint.py", "RuntimeEndpoint")
print("Engine missing:", sorted(required - engine))
print("RuntimeEndpoint missing:", sorted(required - endpoint))
'@ | python -
```

预期：Engine 至少缺 `get_chat_template`、`generate_stream`、`select`、`commit_lazy_operations`；RuntimeEndpoint 缺失集合为空。该实验只证明接口形状，不证明真实服务、GPU 或 OpenAI API 可用。

## 复盘

完整调用链见 [[SGLang-前端语言-源码走读]]，对象变化见 [[SGLang-前端语言-数据流]]。
