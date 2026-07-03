---
type: module-moc
module: 03-HTTP-Server
batch: "03"
doc_type: moc
title: "HTTP Server 入口"
tags:
 - sglang/batch/03
 - sglang/module/http-server
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# HTTP Server 入口

> 阶段 I · 地基 | Git：`70df09b`

## 本模块目标

读完本目录下全部文档后，你应能**不打开 `sglang/` 源码目录**，回答：

1. `launch_server` 启动时主进程与子进程各做什么？
2. HTTP 请求（如 `POST /generate`）如何到达 `TokenizerManager`？
3. `Engine` 类与 `launch_server` 函数的关系是什么？
4. FastAPI `lifespan` 在何时初始化 OpenAI/Ollama 等 Serving 处理器？

## 五件套阅读顺序

| 顺序 | 文件 | 一句话说明 |
|------|------|------------|
| 01 | [[03-HTTP-Server-01-核心概念]] | 三进程架构、`_GlobalState`、HTTP vs Engine API |
| 02 | [[03-HTTP-Server-02-源码走读]] | **主文档**：`engine.py` + `http_server.py` 按调用顺序精读 |
| 03 | [[03-HTTP-Server-03-数据流与交互]] | 启动时序、`POST /generate` 请求路径 |
| 04 | [[03-HTTP-Server-04-关键问题]] | 单/多 tokenizer、warmup、API Key 易错点 |
| ✓ | [[03-HTTP-Server-05-checkpoint]] | 验收清单 |

## 源码范围

| 文件 | 职责 |
|------|------|
| `srt/entrypoints/engine.py` | 引擎核心：子进程启动、Python API（`generate`/`encode`） |
| `srt/entrypoints/http_server.py` | FastAPI HTTP 层：路由、全局状态、uvicorn/Granian 监听 |

## 最关键的一段入口代码

**Explain：** 启动链路 中 `run_server` 默认走 HTTP 分支，最终调用 `http_server.launch_server`。该函数先通过 `Engine._launch_subprocesses` 拉起 Scheduler / Detokenizer 子进程并初始化 TokenizerManager，再进入 `_setup_and_run_http_server` 挂载 FastAPI 并阻塞在 uvicorn/Granian。

**Code：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L2471-L2517
# 提交版本：70df09b
def launch_server(
    server_args: ServerArgs,
    init_tokenizer_manager_func: Callable = init_tokenizer_manager,
    run_scheduler_process_func: Callable = run_scheduler_process,
    run_detokenizer_process_func: Callable = run_detokenizer_process,
    execute_warmup_func: Callable = _execute_server_warmup,
    launch_callback: Optional[Callable[[], None]] = None,
):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    # Launch subprocesses
    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager_func,
        run_scheduler_process_func=run_scheduler_process_func,
        run_detokenizer_process_func=run_detokenizer_process_func,
    )

    _setup_and_run_http_server(
        server_args,
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result.scheduler_infos,
        subprocess_watchdog,
        execute_warmup_func=execute_warmup_func,
        launch_callback=launch_callback,
    )
```

**中文释义：** `launch_server` 先启动 SRT 运行时服务：主进程承载 HTTP Server、Engine 与 TokenizerManager；Scheduler 和 DetokenizerManager 是子进程，三者之间通过 ZMQ IPC 通信。

**Code（FastAPI 实例创建 — `_setup_and_run_http_server` 内，见 [[03-HTTP-Server-02-源码走读|源码走读 §2.2]]）：**

```python
## 来源：python/sglang/srt/entrypoints/http_server.py L342
        from sglang.srt.entrypoints.openai.tool_server import NativeToolServer
```

**Comment（`launch_server` 总览）：**

- `Engine._launch_subprocesses` 是**共享启动逻辑**：HTTP 服务与纯 Python `Engine()` 构造器都调用它。
- 主进程保留 TokenizerManager；Scheduler、Detokenizer 在 `mp.Process` 子进程中运行，经 ZMQ IPC 通信。
- FastAPI `app` 在 `_setup_and_run_http_server` 内创建（见上一段 Code），再 `uvicorn.run(app)` 阻塞主线程。

## 与相邻专题衔接

| 方向 | 模块 | 关系 |
|------|------|------|
| 上游 | [[02-启动链路-00-MOC|02-启动链路]] | `run_server` → `launch_server` |
| 下游 | [[04-OpenAI-API-00-MOC|04-OpenAI-API]] | `/v1/chat/completions` 等路由委托给 `OpenAIServing*` |
| 下游 | [[06-TokenizerManager-00-MOC|06-TokenizerManager]] | 所有生成请求最终进入 `tokenizer_manager.generate_request` |

## 下一模块

→ [[04-OpenAI-API-00-MOC|OpenAI API 兼容层]]
