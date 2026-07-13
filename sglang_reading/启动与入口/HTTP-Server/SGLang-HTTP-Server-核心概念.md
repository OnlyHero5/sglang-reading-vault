---
title: "HTTP-Server · 核心概念"
type: concept
framework: sglang
topic: "HTTP-Server"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# HTTP-Server · 核心概念

这篇先回答：为什么 HTTP Server 看起来很大，却不是推理核心？读完你应该能把源码分成四层：入口选择、运行时拓扑、FastAPI 生命周期、协议路由。

## 先建立模型

把 HTTP Server 想成一个服务前台加门禁：

| 类比 | 源码对象 | 作用 | 失效边界 |
|------|----------|------|----------|
| 前台接待 | FastAPI route | 接收 JSON、选择 native 或 OpenAI handler | 不能解释 batch 调度 |
| 门禁闸机 | warmup 与 health | 决定服务是否从“端口可达”进入“可服务” | 不能证明模型输出正确 |
| 值班总表 | `_GlobalState` | route 能拿到 tokenizer/template/scheduler info | 不能跨进程共享 Python 对象 |
| 后厨传菜口 | `TokenizerManager.generate_request` | 统一进入分词、ZMQ、回包聚合链路 | 后续细节属于 TokenizerManager |

源码里最容易误读的是：HTTP Server 不是一个独立 engine。它借用 `Engine._launch_subprocesses` 点火，再把 HTTP 请求交回 `TokenizerManager`。

## Engine 三组件

`Engine` 的 docstring 直接给出进程边界：HTTP server、Engine、TokenizerManager 在主进程；Scheduler 和 DetokenizerManager 走子进程；进程间通信使用 ZMQ IPC。

```python
# 来源：python/sglang/srt/entrypoints/engine.py L183-L195
class Engine(EngineScoreMixin, EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager, schedules batches, forwards them, and sends the output tokens to the Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and sends the result back to the Tokenizer Manager.

    Note:
    1. The HTTP server, Engine, and TokenizerManager all run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
```

这段给出第一条不变量：HTTP route 不应该直接操作 Scheduler。只要请求要生成、embedding、classify 或 score，入口可以不同，但最终都应该经过主进程的 tokenizer manager 或 OpenAI serving handler，再进入同一条运行时链路。

## `_GlobalState` 保存运行时对象

FastAPI route 是模块级函数，`app` 也是模块级对象。SGLang 用 `_GlobalState` 保存 route 需要访问的运行时对象。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L190-L207
# Store global states
@dataclasses.dataclass
class _GlobalState:
    tokenizer_manager: Union[TokenizerManager, MultiTokenizerRouter, TokenizerWorker]
    template_manager: TemplateManager
    scheduler_info: Dict


_global_state: Optional[_GlobalState] = None


def set_global_state(global_state: _GlobalState):
    global _global_state
    _global_state = global_state


def get_global_state() -> _GlobalState:
    return _global_state
```

单 tokenizer 模式下，`_setup_and_run_http_server` 在 uvicorn 前设置它。多 tokenizer worker 模式下，每个 worker 在 `lifespan` 里从 shared memory 重建自己的 `TokenizerWorker`，再设置同名全局状态。因此 `_GlobalState` 是进程内事实，不是跨 worker 共享对象。

## `app.state` 保存协议 handler

native `/generate` 可以直接走 `_global_state.tokenizer_manager`。OpenAI、Ollama、Anthropic 这类协议要先做格式转换，所以 `lifespan` 会把 handler 挂在 `fast_api_app.state` 上。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L291-L323
    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_classify = OpenAIServingClassify(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_tokenize = OpenAIServingTokenize(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_detokenize = OpenAIServingDetokenize(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_transcription = OpenAIServingTranscription(
        _global_state.tokenizer_manager
    )

    # Initialize Ollama-compatible serving handler
    fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)
```

这里的边界很清楚：route 本身只负责把 typed request 交给 handler，OpenAI messages 到 `GenerateReqInput` 的转换属于 [[SGLang-OpenAI-API]]。

## ServerStatus 是 readiness 账本

端口开始监听不代表模型已经可服务。这里实际上有两层 warmup：`--warmups` 指定的 custom warmups 在 lifespan 中直接 await，完成后才启动 general warmup 线程；general warmup 再通过本机 HTTP 请求检查 `/model_info` 和实际推理入口。显式跳过 general warmup 会直接置 `ServerStatus.Up`。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2145-L2161
def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,
    execute_warmup_func: Callable = _execute_server_warmup,
):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()

    # Send a warmup request
    if not server_args.skip_server_warmup:
        if not execute_warmup_func(server_args):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    # The server is ready for requests
    logger.info("The server is fired up and ready to roll!")
```

源码基线还有一个必须正视的边界：普通模式 warmup 的非 200 或异常会进入 kill-tree 路径；PD disaggregation warmup 非 200 只把状态设为 `UnHealthy`，但 helper 仍返回先前 `/model_info` 阶段的 `success=True`，外层可能继续打印 ready 日志。readiness 排障必须同时看状态、warmup 分支和响应码，不能只搜“ready to roll”。

## 四条主线

| 主线 | 起点 | 终点 | 读源码时抓什么 |
|------|------|------|----------------|
| 启动主线 | `run_server` | uvicorn/Granian 阻塞监听 | `ServerArgs → PortArgs → scheduler_infos → _GlobalState` |
| native 请求主线 | `/generate` | `TokenizerManager.generate_request` | `GenerateReqInput` 是否 stream、abort task 是否挂上 |
| OpenAI 请求主线 | `/v1/chat/completions` | `openai_serving_chat.handle_request` | route 是否只做委托，协议转换在哪里发生 |
| readiness 主线 | custom warmups + general warmup thread | `ServerStatus.Up`、`UnHealthy` 或进程退出 | `/model_info`、模式分支、warmup 响应码、health 时间戳 |

## 常见误解

- “HTTP Server 很厚”：不准确。文件大，是因为协议端点多、运维端点多；推理主逻辑在 TokenizerManager、Scheduler、Detokenizer。
- “`/health` 200 就代表 `ServerStatus.Up`”：不成立。轻量路径只拒绝 graceful exit 和 `Starting`，即使状态是 `UnHealthy` 也可能返回 200。
- “`/health_generate` 200 证明这条探测请求完成”：不准确。它看到探测开始后的任意 `last_receive_tstamp` 更新就成功，繁忙服务中的其他正常回包也可能满足条件。
- “多 worker 只是 uvicorn workers”：不完整。多 tokenizer worker 需要 shared memory 传启动参数，每个 worker 都有自己的 `TokenizerWorker`。
- “`Engine()` 和 `launch_server` 是两套推理”：不准确。它们在 `TokenizerManager.generate_request` 之后共享同一条链路。
- “所有 OpenAI route 都一定有 handler”：不准确。`/v1/responses` handler 初始化被视为 optional，失败只记录 warning，服务本身继续启动。

下一篇 [[SGLang-HTTP-Server-源码走读]] 会把这些判断逐段落到源码分支。
