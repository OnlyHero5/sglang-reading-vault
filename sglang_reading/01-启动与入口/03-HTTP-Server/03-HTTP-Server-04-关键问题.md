---
type: batch-doc
module: 03-HTTP-Server
batch: "03"
doc_type: faq
title: "HTTP Server：关键问题"
tags:
 - sglang/batch/03
 - sglang/module/http-server
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# HTTP Server：关键问题

> 常见问题、易错点与设计取舍。

---

## Q1：`launch_server` 和 `Engine()` 应该什么时候用哪个？

**Explain：** 对外部署 OpenAI 兼容 HTTP 服务时必须走 `launch_server`，它包含 FastAPI、健康检查与 metrics 端点。在 Python 脚本或 Ray driver 内嵌推理时，`Engine()` 更轻、无 HTTP 序列化开销。两者共用 `_launch_subprocesses` 子进程拓扑，但**不应**在同一进程里既构造 `Engine()` 又调用 `launch_server()`，否则会重复拉起 Scheduler。

**Comment：** 选型口诀：对外 API → `launch_server`；Notebook/单测/批处理脚本 → `Engine()`。重复启动是最常见集成错误之一。

---

## Q2：为什么用模块级 `_global_state` 而不是 FastAPI Depends？

**Explain：** 历史原因 + 多 worker 模式：uvicorn multi-worker 会 fork/import 模块级 `app`，每个 worker 需在 `lifespan` 里独立初始化 TokenizerWorker。模块级全局变量 + `set_global_state` 比跨 worker 共享 Depends 单例更简单。

**Code（单 worker 在 uvicorn 前注入）：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2281-L2288
# 提交版本：70df09b
    # Set global states
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )
```

**易错：** 在测试中直接 import `app` 并发请求，若未调用 `set_global_state`，`_global_state` 为 `None` 会 AttributeError。

**Comment：** 单测应 mock `set_global_state` 或使用 `Engine()` 绕过 HTTP 层；生产 multi-worker 时每个 worker 在 lifespan 内独立注入。

---

## Q3：`/health` 与 `/health_generate` 有什么区别？

| 端点 | 行为 |
|------|------|
| `/health` | 默认仅检查进程存活（200）；设 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=1` 才做生成探测 |
| `/health_generate` | 始终尝试 1 token 生成探测 |

**Explain：** K8s liveness 常用轻量 `/health`；readiness 可用 `/health_generate` 确认 Scheduler+Detokenizer 链路通。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L595-L599
# 提交版本：70df09b
    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)
```

**Comment：** 轻量 `/health` 避免 busy 时做无意义 1-token 生成；readiness 探针用 `/health_generate` 确认 Scheduler+Detokenizer 链路通。

---

## Q4：warmup 失败为什么 `kill_process_tree`？

**Explain：** warmup 在独立线程用 `requests` 调本机 HTTP。若模型/graph 编译失败，服务处于半初始化状态；直接杀进程树避免对外暴露不可用实例。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2007-L2009
# 提交版本：70df09b
    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
```

**正确做法：** 开发调试可 `--skip-server-warmup` 跳过；生产环境应修根因而非仅跳过。

**Comment：** warmup 失败说明模型/graph/Detokenizer 链路未就绪；对外暴露半初始化实例会导致 502 或 silent hang。

---

## Q5：单 tokenizer vs 多 tokenizer worker

| 配置 | HTTP 服务器 | Tokenizer | API Key |
|------|-------------|-----------|---------|
| `tokenizer_worker_num=1`（默认） | uvicorn 单进程 或 Granian embedded | 主进程 TokenizerManager | 支持 |
| `tokenizer_worker_num>1` | uvicorn `workers=N` 或 Granian multi-worker | 每 worker 一个 TokenizerWorker | **不支持** |

**易错写法：** 在多 worker 模式配置 `--api-key`，启动 assert 失败。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L224-L227
# 提交版本：70df09b
    # API key authentication is not supported in multi-tokenizer mode
    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"
```

**Comment：** 多 worker 模式下 API Key 校验无法跨进程共享同一 secret 状态；需 Gateway 层统一 auth 或退回单 worker。

---

## Q6：HTTP/2（Granian）与 uvicorn 如何选择？

**Explain：** `server_args.enable_http2` 为真时用嵌入式 Granian（HTTP/2）；单 worker 时直接 serve 内存中的 `app` 对象，复用已初始化的 `_global_state`。多 worker Granian 则加载字符串模块路径，走 multi-tokenizer 路径。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2223-L2226
# 提交版本：70df09b
    Server = GranianEmbeddedServer if tokenizer_worker_num == 1 else Granian
    target = (
        app if tokenizer_worker_num == 1 else "sglang.srt.entrypoints.http_server:app"
    )
```

**Comment：** 单 worker Granian 复用内存 `app` 与 `_global_state`；多 worker 必须字符串 import 路径以便 fork 后各 worker 独立 lifespan。

---

## Q7：流式 `/generate` 客户端断开怎么处理？

**Explain：** 客户端断开会触发 `ValueError`；HTTP 层检测 `request.is_disconnected()` 后静默结束，**不**向已断开连接写 400 错误体。

**Code（正确：区分 disconnect vs 真错误）：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L802-L809
# 提交版本：70df09b
            except ValueError as e:
                # A client disconnect also surfaces here. It's a client-side
                # cancellation, not a server error or bad input -- log it and
                # stop (the request was already aborted upstream) instead of
                # emitting a 400.
                if request is not None and await request.is_disconnected():
                    logger.info(f"[http_server] Client disconnected: {e}")
                    return
```

**易错理解：** 把 disconnect 日志当成服务端 bug；实际是正常取消路径，`create_abort_task` 会在 background 里 abort 上游请求。

**Comment：** 客户端断开应释放 Scheduler 侧 KV 与 req 槽位；若 abort 未触发，检查 `request.is_disconnected()` 分支是否被绕过。

---

## Q8：多机 `node_rank >= 1` 的 worker 节点跑什么 HTTP？

**Explain：** 非 rank-0 节点不初始化 Tokenizer/Detokenizer，Scheduler 加载完后若未设 `SGLANG_BLOCK_NONZERO_RANK_CHILDREN=0`，会挂 **dummy health check server** 并阻塞在 `wait_for_completion`，不提供完整推理 HTTP API。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/engine.py L835-L860
# 提交版本：70df09b
        if server_args.node_rank >= 1:
            # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            scheduler_init_result.wait_for_ready()

            if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
                # When using `Engine` as a Python API, we don't want to block here.
                return (
                    None,
                    None,
                    port_args,
                    scheduler_init_result,
                    None,
                )

            launch_dummy_health_check_server(
                server_args.host, server_args.port, server_args.enable_metrics
            )

            scheduler_init_result.wait_for_completion()
            return (
                None,
                None,
                port_args,
                scheduler_init_result,
                None,
```

**Comment：** TP/PP 非 rank-0 节点不提供完整 HTTP API；客户端应只访问 rank-0 或 Gateway。dummy health 仅用于 K8s 探活子进程存活。

---

## Q9：OpenAI 路由和 Native 路由能否混用？

**可以。** 同一 FastAPI `app` 同时暴露 `/generate` 与 `/v1/chat/completions`；前者直达 TokenizerManager，后者经 Serving 层做模板与格式转换（OpenAI API）。底层 Scheduler 无区别。

**Comment：** 混用不影响 RadixCache 前缀共享——cache key 基于 token 序列，与 HTTP 路由无关。

---

## Q10：与 vLLM「API server + engine core」的对比

| 维度 | SGLang 本模块 | 常见 vLLM 模式 |
|------|-------------|----------------|
| HTTP 框架 | FastAPI | FastAPI / 自定义 |
| 主进程组件 | TokenizerManager + HTTP | 常为 API 进程 + engine 分离 |
| IPC | ZMQ + msgspec | 类似 ZMQ/共享内存 |
| Python API | 一等公民 `Engine` 类 | 亦有但入口侧重 HTTP |

SGLang 把 TokenizerManager 固定在主进程，HTTP 极薄——这是读源码时的关键 mental model。

**Comment：** vLLM 部分部署将 API 与 engine 拆进程；SGLang 选择「厚 TM + 薄 HTTP」，利于 OpenAI 兼容与 ZMQ 调度一体化。

---

## 验证建议（零基础可试）

以下 **4 条**前 3 条零 GPU（读源码/列路由），第 4 条需已启动服务（可用最小模型或 `--skip-server-warmup` 调试）。

1. **操作：** 在已安装 sglang 的环境执行 
 `python -c "from sglang.srt.entrypoints.http_server import app; print('\n'.join(sorted({getattr(r,'path',None) for r in app.routes if getattr(r,'path',None) and str(getattr(r,'path')).startswith('/'))))"` 
 **预期现象：** 同时出现 `/generate`、`/v1/chat/completions`、`/health` 等路径——Native 与 OpenAI 路由挂在同一 FastAPI `app`。 
 **对应文档节：** Q9 OpenAI 与 Native 混用

2. **操作：** 在 sglang 源码树执行 
 `rg "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION|health_generate" python/sglang/srt/entrypoints/http_server.py -n` 
 **预期现象：** 命中 `/health` 轻量返回 200 的分支，以及 `/health_generate` 始终做 1-token 探测的逻辑行号。 
 **对应文档节：** Q3 `/health` vs `/health_generate`

3. **操作：** `rg "api_key is not supported in multi-tokenizer" python/sglang/srt/entrypoints/http_server.py -n` 
 **预期现象：** 命中 `tokenizer_worker_num>1` 时对 `--api-key` 的 assert——多 worker 模式不支持 API Key。 
 **对应文档节：** Q5 单/多 tokenizer worker

4. **操作（需服务已 listen）：** 
 `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:30000/health` 
 **预期现象：** 打印 `200`（默认不做生成探测）；若需验证 Scheduler 链路，再试 `/health_generate`（会触发 1-token 生成，需 GPU）。 
 **对应文档节：** Q3 探针选型、Q4 warmup 失败与 `--skip-server-warmup`

---
