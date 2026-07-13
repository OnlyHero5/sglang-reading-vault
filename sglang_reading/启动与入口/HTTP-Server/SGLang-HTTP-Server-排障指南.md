---
title: "HTTP-Server · 排障指南"
type: troubleshooting
framework: sglang
topic: "HTTP-Server"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# HTTP-Server · 排障指南

这篇按排障症状组织。每个问题都给出源码入口和验证方式，避免只记结论。

## 快速定位表

| 症状 | 先查源码入口 | 常见原因 | 验证 |
|------|--------------|----------|------|
| `/health` 返回 503 | `health_generate` | graceful exit、`ServerStatus.Starting`，或实际走了深度探活且超时 | 查 path、环境变量、时间戳和状态 |
| `/health` 200 但状态是 UnHealthy | 轻量 health 早返回 | 轻量路径不检查 UnHealthy | 改用 `/health_generate` 并结合业务探针 |
| 服务端口开了但不 ready | `_wait_and_warmup`、`_execute_server_warmup` | custom/general warmup 未完成，或分支失败语义不同 | 查 `/model_info`、响应码、状态和 ready 日志 |
| `--tokenizer-worker-num > 1` 加 API key 启动失败 | `init_multi_tokenizer` | multi-tokenizer 模式不支持 API key | 搜 assert 文本 |
| OpenAI route 报 handler 缺失 | `lifespan` | lifespan 未完成；`/v1/responses` 还可能是 optional handler 初始化失败 | 查 app state 和 Responses warning |
| 非 rank 0 节点没有完整 API | `_launch_subprocesses` | node rank 只跑 scheduler/dummy health | 查 `node_rank >= 1` 分支 |
| 客户端断开后日志出现 `ValueError` | `/generate` stream 分支 | 正常取消路径或 abort 未传播 | 查 `request.is_disconnected()` |
| 多 worker SSL refresh 不生效 | server run 分支 | 明确不支持 | 查 warning |

## 1. `launch_server` 和 `Engine()` 何时选哪个？

对外提供 HTTP/OpenAI/Ollama/Anthropic 兼容服务，用 `launch_server`。在 Python 脚本、测试或上层框架内嵌推理，用 `Engine()`。两者不是两套推理核心；差别在是否启动 FastAPI 和 ASGI server。

判断边界：

- `launch_server` = engine 子进程启动 + HTTP setup + ASGI server 阻塞监听。
- `Engine()` = engine 子进程启动 + Python 方法包装。
- 不要在同一进程里同时创建 `Engine()` 和 `launch_server()`，否则会重复拉起运行时拓扑。

验证入口：

```powershell
rg -n "def launch_server|def generate\\(" sglang/python/sglang/srt/entrypoints/http_server.py sglang/python/sglang/srt/entrypoints/engine.py
```

预期：能看到 HTTP `launch_server` 和 Python `Engine.generate` 都接到 `TokenizerManager.generate_request`。

## 2. 为什么用 `_GlobalState`，不是所有 route 都用依赖注入？

SGLang 的 `app` 和 route 是模块级对象，多 worker 模式下 worker 会重新 import 这个模块。运行时对象不能靠主进程 Python 引用跨 worker 共享，所以源码选择：

- single tokenizer：HTTP setup 先 `set_global_state`，再启动 uvicorn。
- multi tokenizer：worker lifespan 从 shared memory 读参数，创建自己的 `TokenizerWorker`，再 `set_global_state`。

排障规则：

- 直接 import `app` 做单测时，如果没有走 `_setup_and_run_http_server` 或 lifespan，`_global_state` 为空是预期风险。
- 生产请求中如果 `_global_state` 为空，先查服务是否绕过了正常启动链。

## 3. `/health` 和 `/health_generate` 有什么区别？

`/health` 默认是低成本探活；`/health_generate` 会构造一次最小请求，确认后端有响应。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L588-L599
    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)
```

如何选择：

| 场景 | 建议 |
|------|------|
| K8s liveness | 用轻量 `/health` |
| readiness 或升级后验证 | 用 `/health_generate` |
| 高负载服务 | 避免每秒深度生成探活 |
| disaggregation 模式 | 注意 health 会补 bootstrap fake 字段 |

验证：

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:30000/health
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:30000/health_generate
```

预期：前者只证明服务非 Starting 且未 graceful exit；后者要求探测开始后出现后端响应。注意它观察共享 `last_receive_tstamp`，不保证该 health 请求自身完成。

## 4. 哪些 warmup 失败会杀进程树？

general warmup 先等 `/model_info`，如果 120 次轮询内都失败，源码直接 `kill_process_tree(os.getpid())`。普通非 PD 请求发生异常或非 200 也会由 assert 进入同一异常路径。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2007-L2010
    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success
```

但不是所有失败都 kill：

- PD disaggregation warmup 非 200 只设置 `ServerStatus.UnHealthy`；当前 helper 仍返回先前的 `success=True`，外层可能继续打印 ready。
- `checkpoint_engine_wait_weights_before_ready` 等待超时只记录 error，然后继续 general warmup。
- `--skip-server-warmup` 直接把状态设为 Up，不执行上述 HTTP 请求。

排查顺序：

1. 先看 `/model_info` 是否能返回 200。
2. 再看 warmup 选择的是 `/generate`、`/encode` 还是 VLM chat completions。
3. 对照 `disaggregation_mode` 判断失败是 kill、UnHealthy 还是继续。
4. 如果只是开发调试，可显式跳过 warmup；生产环境应修复模型加载、handler、CUDA graph 或后端链路问题。

## 5. 为什么 multi-tokenizer 模式不支持 API key？

multi-tokenizer worker 要在 worker 进程中从 shared memory 重建 tokenizer worker。源码明确断言 `server_args.api_key is None`。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L224-L227
    # API key authentication is not supported in multi-tokenizer mode
    assert (
        server_args.api_key is None
    ), "API key is not supported in multi-tokenizer mode"
```

部署建议：

- 需要内置 API key：使用 single tokenizer mode。
- 需要多 worker 吞吐：把鉴权放到 gateway、ingress 或上游代理。
- 如果启动时报这条 assert，不要去查 OpenAI route，问题发生在 worker 初始化阶段。

## 6. 客户端断开为什么不一定是服务端错误？

streaming `/generate` 中，客户端断开也会表现为 `ValueError`。源码先检查 `request.is_disconnected()`，如果已断开就记录日志并停止发送错误响应。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L802-L809
            except ValueError as e:
                # A client disconnect also surfaces here. It's a client-side
                # cancellation, not a server error or bad input -- log it and
                # stop (the request was already aborted upstream) instead of
                # emitting a 400.
                if request is not None and await request.is_disconnected():
                    logger.info(f"[http_server] Client disconnected: {e}")
                    return
```

判断方法：

- 只有 client disconnect 日志，且请求被 abort，是正常取消。
- 如果没有断开却进入 ValueError，则 route 会返回 invalid request error，应继续查请求参数或 TokenizerManager 抛错。
- 如果断开后 GPU/KV 资源没有释放，去查 `create_abort_task` 和 TokenizerManager 的 abort 传播。

## 7. 非 rank 0 节点为什么没有完整 HTTP API？

多机情况下，`node_rank >= 1` 的节点只需要 scheduler，不需要 tokenizer 或 detokenizer。源码等待 scheduler ready 后，要么返回给 Python API，要么启动 dummy health server 并等待完成。

```python
# 来源：python/sglang/srt/entrypoints/engine.py L835-L861
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
            )
```

部署时客户端应访问 rank 0 或统一 gateway，不要把非 rank 0 节点当作完整 OpenAI API server。

## 8. HTTP/2、uvicorn、SSL refresh 如何分叉？

server run 分支按配置选择：

| 配置 | 路径 |
|------|------|
| `tokenizer_worker_num == 1` 且 `enable_http2` | embedded Granian HTTP/2 |
| single tokenizer 且 `enable_ssl_refresh` | uvicorn Config/Server API |
| single tokenizer 默认 | `uvicorn.run` 直接接收内存中的 `app` |
| multi tokenizer 且 `enable_http2` | Granian multi worker |
| multi tokenizer 默认 | `uvicorn.run("sglang.srt.entrypoints.http_server:app", workers=N)` |

multi-tokenizer 不支持 SSL refresh，源码会 warning 后禁用。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2425-L2430
            if server_args.enable_ssl_refresh:
                logger.warning(
                    "--enable-ssl-refresh is not supported with multiple "
                    "tokenizer workers (--tokenizer-worker-num > 1). "
                    "SSL refresh will be disabled."
                )
```

排查 HTTP server 后端时，先确认实际进入了哪个分支，再看对应 server 的日志。

## 9. OpenAI 路由和 native 路由能混用吗？

可以。它们挂在同一个 FastAPI app 上：

- native `/generate` 直接进入 `_global_state.tokenizer_manager.generate_request`。
- OpenAI `/v1/chat/completions` 先进入 `openai_serving_chat.handle_request`，再由 handler 转成内部请求。

底层 Scheduler 看的是 token 和请求参数，不关心 HTTP 路由原名。真正影响缓存、采样、LoRA、priority 的，是内部请求字段和 TokenizerManager/Scheduler 下游逻辑。

例外是 `/v1/responses`：route 总会注册，但对应 handler 初始化被包在 try/except 中。初始化失败只 warning 并让服务继续，随后访问该 route 可能表现为 app state 缺字段；这不等价于整个 lifespan 没运行。

验证：

```powershell
python -c "from sglang.srt.entrypoints.http_server import app; print(sorted({getattr(r, 'path', '') for r in app.routes if getattr(r, 'path', '').startswith('/v1')})[:8])"
```

预期：能看到 `/v1/completions`、`/v1/chat/completions`、`/v1/models` 等 route。若本地环境缺依赖导致 import 失败，用 `rg -n "@app.post\\(\"/v1" sglang/python/sglang/srt/entrypoints/http_server.py` 静态确认。

## 10. `/v1/models` 为什么和生成请求排障思路不同？

`/v1/models` 只读 tokenizer manager 的模型名、context length 和 LoRA registry，不进入 Scheduler 生成链路。它成功不代表生成成功，它失败也可能只是 `_global_state` 或 LoRA registry 状态问题。

排查方式：

- `/v1/models` 失败：查 `_global_state.tokenizer_manager`、served model name、LoRA registry。
- `/v1/chat/completions` 失败：查 OpenAI handler 转换和 TokenizerManager 生成链路。
- `/generate` 失败：跳过 OpenAI handler，直接查 native 请求与 TokenizerManager。

## 11. multi-tokenizer 模式为什么可能重复 warmup？

每个 HTTP worker 都会执行自己的 lifespan。因而 multi-tokenizer 模式下，每个 worker 都会初始化协议 handler、执行 `--warmups` 指定的 custom warmups，并启动 general warmup thread。看到多组 warmup 日志或并发最小请求时，先按 worker pid/label 区分，不要直接判断为重复启动 engine。

内置 API-key middleware 只在 single-tokenizer 分支安装；multi-tokenizer 只显式 assert `api_key is None`。若使用 `admin_api_key` 或外部鉴权，必须在实际部署拓扑中验证端点保护效果，不能从参数存在推断 middleware 已安装。

## 复盘

HTTP Server 排障要先分类：启动链、worker 初始化、route 委托、readiness、下游推理。尤其不要把某一条 ready 日志、某一次轻量 health 或某个 handler 的存在扩大成全服务健康证明。
