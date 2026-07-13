---
title: "HTTP-Server · 学习检查"
type: exercise
framework: sglang
topic: "HTTP-Server"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# HTTP-Server · 学习检查

这一页用来检查你是否真的掌握了 HTTP Server，而不是只看过路由列表。

## 读者能做什么

- [ ] 能画出 `run_server → launch_server → Engine._launch_subprocesses → _setup_and_run_http_server → lifespan → route` 主线。
- [ ] 能说明 HTTP Server 为什么不直接调 Scheduler，而要经 `TokenizerManager.generate_request`。
- [ ] 能区分 `_GlobalState`、`app.state`、`ServerStatus` 三类状态分别解决什么问题。
- [ ] 能解释 single tokenizer 与 multi tokenizer worker 在对象传递上的差异。
- [ ] 能说出 `/health` 与 `/health_generate` 的区别，以及什么时候该用哪个。
- [ ] 能解释轻量 `/health` 为什么可能在 `UnHealthy` 时仍返回 200，以及深度 health 为什么不是 request-specific 完成证明。
- [ ] 能区分 custom warmup、general warmup、PD 非 200、checkpoint wait timeout 和 skip warmup 的不同失败强度。
- [ ] 能指出 OpenAI route 与 native route 的汇合点和分叉点。
- [ ] 能根据一个症状找到对应源码入口，而不是全文搜索路由名后停住。

## 口头复述题

1. 一次普通 `sglang serve` 默认为什么会进入 `http_server.launch_server`？
2. `launch_server` 先调用 `Engine._launch_subprocesses`，再调用 `_setup_and_run_http_server`，这个顺序保护了什么不变量？
3. `scheduler_infos[0]["max_req_input_len"]` 为什么要回填到 tokenizer manager？
4. multi-tokenizer worker 为什么要 shared memory？为什么每个 worker 都要自己的 tokenizer IPC name？
5. `/v1/chat/completions` route 为什么不能直接说明 OpenAI 协议转换逻辑？
6. 如果 `/health` 是 200，但 `/health_generate` 是 503，你会怎么缩小问题范围？
7. 为什么日志出现 `ready to roll` 仍不能排除 PD warmup 已把状态设成 `UnHealthy`？
8. multi-tokenizer 模式为什么可能出现多组 warmup 日志？

## 可执行检查

静态检查默认入口：

```powershell
rg -n "Default mode: HTTP mode|from sglang.srt.entrypoints.http_server import launch_server" sglang/python/sglang/launch_server.py
```

预期：命中默认 HTTP 分支。

静态检查启动主线：

```powershell
rg -n "Engine._launch_subprocesses|_setup_and_run_http_server|scheduler_init_result.wait_for_ready" sglang/python/sglang/srt/entrypoints/http_server.py sglang/python/sglang/srt/entrypoints/engine.py
```

预期：能看到 HTTP Server 先拿 engine 产物，再进入 HTTP setup；engine 侧等待 scheduler ready。

静态检查 worker 分叉：

```powershell
rg -n "api_key is not supported in multi-tokenizer|write_data_for_multi_tokenizer|init_multi_tokenizer" sglang/python/sglang/srt/entrypoints/http_server.py
```

预期：能看到 shared memory、worker 初始化和 API key 断言。

静态检查 route 委托：

```powershell
rg -n "generate_request\\(|openai_serving_chat.handle_request|openai_serving_completion.handle_request" sglang/python/sglang/srt/entrypoints/http_server.py
```

预期：native `/generate` 进入 tokenizer manager，OpenAI route 进入 `app.state` handler。

运行中检查 health：

```powershell
curl.exe -s -o NUL -w "%{http_code}" http://127.0.0.1:30000/health
```

预期：服务非 Starting 且未 graceful exit 时，轻量 health 返回 `200`，包括状态已是 `UnHealthy` 的情况。如果要确认探测开始后后端仍有回包，再访问 `/health_generate`；它仍不是该探测请求完整成功的证明。

## 环境边界

真实 HTTP/health 验收需要 Linux/WSL、可导入的 SGLang runtime、模型与对应设备。当前 Windows 环境在包初始化阶段缺少标准库 `resource`，因此只能执行 `rg` 静态检查；不能把 import 失败误判为 FastAPI route 缺失。

## 改代码前的不变量

- HTTP route 不应绕过 `TokenizerManager` 直接访问 Scheduler。
- multi-tokenizer worker 不能假设主进程 Python 对象可共享。
- OpenAI-compatible route 的协议转换应放在 serving handler，不应塞回 route 函数。
- readiness 不应只看端口监听或 ready 日志；warmup 分支、`ServerStatus`、轻量/深度 health 都是强度不同的信号。
- stream 请求断开时要能触发 abort，避免后端继续占用请求状态和 KV 资源。

## 下一步阅读

- 想看 OpenAI messages 如何转成内部请求：[[SGLang-OpenAI-API]]
- 想看 `generate_request` 之后的请求状态机：[[SGLang-TokenizerManager]]
- 想看 Scheduler 接到请求后的 batch 形态：[[SGLang-ScheduleBatch数据结构]]
- 想看 token id 如何回到字符串响应：[[SGLang-Detokenizer]]
