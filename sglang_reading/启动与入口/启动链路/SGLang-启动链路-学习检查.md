---
title: "启动链路 · 学习检查"
type: exercise
framework: sglang
topic: "启动链路"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 启动链路 · 学习检查

这一页检查你是否能把 `sglang serve` 从 shell 命令追到 runtime 分支。

## 读者能做什么

- [ ] 能画出 `console script → cli.main → serve → prepare_server_args → ServerArgs → run_server` 主线。
- [ ] 能解释为什么根命令不解析 `--model-path`。
- [ ] 能说明 `--model-type` 为什么属于 `serve()`，不属于 `ServerArgs`。
- [ ] 能区分 `extra_argv`、`dispatch_argv`、`raw_args`、`ServerArgs` 四种形态。
- [ ] 能解释 `--config` 如何在 parse 前转成 CLI 参数，以及为什么 CLI 显式参数优先。
- [ ] 能说明插件发现、插件执行、hook apply 的先后顺序。
- [ ] 能判断 `encoder_only`、`grpc_mode`、`use_ray` 如何影响 `run_server` 分支。
- [ ] 能说出启动链路和 [[SGLang-HTTP-Server]] 的交界在哪里。

## 口头复述题

1. `sglang serve --model-path M --tp-size 2 --port 8080` 中，哪些参数由根 parser 处理，哪些留给 `serve()`？
2. 如果传了 `--model-type diffusion`，为什么它不会进入 LLM `prepare_server_args`？
3. diffusion 自动检测失败为什么默认回到 LLM 路径？
4. `ServerArgs.add_cli_args` 哪些参数自动生成，哪些必须手写？
5. `ServerArgs.__post_init__` 为什么可能在 HTTP Server 启动前报错？
6. 插件 hook 为什么要在 engine starts serving 前 apply？
7. `encoder_only=True` 和 `grpc_mode=True` 同时存在时会走哪条分支？

## 可执行检查

确认 console script：

```powershell
rg -n "sglang = \"sglang.cli.main:main\"" sglang/python/pyproject.toml
```

预期：命中 `python/pyproject.toml` 的 project scripts。

确认根命令只透传：

```powershell
rg -n "parse_known_args|serve\\(args, extra_argv\\)" sglang/python/sglang/cli/main.py
```

预期：看到 `extra_argv` 被传给 `serve()`。

确认 `--model-type` 是 dispatcher hint：

```powershell
rg -n "def _extract_model_type_override|model_type, dispatch_argv" sglang/python/sglang/cli/serve.py
```

预期：命中剥离函数和 `dispatch_argv`。

确认 LLM 参数工厂：

```powershell
rg -n "def prepare_server_args|ServerArgs.add_cli_args|ServerArgs.from_cli_args" sglang/python/sglang/srt/server_args.py
```

预期：看到 `prepare_server_args` 先注册 CLI，再 parse，再构造 `ServerArgs`。

确认 runtime 分支：

```powershell
rg -n "encoder_only|grpc_mode|use_ray|Default mode: HTTP mode" sglang/python/sglang/launch_server.py
```

预期：看到 `run_server` 的四路分支。

确认插件入口：

```powershell
rg -n "SGLANG_PLUGINS|SGLANG_PLATFORM|HookRegistry.apply_hooks|def load_plugins" sglang/python/sglang/srt/plugins
```

预期：看到插件白名单、平台过滤和 apply hooks。

## 改代码前的不变量

- 根 parser 不应直接解析完整 serve 参数。
- serve 私有参数必须在进入 LLM parser 前剥离。
- `--config` 必须在 argparse parse 前合并。
- CLI 显式参数应覆盖 config。
- 插件加载应早于 runtime 入口 import。
- `run_server` 分支顺序不能随意调换，特别是 `encoder_only` 必须优先。
- `PortArgs` 属于 runtime 派生对象，不应混进 CLI 参数解析。

## 下一步阅读

- 默认 HTTP 入口内部：[[SGLang-HTTP-Server]]
- OpenAI route 如何转换请求：[[SGLang-OpenAI-API]]
- gRPC 协议入口：[[SGLang-gRPC-Proto]]
- 请求进入调度前台：[[SGLang-TokenizerManager]]
