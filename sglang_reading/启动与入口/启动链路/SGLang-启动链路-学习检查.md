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
updated: 2026-07-11
---
# 启动链路 · 学习检查

这一页检查你是否能把 `sglang serve` 从 shell 命令追到 runtime 分支。

## 读者能做什么

- [ ] 能画出 `console script → cli.main → serve → prepare_server_args → ServerArgs → run_server` 主线。
- [ ] 能解释为什么根命令不解析 `--model-path`。
- [ ] 能说明 `--model-type` 为什么属于 `serve()`，不属于 `ServerArgs`。
- [ ] 能区分 `extra_argv`、`dispatch_argv`、`raw_args`、`ServerArgs` 四种形态。
- [ ] 能解释 `--config` 如何在 parse 前转成 CLI 参数，以及为什么 CLI 显式参数优先。
- [ ] 能解释为什么 `--config=prod.yaml` 不触发当前合并器，以及为什么 `model_path` 不能只放在 YAML。
- [ ] 能说明插件发现、插件执行、hook apply 的先后顺序。
- [ ] 能区分“当前入口尽早加载插件”和“各进程必须在开始服务前完成 apply”两种表述。
- [ ] 能判断 `encoder_only`、`grpc_mode`、`use_ray` 如何影响 `run_server` 分支。
- [ ] 能说明普通 `grpc_mode` 是 legacy SMG 分支，不是 native Rust gRPC runtime 接线。
- [ ] 能说出启动链路和 [[SGLang-HTTP-Server]] 的交界在哪里。

## 口头复述题

1. `sglang serve --model-path M --tp-size 2 --port 8080` 中，哪些参数由根 parser 处理，哪些留给 `serve()`？
2. 如果传了 `--model-type diffusion`，为什么它不会进入 LLM `prepare_server_args`？
3. diffusion 自动检测失败为什么默认回到 LLM 路径？
4. `ServerArgs.add_cli_args` 哪些参数自动生成，哪些必须手写？
5. `ServerArgs.__post_init__` 为什么可能在 HTTP Server 启动前报错？
6. 哪些约束在 `__post_init__`，哪些还要等 engine 调用 `check_server_args()`？
7. 插件 hook 为什么要在 engine starts serving 前 apply？为什么这不等价于“target 模块绝不能提前 import”？
8. `encoder_only=True` 和 `grpc_mode=True` 同时存在时会走哪条分支？`grpc_mode=True` 与 `use_ray=True` 呢？

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

确认 config 覆盖顺序，而不导入完整 SGLang runtime：

```powershell
@'
import argparse, importlib.util, pathlib, tempfile

path = pathlib.Path("sglang/python/sglang/srt/server_args_config_parser.py")
spec = importlib.util.spec_from_file_location("config_parser", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=30000)
parser.add_argument("--config")

with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    f.write("port: 30001\n")
    config_path = f.name

merged = module.ConfigArgumentMerger(parser).merge_config_with_args(
    ["--config", config_path, "--port", "30002"]
)
print(merged)
print(parser.parse_args(merged).port)
pathlib.Path(config_path).unlink()
'@ | python -
```

预期：合并后的 argv 中 config 的 `30001` 在前、显式 CLI 的 `30002` 在后，最终打印 `30002`。

确认模型路径窥探发生在 config 合并之前：

```powershell
rg -n "model_path = get_model_path|server_args = prepare_server_args" sglang/python/sglang/cli/serve.py
```

预期：`get_model_path(dispatch_argv)` 的行号更小。因此 YAML-only `model_path` 无法帮助这次早期分发。

## 环境边界

完整执行 `sglang serve --help` 需要可导入 SGLang runtime 的 Linux/WSL 安装环境。当前 Windows 环境会在包初始化阶段因缺少标准库 `resource` 失败；这类失败不证明 CLI 逻辑错误，应先用上面的源码定位和独立 config 脚本验收，再到 Linux 环境执行真实 CLI help。

## 改代码前的不变量

- 根 parser 不应直接解析完整 serve 参数。
- serve 私有参数必须在进入 LLM parser 前剥离。
- `--config` 必须在 argparse parse 前合并。
- 当前 config 语法必须使用 `--config FILE`；模型路径仍须在 CLI 上供早期分发读取。
- CLI 显式参数应覆盖 config。
- 各相关进程必须在开始服务前完成插件加载与 hook apply；尽早加载是当前入口顺序，不是禁止 target 提前 import 的绝对规则。
- `run_server` 分支顺序不能随意调换，特别是 `encoder_only` 必须优先。
- `grpc_mode` 的 legacy SMG 分支不能与 native Rust gRPC 能力混写。
- `PortArgs` 属于 runtime 派生对象，不应混进 CLI 参数解析。

## 下一步阅读

- 默认 HTTP 入口内部：[[SGLang-HTTP-Server]]
- OpenAI route 如何转换请求：[[SGLang-OpenAI-API]]
- gRPC 协议入口：[[SGLang-gRPC-Proto]]
- 请求进入调度前台：[[SGLang-TokenizerManager]]
