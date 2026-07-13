---
title: "启动链路 · 排障指南"
type: troubleshooting
framework: sglang
topic: "启动链路"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# 启动链路 · 排障指南

这篇按症状排查启动链路。先判断问题发生在哪个对象阶段，再去对应源码入口。

## 快速定位表

| 症状 | 先查源码入口 | 可能原因 | 验证 |
|------|--------------|----------|------|
| `sglang --help` 看不到 `--model-path` | `cli/main.py` | 根命令只显示子命令 | 改跑 `sglang serve --help` |
| `--model-type` 进了 LLM parser | `_extract_model_type_override` | 分发 hint 没被剥离 | 查 `dispatch_argv` |
| 不带模型路径启动失败 | `get_model_path` | `--model-path` 或 `--model` 缺失 | 静态 grep error 文本 |
| config 没按预期覆盖 | `ConfigArgumentMerger` | YAML 转 argv 后顺序误解，或使用了 `--config=...` | 打印合并后的 argv |
| YAML 里有 `model_path` 仍报必填 | `get_model_path` | 模型族分发早于 config 合并 | 把 `--model-path` 留在 CLI |
| 插件没生效 | `load_plugins`、`HookRegistry.apply_hooks` | 白名单、平台过滤、apply 时机 | 查 Loaded/Executed/Applied 日志 |
| Ray 模式启动 ImportError | `run_server` | 未装 Ray extra | 只在 `use_ray=True` 分支触发 |
| `python -m sglang.launch_server` 行为和 `sglang serve` 不同 | 旧入口 | 缺少模型族自动检测和双 help | 优先用 `sglang serve` |

## 1. 为什么 `sglang --help` 不显示 `--model-path`？

因为根命令只注册子命令，`serve` 子命令关闭自己的 root-level help，具体服务参数由 `cli/serve.py` 再展示。

```python
# 来源：python/sglang/cli/main.py L15-L26
    # complex sub commands
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    subparsers.add_parser(
        "serve",
        help="Launch an SGLang server.",
        add_help=False,
    )
    subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,
    )
```

验证：

```powershell
sglang --help
sglang serve --help
```

预期：前者看到子命令，后者才看到 LLM 与 diffusion 相关 serve 参数。

## 2. 旧入口还能用吗？

能用于 LLM 兼容路径，但新入口是 `sglang serve`。旧入口加载插件、解析 `ServerArgs`、调用 `run_server`，但没有 `cli/serve.py` 的模型族检测和 serve help。

```python
# 来源：python/sglang/launch_server.py L63-L72
    from sglang.srt.plugins import load_plugins

    load_plugins()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
```

建议：新部署、脚本和文档都用 `sglang serve`。只有确认自己不需要 diffusion 分发和新版 help 时，才考虑旧入口兼容。

## 3. `--model-type` 为什么不是 `ServerArgs` 字段？

`--model-type` 只是 serve dispatcher 用来选择 LLM 或 diffusion 的 hint。LLM parser 不应该看到它。

排查方法：

```powershell
rg -n "def _extract_model_type_override|--model-type" sglang/python/sglang/cli/serve.py
```

预期：命中 `cli/serve.py`，而不是 `ServerArgs` 字段定义。

## 4. model path 缺失时为什么很早失败？

`serve()` 在进入完整 LLM parser 前必须先拿到 `--model-path` 判断模型族，所以缺少 model path 会在 `get_model_path` 里失败。

```python
# 来源：python/sglang/cli/utils.py L122-L127
        else:
            raise Exception(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path
```

注意：这不是模型加载失败，而是 CLI 分发前置条件失败。

## 5. `--config` 和 CLI 显式参数谁优先？

源码注释和合并顺序都指向：CLI 显式参数优先于 config，config 优先于默认值。

```python
# 来源：python/sglang/srt/server_args_config_parser.py L52-L83
    def merge_config_with_args(self, cli_args: List[str]) -> List[str]:
        """
        Merge configuration file arguments with command-line arguments.

        Configuration arguments are inserted after the subcommand to maintain
        proper precedence: CLI > Config > Defaults

        Args:
            cli_args: List of command-line arguments

        Returns:
            Merged argument list with config values inserted

        Raises:
            ValueError: If multiple config files specified or no config file provided
        """
        config_file_path = self._extract_config_file_path(cli_args)
        if not config_file_path:
            return cli_args

        config_data = self._parse_yaml_config(config_file_path)
        config_args = self._convert_config_to_args(config_data)

        # Merge config args into CLI args
        config_index = cli_args.index("--config")

        # Split arguments around config file
        before_config = cli_args[:config_index]
        after_config = cli_args[config_index + 2 :]  # Skip --config and file path

        # Simple merge: config args + CLI args
        return config_args + before_config + after_config
```

失败模式：

- 多个 `--config` 会抛 ValueError。
- `--config` 后没有路径会抛 ValueError。
- 文件不是 `.yaml` 或 `.yml` 会抛 ValueError。
- 根节点不是 dict 会抛 ValueError。
- 当前只支持 `--config FILE`；`--config=FILE` 不会进入 merge 分支，YAML 内容不会生效。
- `serve()` 在 config merge 前需要命令行模型路径；只在 YAML 中写 `model_path` 仍会报 `--model-path is required`。

## 6. 插件加载失败会不会阻断启动？

单个 plugin load 或 execute 失败会记录异常并继续；hook apply 失败也会按 target 记录异常。是否影响业务取决于该 hook 是否是你依赖的修改。还要注意 `_plugins_loaded` 在真正发现插件前就被置为 True：同一进程后续再次调用不会自动重试失败项。

```python
# 来源：python/sglang/srt/plugins/__init__.py L119-L122
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
```

```python
# 来源：python/sglang/srt/plugins/__init__.py L79-L86
        try:
            func = ep.load()
            plugins[ep.name] = (func, dist_name)
            logger.info("Loaded plugin %s from group %s", ep.name, group)
        except Exception:
            logger.exception("Failed to load plugin %s from group %s", ep.name, group)

    return plugins
```

排查顺序：

1. 查 `Available plugins` 和 `Loaded plugin`。
2. 查是否被 `SGLANG_PLUGINS` 白名单跳过。
3. 查是否被 `SGLANG_PLATFORM` 排除。
4. 查 `Executed general plugin`。
5. 查 `Applied hook` 或 `Failed to apply hooks`。

不要用“目标模块已经 import，所以 hook 必然失效”作为结论。registry 会解析目标并传播到一部分旧 `from import` 绑定；真正应核对的是该进程是否在开始服务前执行了 `load_plugins()`，以及目标是否出现在 Applied/Failed 日志中。

## 7. 为什么 Ray 缺依赖只在 `--use-ray` 时失败？

`run_server` 采用延迟 import。只有 `server_args.use_ray=True` 时才导入 Ray HTTP server；缺依赖时给出安装提示。

```python
# 来源：python/sglang/launch_server.py L36-L46
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
```

这也是启动链路中多处延迟 import 的共同目的：没走到的模式，不应该强迫用户安装对应依赖。

## 8. `encoder_only` 和 `grpc_mode` 同时为真时走哪？

先命中 `encoder_only`，再在 encoder 内部选择 gRPC 或 HTTP encoder server。它不会走普通 LLM gRPC。

验证：

```powershell
rg -n "if server_args.encoder_only|elif server_args.grpc_mode" sglang/python/sglang/launch_server.py
```

预期：`encoder_only` 出现在 `grpc_mode` 前。

同理，`grpc_mode=True` 与 `use_ray=True` 同时出现时先走 legacy SMG gRPC，Ray 分支被前一个 `elif` 遮住。组合参数不是“同时开启两套 server”。

## 9. 为什么 `ServerArgs` parse 成功后还会报配置错误？

`argparse` 只把字符串变成字段；第一批语义处理在 `ServerArgs.__post_init__`。例如 session radix cache、PD、DCP、SSL、ASR、模型配置与 backend 默认值会在这里处理。另一批约束位于 `check_server_args()`，由后续 engine 初始化调用。

排查方法：

- 如果错误发生在 `prepare_server_args` 调用期间，先看 `__post_init__`。
- 如果 `prepare_server_args` 已返回、engine 初始化才失败，再看 `check_server_args()` 和具体 runtime 分支。

## 10. 启动链路和 HTTP Server 怎么分工？

启动链路结束于 `run_server(server_args)` 选择 runtime 分支。普通 `grpc_mode=True` 进入的是 legacy SMG wrapper；默认 HTTP 分支从 `server_args` 接手，才开始分配 `PortArgs`、启动 Scheduler/Detokenizer/TokenizerManager、设置 `_GlobalState`、启动 FastAPI。native Rust gRPC 能力不能从 `grpc_mode` 这条分支推导出来。

读者路径：

```text
启动链路：argv 到 ServerArgs，再到 run_server
HTTP Server：ServerArgs 到 SRT engine 和 FastAPI route
TokenizerManager：GenerateReqInput 到 Scheduler IPC
```

## 复盘

启动问题要先定位阶段：根命令、serve 分发、config 合并、`ServerArgs` post-init、engine 的 `check_server_args`、插件 hook、runtime branch。定位错阶段，搜索再多函数名也会绕远。
