---
title: "启动链路 · 核心概念"
type: concept
framework: sglang
topic: "启动链路"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# 启动链路 · 核心概念

这篇先建立启动链路的心理模型。你应该把它看成一台“命令行控制台”，而不是一个普通 argparse 文件：每一层只解释自己有权解释的参数，解释完再交给下一层。

## 先建立模型

| 控制台层级 | 源码对象 | 它解释什么 | 它不解释什么 |
|------------|----------|------------|--------------|
| 命令入口 | `python/pyproject.toml` | shell 中的 `sglang` 映射到哪个函数 | 不解释服务参数 |
| 根路由 | `cli/main.py` | 子命令是 `serve`、`generate` 还是 `version` | 不解释 `--model-path` |
| serve 分发 | `cli/serve.py` | help、插件、`--model-type`、LLM vs diffusion | 不解释全部 LLM 参数 |
| LLM 参数工厂 | `prepare_server_args` | argv 和 YAML config 如何变成 `ServerArgs` | 不启动进程 |
| runtime 分发 | `run_server` | HTTP/gRPC/Ray/Encoder 哪条入口 | 不解析 CLI |

如果读源码时把这些层混在一起，就会得出错误判断，例如“`cli/main.py` 没有 `--model-path` 所以 CLI 缺参数”或“`--model-type` 应该是 `ServerArgs` 字段”。正确读法是先问：这个参数属于哪一层。

## console script 是真正的第一跳

安装包把 shell 命令 `sglang` 绑定到 `sglang.cli.main:main`。

```toml
# 来源：python/pyproject.toml L178-L180
[project.scripts]
sglang = "sglang.cli.main:main"
killall_sglang = "sglang.cli.killall:main"
```

这解释了为什么读启动链路要从 `cli/main.py` 开始，而不是从 `launch_server.py` 开始。`launch_server.py` 仍保留旧入口，但它不是推荐的第一跳。

## 根命令只做子命令分发

`main()` 只注册 `serve`、`generate`、`version`，然后用 `parse_known_args()` 把未知参数留给子命令。

```python
# 来源：python/sglang/cli/main.py L12-L46
def main():
    parser = argparse.ArgumentParser()

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

    # simple commands
    version_parser = subparsers.add_parser(
        "version",
        help="Show the version information.",
    )
    version_parser.set_defaults(func=version)

    args, extra_argv = parser.parse_known_args()

    if args.subcommand == "serve":
        from sglang.cli.serve import serve

        serve(args, extra_argv)
    elif args.subcommand == "generate":
        from sglang.cli.generate import generate

        generate(args, extra_argv)
    elif args.subcommand == "version":
        version(args, extra_argv)
```

两个设计点要记住：

- `add_help=False` 不是少写 help，而是因为 `serve` 需要按 LLM/diffusion 展示不同参数面。
- `parse_known_args()` 是参数分层的关键，根 parser 不消耗 `--model-path`、`--tp-size`、`--port`。

## `serve()` 是模型族分发器

`serve()` 正常路径先加载插件，再剥离 `--model-type`，再读取 `--model-path` 做模型族判断。LLM 路径才进入 `prepare_server_args`。

```python
# 来源：python/sglang/cli/serve.py L89-L130
    from sglang.srt.plugins import load_plugins

    load_plugins()

    model_type, dispatch_argv = _extract_model_type_override(extra_argv)
    model_path = get_model_path(dispatch_argv)
    try:
        if model_type == "auto":
            is_diffusion_model = get_is_diffusion_model(model_path)
            if is_diffusion_model:
                logger.info("Diffusion model detected")
        else:
            is_diffusion_model = model_type == "diffusion"
            logger.info(
                "Dispatch override enabled: --model-type=%s " "(skip auto detection)",
                model_type,
            )

        if is_diffusion_model:
            # Logic for Diffusion Models
            from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
                add_multimodal_gen_serve_args,
                execute_serve_cmd,
            )

            parser = argparse.ArgumentParser(
                description="SGLang Diffusion Model Serving"
            )
            add_multimodal_gen_serve_args(parser)
            parsed_args, remaining_argv = parser.parse_known_args(dispatch_argv)

            execute_serve_cmd(parsed_args, remaining_argv)
        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            server_args = prepare_server_args(dispatch_argv)

            run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
```

这里的核心不变量是：插件要在 runtime import 前加载；`--model-type` 不进入 LLM parser；LLM 与 diffusion 有两套 parser；退出时要清理子进程树。

## `ServerArgs` 是 LLM 服务配置事实

`ServerArgs` 的字段用 `Annotated` metadata 生成 CLI。`model_path` 支持 `--model` 别名；HTTP 相关字段定义默认 host、port 和 gRPC 开关。

```python
# 来源：python/sglang/srt/server_args.py L419-L425
    model_path: A[
        str,
        Arg(
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            aliases=["--model"],
        ),
    ]
```

```python
# 来源：python/sglang/srt/server_args.py L527-L534
    # -------------------------------------------------------------------------
    # HTTP server
    # -------------------------------------------------------------------------
    host: A[str, "The host of the HTTP server."] = "127.0.0.1"
    port: A[int, "The port of the HTTP server."] = 30000
    fastapi_root_path: A[str, "App is behind a path based routing proxy."] = ""
    grpc_mode: A[bool, "If set, use gRPC server instead of HTTP server."] = False
    skip_server_warmup: A[bool, "If set, skip warmup."] = False
```

`ServerArgs` 不是纯字段表。它在 `__post_init__` 做第二遍语义处理，处理模型无关校验、PD disaggregation、deprecated 参数、dummy model、量化语义等。

```python
# 来源：python/sglang/srt/server_args.py L2567-L2616
    def __post_init__(self):
        """
        Orchestrates the handling of various server arguments, ensuring proper configuration and validation.
        """

        self._maybe_download_model_for_runai()

        # Normalize load balancing defaults early (before dummy-model short-circuit).
        self._handle_load_balance_method()

        # Validate mm_process_config before dummy-model early return.
        self._handle_multimodal()
        # Validate SSL arguments early (before dummy-model short-circuit).
        self._handle_ssl_validation()
        # Validate transcription/ASR-specific server args (model-independent).
        self._handle_asr_validation()

        # Validate PD disaggregation flags early (before dummy-model short-circuit).
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(self)
        if self.enable_session_radix_cache and self.radix_eviction_policy != "priority":
            raise ValueError(
                "--enable-session-radix-cache requires --radix-eviction-policy priority"
            )

        # Normalize deprecated CP aliases before validations or model-specific
        # defaults inspect enable_prefill_cp/cp_strategy.
        self._handle_legacy_cp_arguments()
        self._validate_prefill_only_disable_kv_cache_args()
        self._handle_dcp_validation()

        if self.model_path.lower() in ["none", "dummy"]:
            # Skip for dummy models
            return

        # Handle deprecated arguments.
        self._handle_deprecated_args()

        # Handle deprecated environment variables for prefill delayer.
        self._handle_prefill_delayer_env_compat()

        # Resolve --quantization unquant: explicitly opt out of quantization.
        # Convert to None now (before model config validation), but record
        # the intent so auto-detection in _handle_model_specific_adjustments
        # does not override it.
        if self.quantization == "unquant":
            self.quantization = None
```

因此排查参数时要区分三种状态：CLI 原始字符串、`argparse.Namespace`、经过 `__post_init__` 的 `ServerArgs`。服务真正使用的是第三种。

## 插件是启动前的改线器

插件通过 setuptools entry points 被发现，通过 `SGLANG_PLUGINS` 白名单和 `SGLANG_PLATFORM` 平台过滤控制加载范围。

```python
# 来源：python/sglang/srt/plugins/__init__.py L1-L11
"""
SGLang Unified Plugin Framework.

Supports two types of plugins via setuptools entry_points:
1. Hardware Platform Plugins (sglang.srt.platforms) - register custom hardware platforms
2. General Plugins (sglang.srt.plugins) - inject hooks into functions/methods, replace classes, etc.

Plugins are discovered automatically when installed via pip.
- Platform plugins: use ``SGLANG_PLATFORM`` to select when multiple are installed.
- General plugins: use ``SGLANG_PLUGINS`` (comma-separated) to restrict which are loaded.
"""
```

`HookRegistry` 的约束很明确：注册应发生在 `load_plugins()` 阶段，`apply_hooks()` 应在 engine serving 前调用。

```python
# 来源：python/sglang/srt/plugins/hook_registry.py L60-L76
class HookType(Enum):
    """Types of hooks that can be applied to functions or classes."""

    BEFORE = "before"  # Execute before original; can modify args
    AFTER = "after"  # Execute after original; can modify return value
    AROUND = "around"  # Wrap original; full control over execution
    REPLACE = "replace"  # Replace the original function or class entirely


class HookRegistry:
    """
    Global registry for function/method/class hooks.

    Thread safety: All registration should happen during load_plugins()
    phase (single-threaded). apply_hooks() should be called once before the
    engine starts serving requests.
    """
```

所以插件不是请求时动态扩展系统，而是启动前修改 runtime 行为。读启动链路时要把它看成和 argv 并行的一条控制面。

## `run_server` 是最后分叉

`ServerArgs` 里三个关键字段影响分支：`encoder_only`、`grpc_mode`、`use_ray`。判断顺序是 encoder-only 优先，其次普通 gRPC，再 Ray，最后默认 HTTP。

```python
# 来源：python/sglang/launch_server.py L15-L51
def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        # TODO: Once the native Rust gRPC server starts alongside HTTP in the
        # default path below (controlled by SGLANG_ENABLE_GRPC / SGLANG_GRPC_PORT),
        # remove this legacy SMG path and the grpc_mode flag.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
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
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)
```

这段解释了为什么 `encoder_only=True` 与 `grpc_mode=True` 时不会走普通 `grpc_server.serve_grpc`。它先命中 encoder 分支，再在 encoder 内部选择 gRPC 或 HTTP encoder server。

## 运行验证

启动链路的轻量验证可以只看控制面，不必真的拉起模型。下面的命令覆盖入口注册、CLI 子命令、`ServerArgs` 字段、插件加载和最终 server 分支：

```powershell
rg -n 'sglang =|sglang\.launch_server|def main\(|def serve\(|class ServerArgs|class PortArgs|def prepare_server_args|def load_plugins|HookRegistry|apply_hooks|def run_server|encoder_only|grpc_mode|use_ray' sglang/python/pyproject.toml sglang/python/sglang/cli/main.py sglang/python/sglang/cli/serve.py sglang/python/sglang/srt/server_args.py sglang/python/sglang/srt/plugins/__init__.py sglang/python/sglang/srt/plugins/hook_registry.py sglang/python/sglang/launch_server.py
```

读输出时按这个顺序确认：包入口先到 CLI，`serve` 再构造 `ServerArgs`，插件在 `run_server` 前加载，最后由 `encoder_only`、`grpc_mode`、`use_ray` 决定实际 server 实现。这样可以快速判断一次启动问题是 CLI 层、参数层、插件层还是 server 分支层。

## 常见误解

- `sglang --help` 没有 `--model-path`，不是缺参数，而是根命令只显示子命令。
- `--model-type` 是 serve dispatcher 的 hint，不是 LLM `ServerArgs` 字段。
- `ServerArgs` 字段默认值不是最终真相，`__post_init__` 可能会校验、规范化或改写。
- 插件加载失败不一定阻断启动，但 hook 没 apply 会改变后续 runtime 行为。
- `PortArgs` 不是 CLI 解析结果，它是 runtime 启动时从 `ServerArgs` 派生出的 IPC/NCCL 坐标。

下一篇 [[SGLang-启动链路-源码走读]] 沿一条命令主线把这些概念落到源码。
