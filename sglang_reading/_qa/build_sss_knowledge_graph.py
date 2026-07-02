#!/usr/bin/env python3
"""Build SSS+ knowledge-graph.json and domain-graph.json for SGLang reading project."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SGLANG = Path(__file__).resolve().parent.parent.parent / "sglang"
OUT = SGLANG / ".understand-anything"
GIT = "70df09b83363e0127b43c83a6007d3938f815b2d"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def n(node_id: str, typ: str, name: str, summary: str, tags: list[str], complexity: str = "moderate", file_path: str | None = None):
    d = {"id": node_id, "type": typ, "name": name, "summary": summary, "tags": tags, "complexity": complexity}
    if file_path:
        d["filePath"] = file_path
    return d


def e(src: str, tgt: str, etype: str, weight: float = 0.7):
    return {"source": src, "target": tgt, "type": etype, "direction": "forward", "weight": weight}


# --- Core file nodes (30-batch coverage) ---
FILES = [
    ("document:README.md", "document", "项目 README", "SGLang 官方介绍：RadixAttention、连续批处理、PD 分离、投机解码与硬件支持。", ["overview"], "simple", "README.md"),
    ("config:python/pyproject.toml", "config", "Python 包配置", "setuptools 构建、sglang CLI 入口、依赖与 Rust 扩展声明。", ["config", "packaging"], "simple", "python/pyproject.toml"),
    ("document:python/sglang/README.md", "document", "包内结构说明", "srt/lang/multimodal_gen 等子目录职责索引。", ["docs"], "simple", "python/sglang/README.md"),
    ("file:python/sglang/cli/main.py", "file", "CLI 主入口", "解析 serve/generate/version 子命令并分发。", ["entry", "cli"], "simple", "python/sglang/cli/main.py"),
    ("file:python/sglang/cli/serve.py", "file", "serve 子命令", "检测 LLM/diffusion，调用 prepare_server_args 与 run_server。", ["entry", "cli"], "moderate", "python/sglang/cli/serve.py"),
    ("file:python/sglang/launch_server.py", "file", "服务启动分发", "HTTP/gRPC/Ray/Encoder 四条启动路径路由。", ["entry"], "moderate", "python/sglang/launch_server.py"),
    ("file:python/sglang/srt/entrypoints/http_server.py", "file", "HTTP Server", "FastAPI 路由、挂载 OpenAI/Ollama/Anthropic API，启动 Engine 子进程。", ["entry", "http", "fastapi"], "complex", "python/sglang/srt/entrypoints/http_server.py"),
    ("file:python/sglang/srt/entrypoints/engine.py", "file", "Engine 编排", "初始化 TokenizerManager、Scheduler、Detokenizer 子进程与 ZMQ 通道。", ["runtime", "orchestration"], "complex", "python/sglang/srt/entrypoints/engine.py"),
    ("file:python/sglang/srt/entrypoints/openai/serving_base.py", "file", "OpenAI Serving 基类", "模板方法：校验请求、转换内部结构、流式/非流式响应。", ["openai", "api"], "moderate", "python/sglang/srt/entrypoints/openai/serving_base.py"),
    ("file:python/sglang/srt/entrypoints/openai/serving_completions.py", "file", "Completions API", "OpenAI completion 协议 → GenerateReqInput 转换。", ["openai", "api"], "moderate", "python/sglang/srt/entrypoints/openai/serving_completions.py"),
    ("file:python/sglang/srt/entrypoints/openai/serving_chat.py", "file", "Chat Completions API", "多轮对话、tool/reasoning/multimodal 字段映射。", ["openai", "api"], "complex", "python/sglang/srt/entrypoints/openai/serving_chat.py"),
    ("file:python/sglang/srt/entrypoints/ollama/serving.py", "file", "Ollama Serving", "Ollama JSON 协议适配，构造 GenerateReqInput。", ["ollama", "api"], "moderate", "python/sglang/srt/entrypoints/ollama/serving.py"),
    ("file:python/sglang/srt/entrypoints/grpc_server.py", "file", "gRPC Server (legacy)", "Python asyncio gRPC 服务入口（SMG 路径）。", ["grpc"], "moderate", "python/sglang/srt/entrypoints/grpc_server.py"),
    ("file:python/sglang/srt/entrypoints/grpc_bridge.py", "file", "gRPC Bridge", "Rust Tonic 与 Python RuntimeHandle 跨语言桥接。", ["grpc", "bridge"], "complex", "python/sglang/srt/entrypoints/grpc_bridge.py"),
    ("file:python/sglang/srt/managers/tokenizer_manager.py", "file", "TokenizerManager", "HTTP 侧 tokenize、下发 TokenizedGenerateReqInput 至 Scheduler。", ["scheduling", "ipc"], "complex", "python/sglang/srt/managers/tokenizer_manager.py"),
    ("file:python/sglang/srt/managers/scheduler.py", "file", "Scheduler", "GPU 进程事件循环：收请求、组 batch、驱动 TpWorker、回传输出。", ["scheduling", "core"], "complex", "python/sglang/srt/managers/scheduler.py"),
    ("file:python/sglang/srt/managers/schedule_policy.py", "file", "SchedulePolicy", "PrefillAdder、连续批处理策略与 retract 逻辑。", ["scheduling", "policy"], "complex", "python/sglang/srt/managers/schedule_policy.py"),
    ("file:python/sglang/srt/managers/schedule_batch.py", "file", "ScheduleBatch", "Req/ScheduleBatch/ForwardBatch 数据结构与会话状态。", ["scheduling", "data"], "complex", "python/sglang/srt/managers/schedule_batch.py"),
    ("file:python/sglang/srt/managers/io_struct.py", "file", "IO 消息结构", "跨进程 ZMQ 消息的 dataclass 定义（Generate/Embedding/Abort 等）。", ["scheduling", "ipc"], "moderate", "python/sglang/srt/managers/io_struct.py"),
    ("file:python/sglang/srt/managers/detokenizer_manager.py", "file", "DetokenizerManager", "token id → 文本，增量解码与流式 chunk 组装。", ["output", "ipc"], "moderate", "python/sglang/srt/managers/detokenizer_manager.py"),
    ("file:python/sglang/srt/managers/communicator.py", "file", "Communicator", "FanOut 控制面通信，协调多 worker 状态。", ["output", "ipc"], "moderate", "python/sglang/srt/managers/communicator.py"),
    ("file:python/sglang/srt/model_executor/model_runner.py", "file", "ModelRunner", "模型加载、KV cache 初始化、forward 执行与 CUDA graph。", ["execution", "core"], "complex", "python/sglang/srt/model_executor/model_runner.py"),
    ("file:python/sglang/srt/managers/tp_worker.py", "file", "TpModelWorker", "Tensor Parallel worker，封装 ModelRunner 供 Scheduler 调用。", ["execution"], "complex", "python/sglang/srt/managers/tp_worker.py"),
    ("file:python/sglang/srt/model_executor/forward_batch_info.py", "file", "ForwardBatch", "前向模式 EXTEND/DECODE 与 batch 元信息。", ["execution", "data"], "moderate", "python/sglang/srt/model_executor/forward_batch_info.py"),
    ("file:python/sglang/srt/model_loader/loader.py", "file", "ModelLoader", "权重加载、量化格式与 device 映射。", ["execution", "loading"], "moderate", "python/sglang/srt/model_loader/loader.py"),
    ("file:python/sglang/srt/models/registry.py", "file", "ModelRegistry", "architectures → EntryClass 自动注册与 Transformers 回退。", ["models", "registry"], "moderate", "python/sglang/srt/models/registry.py"),
    ("file:python/sglang/srt/models/llama.py", "file", "Llama 模型", "LlamaAttention/MLP/ForCausalLM 与 RadixAttention 集成。", ["models", "llama"], "moderate", "python/sglang/srt/models/llama.py"),
    ("file:python/sglang/srt/models/qwen3.py", "file", "Qwen3 模型", "QK norm、GQA 与 Qwen3ForCausalLM 实现。", ["models", "qwen"], "moderate", "python/sglang/srt/models/qwen3.py"),
    ("file:python/sglang/srt/models/deepseek_v2.py", "file", "DeepSeek V2/V3", "MLA、MoE 与 DeepSeek 专用层结构。", ["models", "deepseek"], "complex", "python/sglang/srt/models/deepseek_v2.py"),
    ("file:python/sglang/srt/mem_cache/radix_cache.py", "file", "RadixCache", "Radix Tree 前缀 KV 匹配/插入/淘汰。", ["memory", "radix"], "complex", "python/sglang/srt/mem_cache/radix_cache.py"),
    ("file:python/sglang/srt/mem_cache/unified_radix_cache.py", "file", "UnifiedRadixCache", "多组件 UnifiedTreeNode、HiCache 与 session 前缀。", ["memory", "radix"], "complex", "python/sglang/srt/mem_cache/unified_radix_cache.py"),
    ("file:python/sglang/srt/layers/radix_attention.py", "file", "RadixAttention", "Attention 层对接 cache 与 flashinfer/triton backend。", ["memory", "attention"], "complex", "python/sglang/srt/layers/radix_attention.py"),
    ("file:python/sglang/srt/layers/attention/flashattention_backend.py", "file", "FlashAttention Backend", "FlashInfer/FlashAttention 内核调度。", ["attention", "kernel"], "complex", "python/sglang/srt/layers/attention/flashattention_backend.py"),
    ("file:python/sglang/srt/layers/moe/fused_moe_triton.py", "file", "Fused MoE Triton", "MoE 路由与 fused expert 计算。", ["moe", "kernel"], "complex", "python/sglang/srt/layers/moe/fused_moe_triton.py"),
    ("file:python/sglang/srt/layers/quantization/fp8.py", "file", "FP8 量化", "FP8 权重量化与 GEMM 路径。", ["quantization"], "moderate", "python/sglang/srt/layers/quantization/fp8.py"),
    ("file:python/sglang/srt/sampling/sampling_params.py", "file", "SamplingParams", "temperature/top_p/penalty 等采样参数。", ["sampling"], "moderate", "python/sglang/srt/sampling/sampling_params.py"),
    ("file:python/sglang/srt/speculative/eagle_worker_v2.py", "file", "EAGLE Worker", "投机解码 draft/verify 双模型 worker。", ["speculative"], "complex", "python/sglang/srt/speculative/eagle_worker_v2.py"),
    ("file:python/sglang/srt/disaggregation/prefill.py", "file", "PD Prefill 节点", "Prefill 侧 disaggregation 服务与 KV 传输。", ["disaggregation"], "complex", "python/sglang/srt/disaggregation/prefill.py"),
    ("file:python/sglang/srt/disaggregation/decode.py", "file", "PD Decode 节点", "Decode 侧接收 KV 并继续生成。", ["disaggregation"], "complex", "python/sglang/srt/disaggregation/decode.py"),
    ("file:python/sglang/srt/distributed/parallel_state.py", "file", "Parallel State", "TP/PP/EP/DP GroupCoordinator 与通信组。", ["distributed"], "complex", "python/sglang/srt/distributed/parallel_state.py"),
    ("file:python/sglang/srt/managers/data_parallel_controller.py", "file", "DataParallelController", "多 replica 路由与负载均衡。", ["distributed"], "moderate", "python/sglang/srt/managers/data_parallel_controller.py"),
    ("file:python/sglang/srt/managers/multimodal_processor.py", "file", "MultimodalProcessor", "VLM 图像/视频预处理与 embedding 注入。", ["multimodal"], "moderate", "python/sglang/srt/managers/multimodal_processor.py"),
    ("file:python/sglang/srt/lora/lora_manager.py", "file", "LoRAManager", "多 LoRA adapter 加载与 batch 内切换。", ["lora"], "moderate", "python/sglang/srt/lora/lora_manager.py"),
    ("file:python/sglang/lang/api.py", "file", "Frontend API", "gen/user/Engine 等结构化生成 DSL 入口。", ["frontend", "lang"], "moderate", "python/sglang/lang/api.py"),
    ("file:python/sglang/multimodal_gen/runtime/launch_server.py", "file", "Diffusion Launch", "multimodal_gen 扩散模型服务启动与 worker 管理。", ["multimodal_gen"], "moderate", "python/sglang/multimodal_gen/runtime/launch_server.py"),
    ("file:sgl-kernel/python/sgl_kernel/__init__.py", "file", "sgl-kernel 入口", "按 GPU 架构加载 common_ops CUDA 扩展并 re-export 算子。", ["kernel", "cuda"], "moderate", "sgl-kernel/python/sgl_kernel/__init__.py"),
    ("file:sgl-kernel/python/sgl_kernel/load_utils.py", "file", "Kernel 加载器", "sm90/sm100 架构检测与 .so 动态 import。", ["kernel", "cuda"], "moderate", "sgl-kernel/python/sgl_kernel/load_utils.py"),
    ("file:sgl-model-gateway/src/server.rs", "file", "Gateway Server", "Rust 模型网关 HTTP/gRPC 路由与健康检查。", ["gateway", "rust"], "complex", "sgl-model-gateway/src/server.rs"),
    ("file:python/sglang/__init__.py", "file", "公共 API", "Frontend gen API 与 LazyImport Runtime Engine。", ["api"], "moderate", "python/sglang/__init__.py"),
    ("file:python/sglang/global_config.py", "file", "GlobalConfig", "全局常量与运行时配置。", ["config"], "simple", "python/sglang/global_config.py"),
    ("file:python/sglang/srt/server_args.py", "file", "ServerArgs", "CLI 参数解析为统一 ServerArgs 对象。", ["config", "entry"], "moderate", "python/sglang/srt/server_args.py"),
]

MODULES = [
    ("module:srt", "module", "SGLang Runtime", "python/sglang/srt 推理运行时核心。", ["runtime"]),
    ("module:lang", "module", "Frontend Language", "结构化生成 DSL。", ["frontend"]),
    ("module:sgl-kernel", "module", "sgl-kernel", "CUDA/Triton 高性能算子库。", ["kernel"]),
    ("module:sgl-model-gateway", "module", "sgl-model-gateway", "Rust 模型路由网关。", ["gateway"]),
    ("module:multimodal_gen", "module", "multimodal_gen", "扩散/视频生成 runtime。", ["multimodal_gen"]),
    ("module:openai-api", "module", "OpenAI API 层", "OpenAI 兼容协议适配。", ["openai"]),
]

CONCEPTS = [
    ("concept:radix-attention", "concept", "RadixAttention", "基于 Radix Tree 的前缀 KV 共享，SGLang 核心卖点。", ["cache", "performance"]),
    ("concept:continuous-batching", "concept", "Continuous Batching", "Scheduler 动态合并 prefill/decode 请求。", ["scheduling"]),
    ("concept:pd-disaggregation", "concept", "PD Disaggregation", "Prefill 与 Decode 节点分离部署。", ["disaggregation"]),
    ("concept:speculative-decoding", "concept", "Speculative Decoding", "Draft 模型投机验证加速 decode。", ["speculative"]),
    ("concept:prefix-cache", "concept", "Prefix Cache", "跨请求共享相同 prompt 前缀的 KV。", ["cache"]),
    ("concept:forward-mode", "concept", "Forward Mode", "EXTEND prefill 与 DECODE 两种前向模式。", ["execution"]),
    ("concept:grpc-py-bridge", "concept", "gRPC Python Bridge", "Rust gRPC 服务与 Python runtime 的 mpsc 桥。", ["grpc"]),
]

CLASSES = [
    ("class:ModelRunner", "class", "ModelRunner", "model_runner.py 中模型执行主类。", ["execution"], "complex"),
    ("class:TpModelWorker", "class", "TpModelWorker", "tp_worker.py 中 TP worker 封装。", ["execution"], "complex"),
    ("class:RadixCache", "class", "RadixCache", "radix_cache.py 前缀树缓存实现。", ["memory"], "complex"),
    ("class:LoRAManager", "class", "LoRAManager", "lora_manager.py 多 adapter 管理。", ["lora"], "moderate"),
    ("class:DetokenizerManager", "class", "DetokenizerManager", "detokenizer_manager.py 反 tokenize。", ["output"], "moderate"),
]

nodes = [n(*args) for args in FILES] + [n(*args) for args in MODULES] + [n(*args) for args in CONCEPTS] + [n(*args) for args in CLASSES]

edges = [
    e("file:python/sglang/cli/main.py", "file:python/sglang/cli/serve.py", "calls", 0.8),
    e("file:python/sglang/cli/serve.py", "file:python/sglang/launch_server.py", "calls", 0.8),
    e("file:python/sglang/launch_server.py", "file:python/sglang/srt/entrypoints/http_server.py", "calls", 0.8),
    e("file:python/sglang/srt/entrypoints/http_server.py", "file:python/sglang/srt/entrypoints/engine.py", "calls", 0.9),
    e("file:python/sglang/srt/entrypoints/http_server.py", "file:python/sglang/srt/entrypoints/openai/serving_completions.py", "contains", 1.0),
    e("file:python/sglang/srt/entrypoints/openai/serving_completions.py", "file:python/sglang/srt/entrypoints/openai/serving_base.py", "inherits", 0.9),
    e("file:python/sglang/srt/entrypoints/openai/serving_completions.py", "file:python/sglang/srt/managers/tokenizer_manager.py", "calls", 0.8),
    e("file:python/sglang/srt/entrypoints/ollama/serving.py", "file:python/sglang/srt/managers/tokenizer_manager.py", "calls", 0.8),
    e("file:python/sglang/srt/entrypoints/engine.py", "file:python/sglang/srt/managers/tokenizer_manager.py", "calls", 0.9),
    e("file:python/sglang/srt/entrypoints/engine.py", "file:python/sglang/srt/managers/scheduler.py", "calls", 0.9),
    e("file:python/sglang/srt/entrypoints/engine.py", "file:python/sglang/srt/managers/detokenizer_manager.py", "calls", 0.9),
    e("file:python/sglang/srt/managers/tokenizer_manager.py", "file:python/sglang/srt/managers/scheduler.py", "calls", 0.8),
    e("file:python/sglang/srt/managers/scheduler.py", "file:python/sglang/srt/managers/tp_worker.py", "calls", 0.9),
    e("file:python/sglang/srt/managers/scheduler.py", "file:python/sglang/srt/managers/schedule_policy.py", "calls", 0.8),
    e("file:python/sglang/srt/managers/scheduler.py", "file:python/sglang/srt/managers/schedule_batch.py", "depends_on", 0.8),
    e("file:python/sglang/srt/managers/scheduler.py", "file:python/sglang/srt/mem_cache/radix_cache.py", "calls", 0.8),
    e("file:python/sglang/srt/managers/tp_worker.py", "file:python/sglang/srt/model_executor/model_runner.py", "calls", 0.9),
    e("file:python/sglang/srt/model_executor/model_runner.py", "file:python/sglang/srt/models/registry.py", "depends_on", 0.7),
    e("file:python/sglang/srt/models/registry.py", "file:python/sglang/srt/models/llama.py", "contains", 1.0),
    e("file:python/sglang/srt/models/llama.py", "file:python/sglang/srt/layers/radix_attention.py", "calls", 0.8),
    e("file:python/sglang/srt/layers/radix_attention.py", "file:python/sglang/srt/layers/attention/flashattention_backend.py", "calls", 0.8),
    e("file:python/sglang/srt/mem_cache/radix_cache.py", "concept:prefix-cache", "related", 0.7),
    e("file:python/sglang/srt/layers/radix_attention.py", "concept:radix-attention", "related", 0.9),
    e("file:python/sglang/srt/managers/scheduler.py", "concept:continuous-batching", "related", 0.9),
    e("file:python/sglang/srt/disaggregation/prefill.py", "concept:pd-disaggregation", "related", 0.9),
    e("file:python/sglang/srt/speculative/eagle_worker_v2.py", "concept:speculative-decoding", "related", 0.9),
    e("module:srt", "file:python/sglang/srt/managers/scheduler.py", "contains", 1.0),
    e("module:sgl-kernel", "file:sgl-kernel/python/sgl_kernel/__init__.py", "contains", 1.0),
    e("module:sgl-model-gateway", "file:sgl-model-gateway/src/server.rs", "contains", 1.0),
    e("module:lang", "file:python/sglang/lang/api.py", "contains", 1.0),
    e("module:multimodal_gen", "file:python/sglang/multimodal_gen/runtime/launch_server.py", "contains", 1.0),
    e("class:ModelRunner", "file:python/sglang/srt/model_executor/model_runner.py", "related", 1.0),
    e("class:RadixCache", "file:python/sglang/srt/mem_cache/radix_cache.py", "related", 1.0),
]

file_ids = [x[0] for x in FILES if x[1] == "file"] + [x[0] for x in FILES if x[1] in ("config", "document")]

layers = [
    {"id": "layer:documentation", "name": "文档与配置层", "description": "README、pyproject 与版本信息。", "nodeIds": ["document:README.md", "config:python/pyproject.toml", "document:python/sglang/README.md", "file:python/sglang/global_config.py"]},
    {"id": "layer:entrypoint", "name": "入口层", "description": "CLI、launch_server、HTTP/gRPC 入口。", "nodeIds": ["file:python/sglang/cli/main.py", "file:python/sglang/cli/serve.py", "file:python/sglang/launch_server.py", "file:python/sglang/srt/server_args.py", "file:python/sglang/srt/entrypoints/http_server.py", "file:python/sglang/srt/entrypoints/grpc_server.py", "file:python/sglang/srt/entrypoints/grpc_bridge.py"]},
    {"id": "layer:api-adapters", "name": "协议适配层", "description": "OpenAI/Ollama/Anthropic 兼容 API。", "nodeIds": ["file:python/sglang/srt/entrypoints/openai/serving_base.py", "file:python/sglang/srt/entrypoints/openai/serving_completions.py", "file:python/sglang/srt/entrypoints/openai/serving_chat.py", "file:python/sglang/srt/entrypoints/ollama/serving.py", "module:openai-api"]},
    {"id": "layer:orchestration", "name": "引擎编排层", "description": "Engine 启动子进程与 IPC。", "nodeIds": ["file:python/sglang/srt/entrypoints/engine.py", "file:python/sglang/srt/managers/io_struct.py", "file:python/sglang/srt/managers/communicator.py"]},
    {"id": "layer:scheduling", "name": "请求调度层", "description": "Tokenizer → Scheduler → Detokenizer。", "nodeIds": ["file:python/sglang/srt/managers/tokenizer_manager.py", "file:python/sglang/srt/managers/scheduler.py", "file:python/sglang/srt/managers/schedule_policy.py", "file:python/sglang/srt/managers/schedule_batch.py", "file:python/sglang/srt/managers/detokenizer_manager.py", "concept:continuous-batching"]},
    {"id": "layer:model-execution", "name": "模型执行层", "description": "ModelRunner、TpWorker、ModelLoader、Registry。", "nodeIds": ["file:python/sglang/srt/model_executor/model_runner.py", "file:python/sglang/srt/managers/tp_worker.py", "file:python/sglang/srt/model_executor/forward_batch_info.py", "file:python/sglang/srt/model_loader/loader.py", "file:python/sglang/srt/models/registry.py", "file:python/sglang/srt/models/llama.py", "file:python/sglang/srt/models/qwen3.py", "file:python/sglang/srt/models/deepseek_v2.py", "class:ModelRunner", "class:TpModelWorker", "concept:forward-mode"]},
    {"id": "layer:memory-attention", "name": "内存与 Attention 层", "description": "RadixCache、Attention backend、MoE、量化。", "nodeIds": ["file:python/sglang/srt/mem_cache/radix_cache.py", "file:python/sglang/srt/mem_cache/unified_radix_cache.py", "file:python/sglang/srt/layers/radix_attention.py", "file:python/sglang/srt/layers/attention/flashattention_backend.py", "file:python/sglang/srt/layers/moe/fused_moe_triton.py", "file:python/sglang/srt/layers/quantization/fp8.py", "class:RadixCache", "concept:radix-attention", "concept:prefix-cache"]},
    {"id": "layer:advanced", "name": "高级特性层", "description": "Sampling、投机、PD 分离、分布式。", "nodeIds": ["file:python/sglang/srt/sampling/sampling_params.py", "file:python/sglang/srt/speculative/eagle_worker_v2.py", "file:python/sglang/srt/disaggregation/prefill.py", "file:python/sglang/srt/disaggregation/decode.py", "file:python/sglang/srt/distributed/parallel_state.py", "file:python/sglang/srt/managers/data_parallel_controller.py", "concept:pd-disaggregation", "concept:speculative-decoding"]},
    {"id": "layer:extensions", "name": "扩展组件层", "description": "Multimodal、LoRA、kernel、gateway、lang、diffusion。", "nodeIds": ["file:python/sglang/srt/managers/multimodal_processor.py", "file:python/sglang/srt/lora/lora_manager.py", "file:sgl-kernel/python/sgl_kernel/__init__.py", "file:sgl-kernel/python/sgl_kernel/load_utils.py", "file:sgl-model-gateway/src/server.rs", "file:python/sglang/lang/api.py", "file:python/sglang/multimodal_gen/runtime/launch_server.py", "module:sgl-kernel", "module:sgl-model-gateway", "module:lang", "module:multimodal_gen", "class:LoRAManager"]},
    {"id": "layer:public-api", "name": "公共 API 层", "description": "Python 包对外 export。", "nodeIds": ["file:python/sglang/__init__.py", "module:srt"]},
]

tour = [
    {"order": 1, "title": "项目总览", "description": "从 README 理解 SGLang 定位与核心卖点。", "nodeIds": ["document:README.md", "concept:radix-attention", "concept:continuous-batching"]},
    {"order": 2, "title": "Monorepo 结构", "description": "srt / lang / sgl-kernel / gateway / multimodal_gen 分工。", "nodeIds": ["document:python/sglang/README.md", "module:srt", "module:lang", "module:sgl-kernel"]},
    {"order": 3, "title": "CLI 入口", "description": "sglang serve 命令如何到达 run_server。", "nodeIds": ["file:python/sglang/cli/main.py", "file:python/sglang/cli/serve.py"]},
    {"order": 4, "title": "HTTP 服务启动", "description": "launch_server → http_server → Engine 子进程。", "nodeIds": ["file:python/sglang/launch_server.py", "file:python/sglang/srt/entrypoints/http_server.py", "file:python/sglang/srt/entrypoints/engine.py"]},
    {"order": 5, "title": "OpenAI API", "description": "Serving 模板方法如何将 HTTP 转为内部 Generate 请求。", "nodeIds": ["file:python/sglang/srt/entrypoints/openai/serving_base.py", "file:python/sglang/srt/entrypoints/openai/serving_completions.py"]},
    {"order": 6, "title": "TokenizerManager", "description": "tokenize 与 ZMQ 下发至 Scheduler。", "nodeIds": ["file:python/sglang/srt/managers/tokenizer_manager.py", "file:python/sglang/srt/managers/io_struct.py"]},
    {"order": 7, "title": "Scheduler 核心", "description": "事件循环、continuous batching、run_batch。", "nodeIds": ["file:python/sglang/srt/managers/scheduler.py", "file:python/sglang/srt/managers/schedule_policy.py", "concept:continuous-batching"]},
    {"order": 8, "title": "ModelRunner 前向", "description": "TpWorker 驱动 ModelRunner forward。", "nodeIds": ["file:python/sglang/srt/managers/tp_worker.py", "file:python/sglang/srt/model_executor/model_runner.py"]},
    {"order": 9, "title": "RadixAttention", "description": "前缀 KV 共享与 RadixCache 协作。", "nodeIds": ["file:python/sglang/srt/mem_cache/radix_cache.py", "file:python/sglang/srt/layers/radix_attention.py", "concept:prefix-cache"]},
    {"order": 10, "title": "Detokenizer 输出", "description": "token → 文本流式返回客户端。", "nodeIds": ["file:python/sglang/srt/managers/detokenizer_manager.py"]},
    {"order": 11, "title": "投机解码", "description": "EAGLE draft/verify 加速。", "nodeIds": ["file:python/sglang/srt/speculative/eagle_worker_v2.py", "concept:speculative-decoding"]},
    {"order": 12, "title": "PD 分离", "description": "Prefill/Decode 节点拆分部署。", "nodeIds": ["file:python/sglang/srt/disaggregation/prefill.py", "file:python/sglang/srt/disaggregation/decode.py", "concept:pd-disaggregation"]},
    {"order": 13, "title": "分布式并行", "description": "TP/PP/EP/DP 通信组。", "nodeIds": ["file:python/sglang/srt/distributed/parallel_state.py"]},
    {"order": 14, "title": "sgl-kernel", "description": "CUDA 算子按架构加载。", "nodeIds": ["file:sgl-kernel/python/sgl_kernel/load_utils.py", "module:sgl-kernel"]},
    {"order": 15, "title": "model-gateway", "description": "Rust 网关路由 prefill/decode worker。", "nodeIds": ["file:sgl-model-gateway/src/server.rs", "module:sgl-model-gateway"]},
    {"order": 16, "title": "Frontend lang", "description": "结构化生成 DSL。", "nodeIds": ["file:python/sglang/lang/api.py", "module:lang"]},
    {"order": 17, "title": "扩散 runtime", "description": "multimodal_gen 独立服务。", "nodeIds": ["file:python/sglang/multimodal_gen/runtime/launch_server.py", "module:multimodal_gen"]},
]

# Validate layer assignment
node_ids = {x["id"] for x in nodes}
for layer in layers:
    layer["nodeIds"] = [i for i in layer["nodeIds"] if i in node_ids]

kg = {
    "version": "1.0.0",
    "project": {
        "name": "sglang",
        "languages": ["python", "rust", "cpp", "cuda"],
        "frameworks": ["FastAPI", "PyTorch", "Docker"],
        "description": "面向大语言模型与多模态模型的高性能推理服务框架，核心运行时位于 python/sglang/srt（SGLang Runtime）。",
        "analyzedAt": NOW,
        "gitCommitHash": GIT,
        "scope": "batch-30-final-sss",
        "readingBatches": [f"{i:02d}" for i in range(1, 31)],
    },
    "nodes": nodes,
    "edges": edges,
    "layers": layers,
    "tour": tour,
}

# Domain graph
domain_nodes = [
    {"id": "domain:llm-serving", "type": "domain", "name": "LLM 推理服务", "summary": "从 CLI/HTTP 到 token 输出的端到端服务域。", "tags": ["domain"]},
    {"id": "flow:generate-lifecycle", "type": "flow", "name": "Generate 请求生命周期", "summary": "用户 prompt → tokenize → schedule → forward → detokenize → SSE。", "tags": ["flow"]},
    {"id": "step:cli-entry", "type": "step", "name": "CLI 入口", "summary": "sglang serve 解析参数并启动服务。", "tags": ["step"]},
    {"id": "step:http-route", "type": "step", "name": "HTTP 路由", "summary": "FastAPI 接收 OpenAI 兼容请求。", "tags": ["step"]},
    {"id": "step:tokenize", "type": "step", "name": "Tokenize", "summary": "TokenizerManager 将文本转为 token 并 IPC 发送。", "tags": ["step"]},
    {"id": "step:schedule", "type": "step", "name": "Schedule Batch", "summary": "Scheduler 组 batch 并触发 ModelRunner。", "tags": ["step"]},
    {"id": "step:forward", "type": "step", "name": "Model Forward", "summary": "ModelRunner 执行 attention/MLP 产生 logits。", "tags": ["step"]},
    {"id": "step:sample", "type": "step", "name": "Sample Token", "summary": "采样下一个 token id。", "tags": ["step"]},
    {"id": "step:detokenize", "type": "step", "name": "Detokenize", "summary": "DetokenizerManager 组装流式文本 chunk。", "tags": ["step"]},
    {"id": "domain:kv-cache", "type": "domain", "name": "KV 缓存域", "summary": "RadixCache 前缀共享与 paged 分配。", "tags": ["domain"]},
    {"id": "flow:prefix-match", "type": "flow", "name": "前缀匹配流程", "summary": "match_prefix → 跳过已缓存 prefill → insert 完成请求。", "tags": ["flow"]},
]

domain_edges = [
    e("domain:llm-serving", "flow:generate-lifecycle", "contains_flow", 1.0),
    e("flow:generate-lifecycle", "step:cli-entry", "flow_step", 1.0),
    e("flow:generate-lifecycle", "step:http-route", "flow_step", 1.0),
    e("flow:generate-lifecycle", "step:tokenize", "flow_step", 1.0),
    e("flow:generate-lifecycle", "step:schedule", "flow_step", 1.0),
    e("flow:generate-lifecycle", "step:forward", "flow_step", 1.0),
    e("flow:generate-lifecycle", "step:sample", "flow_step", 1.0),
    e("flow:generate-lifecycle", "step:detokenize", "flow_step", 1.0),
    e("step:cli-entry", "file:python/sglang/cli/serve.py", "related", 0.8),
    e("step:http-route", "file:python/sglang/srt/entrypoints/http_server.py", "related", 0.8),
    e("step:tokenize", "file:python/sglang/srt/managers/tokenizer_manager.py", "related", 0.8),
    e("step:schedule", "file:python/sglang/srt/managers/scheduler.py", "related", 0.9),
    e("step:forward", "file:python/sglang/srt/model_executor/model_runner.py", "related", 0.9),
    e("step:detokenize", "file:python/sglang/srt/managers/detokenizer_manager.py", "related", 0.8),
    e("domain:kv-cache", "flow:prefix-match", "contains_flow", 1.0),
    e("flow:prefix-match", "file:python/sglang/srt/mem_cache/radix_cache.py", "related", 0.9),
]

dg = {"version": "1.0.0", "project": kg["project"], "nodes": domain_nodes, "edges": domain_edges}

meta = {
    "lastAnalyzedAt": NOW,
    "gitCommitHash": GIT,
    "version": "1.0.0",
    "analyzedFiles": len([x for x in FILES if x[1] == "file"]),
    "scope": "batch-30-final-sss",
    "lastBatch": 30,
    "note": "SSS+ 知识库重建：覆盖 30 批关键 file/module/concept 节点 + domain-graph",
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "knowledge-graph.json").write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "domain-graph.json").write_text(json.dumps(dg, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"KG: {len(nodes)} nodes, {len(edges)} edges, {len(layers)} layers, {len(tour)} tour steps")
print(f"DG: {len(domain_nodes)} nodes, {len(domain_edges)} edges")
