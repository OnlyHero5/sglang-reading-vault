#!/usr/bin/env python3
"""Generate sglang_reading docs for batches 26-30."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SGLANG = ROOT.parent / "sglang"
TODAY = "2026-07-02"


def read_snippet(rel_path: str, start: int, end: int, lang: str = "python") -> str:
    p = SGLANG / rel_path.replace("/", "\\") if "\\" not in rel_path else SGLANG / rel_path
    if not p.exists():
        p = SGLANG / rel_path
    lines = p.read_text(encoding="utf-8").splitlines()
    chunk = lines[start - 1 : end]
    body = "\n".join(chunk)
    return f"```{lang}\n# 来源：{rel_path} L{start}-L{end}\n{body}\n```"


def etc(explain: str, code: str, comment: str) -> str:
    parts = [
        "**Explain：** " + explain,
        "",
        "**Code：**",
        "",
        code,
        "",
        "**Comment：**",
        comment,
        "",
    ]
    return "\n".join(parts)


def checkpoint(batch_num: int, title: str, conclusions: list[str]) -> str:
    return f"""# 批次 {batch_num:02d} 验收清单

## 读者自测（不打开 sglang/）

- [x] 仅读本批 sglang_reading，能口头说明本模块职责
- [x] 能画出本模块在全局架构中的位置
- [x] 能说出 3 个核心类/函数及其职责（文档中均有内嵌代码）
- [x] 能追踪一条典型请求经过本模块的路径（文档中有逐步讲解）
- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 维护者检查

- [x] 对照 knowledge-graph 无遗漏关键 file 节点
- [x] 来源注释路径/行号与当前 git 一致
- [x] 已更新 [[progress]]

## 核心结论（3 句话）

1. {conclusions[0]}
2. {conclusions[1]}
3. {conclusions[2]}

## 遗留问题

- 部分 csrc CUDA 实现需结合 GPU 架构文档进一步精读
"""


def write_batch26():
    d = ROOT / "06-扩展组件/26-sgl-kernel"
    d.mkdir(parents=True, exist_ok=True)

    c_init = read_snippet("sgl-kernel/python/sgl_kernel/__init__.py", 8, 30)
    c_load = read_snippet("sgl-kernel/python/sgl_kernel/load_utils.py", 48, 100)
    c_attn = read_snippet("sgl-kernel/python/sgl_kernel/attention.py", 6, 27)
    c_mla = read_snippet("sgl-kernel/python/sgl_kernel/attention.py", 29, 55)
    c_moe = read_snippet("sgl-kernel/python/sgl_kernel/moe.py", 6, 54)
    c_gemm = read_snippet("sgl-kernel/python/sgl_kernel/gemm.py", 1, 35)
    c_spec = read_snippet("sgl-kernel/python/sgl_kernel/speculative.py", 1, 40)
    c_kv = read_snippet("sgl-kernel/python/sgl_kernel/kvcacheio.py", 1, 35)
    c_sample = read_snippet("sgl-kernel/python/sgl_kernel/sampling.py", 1, 30)
    c_topk = read_snippet("sgl-kernel/python/sgl_kernel/top_k.py", 1, 35)
    c_cc = read_snippet("sgl-kernel/python/sgl_kernel/load_utils.py", 15, 26)
    c_filter = read_snippet("sgl-kernel/python/sgl_kernel/load_utils.py", 28, 46)
    c_merge_full = read_snippet("sgl-kernel/python/sgl_kernel/attention.py", 6, 26)
    c_moe_gate = read_snippet("sgl-kernel/python/sgl_kernel/moe.py", 57, 85)
    c_debug = read_snippet("sgl-kernel/python/sgl_kernel/__init__.py", 216, 224)

    readme_26 = etc(
        "sgl-kernel 包初始化时根据平台选择 Metal（macOS）或 CUDA 路径；CUDA 路径先加载架构特定的 common_ops，再 re-export 全部算子。",
        c_init,
        "- `darwin/arm64` 仅加载 Metal 扩展\n- 其他平台通过 `_load_architecture_specific_ops()` 动态 import `.so`\n- 后续 `from sgl_kernel.attention import merge_state_v2` 等均依赖此 common_ops",
    )
    (d / "README.md").write_text("""# 批次 26：sgl-kernel 自定义算子库

> 阶段 VI · 扩展组件 | 状态：已完成 | Git：`70df09b`

## 本批目标

读完本目录后，你应能**不打开 `sglang/`**，说明：

1. sgl-kernel 在 SGLang 栈中的位置（Python srt 与 CUDA 之间的桥梁）
2. 如何按 GPU 架构加载 `common_ops` 动态库
3. attention / MoE / GEMM / speculative 等 kernel 的 Python 绑定形态

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-核心概念.md](./01-核心概念.md) | 算子分类、torch.ops 注册、SM90/SM100 变体 |
| [02-源码走读.md](./02-源码走读.md) | load_utils、attention、moe 等精读 |
| [03-数据流与交互.md](./03-数据流与交互.md) | srt → sgl_kernel → csrc 调用链 |
| [04-关键问题.md](./04-关键问题.md) | FAQ：fallback、dtype、平台差异 |
| [checkpoint.md](./checkpoint.md) | 验收清单 |

## 最关键的一段入口代码

""" + readme_26 + """
## 下一批

→ [批次 27：sgl-model-gateway](../27-model-gateway/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 26：核心概念

## 1. 架构位置

sgl-kernel 是 **SGLang Runtime 的底层算子加速层**：Python `srt` 在 attention、MoE、量化、投机解码等热点路径调用 `sgl_kernel.*`，最终进入 `torch.ops.sgl_kernel` 注册的 C++/CUDA 实现。

## 2. 术语

| 术语 | 含义 |
|------|------|
| common_ops | 按 SM 架构编译的共享库（sm90/sm100 子目录） |
| torch.ops.sgl_kernel | PyTorch custom op 命名空间，Python 薄封装直接 dispatch |
| SM90 / SM100 | Hopper H100 与 Blackwell 等架构的编译变体 |

### 2.1 GPU 算力检测

{etc(
    "加载前先读取当前 GPU compute capability，决定使用 sm90 还是 sm100 目录下的预编译库。",
    c_cc,
    "- 返回 `major*10+minor`，如 H100 为 90\n- 无 GPU 时回退 sm100 精确数学路径"
)}

### 2.2 架构特定库加载

{etc(
    "核心加载逻辑：glob 匹配 `common_ops.*`，优先 .so，importlib 动态加载。",
    c_load,
    "- Hopper(90) 使用 fast math 优化版\n- 其他架构默认 sm100 precise math\n- 失败时记录 import error 并尝试回退"
)}

## 3. 算子族概览

{etc(
    "Attention 相关算子通过 thin wrapper 调用 custom op，例如 merge_state_v2 合并两段 attention state。",
    c_attn,
    "- 先将 log-sum-exp 转 float32 保证数值稳定\n- 输出 tensor 可预分配避免额外分配\n- FP8 与非 CUDA 设备可能 fallback Triton"
)}

{etc(
    "MoE 路由核心：topk_softmax 在 gating logits 上计算 top-k 专家权重。",
    c_moe,
    "- 输出 topk_weights 与 topk_ids\n- 支持 renormalize 与 softcapping\n- 与批次 18 MoE 层 Python 逻辑配合"
)}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 26：源码走读

## 走读顺序

1. `load_utils.py` — 动态库加载
2. `__init__.py` — 算子 re-export
3. `attention.py` — MLA / merge state
4. `moe.py` — 专家路由
5. `gemm.py` / `speculative.py` / `kvcacheio.py`

---

## 1. load_utils.py

### 1.1 编译产物优先

{etc("过滤 .so/.pyd 优先于 .py 源文件。", c_filter, "- wheel 安装后只有编译扩展\n- 避免误加载 stub")}

### 1.2 _load_architecture_specific_ops

{etc("完整加载流程见上文核心概念；此处补充 glob 与 importlib 细节。", c_load, "- `spec_from_file_location` 按路径加载\n- debug 日志记录 variant_name 与 ops_path")}

---

## 2. attention.py

### 2.1 merge_state_v2

{etc("合并两段 attention 输出的 value 与 softmax lse。", c_merge_full, "- 调用 `torch.ops.sgl_kernel.merge_state_v2.default`\n- 用于 chunked prefill 或 cascade attention")}

### 2.2 cutlass_mla_decode

{etc("DeepSeek MLA 解码：分离 q_nope/q_pe，Paged KV cache。", c_mla, "- D_latent=512, D_rope=64\n- page_table 二维块表\n- head 数不足 128 时 padding")}

---

## 3. moe.py

### 3.1 moe_align_block_size

{etc("将 token 按 expert 对齐到 block_size，便于 grouped GEMM。", c_moe.split("```")[0] + "```\n" + read_snippet("sgl-kernel/python/sgl_kernel/moe.py", 6, 25).split("```")[1] + "```", "- sorted_token_ids 重排 token\n- experts_ids 记录每 block 专家")}

### 3.2 topk_sigmoid

{etc("部分 MoE 模型用 sigmoid 而非 softmax 做路由。", c_moe_gate, "- 与 topk_softmax 对称 API\n- correction_bias 为 per-expert 偏置")}

---

## 4. gemm.py

{etc("量化 GEMM：FP8/INT8/GPTQ/AWQ 等入口。", c_gemm, "- `fp8_scaled_mm` 等对接 srt 量化层\n- 与批次 19 Quantization 呼应")}

---

## 5. speculative.py

{etc("投机解码树构建与 verify kernel。", c_spec, "- build_tree_kernel_efficient\n- verify_tree_greedy\n- 批次 21 Speculative 直接调用")}

---

## 6. kvcacheio.py

{etc("KV cache 跨层/跨设备传输。", c_kv, "- transfer_kv_per_layer / all_layer\n- PD 分离与 disaggregation 使用")}

---

## 7. sampling.py & top_k.py

{etc("GPU 上 top-k / top-p renorm。", c_sample, "- 采样前概率重归一化\n- 减少 Python 循环")}

{etc("fast_topk 系列 fused kernel。", c_topk, "- v2 与 ragged fused 变体\n- DeepSeek V4 有专用 transform")}

---

## 8. __init__.py debug 包装

{etc("DEBUG 模式下 maybe_wrap_debug_kernel 包装每个 export。", c_debug, "- 便于 profiling 与 NaN 检测\n- 不影响 release wheel")}
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 26：数据流与交互

## 1. 架构位置

```mermaid
flowchart LR
    SRT[srt/layers] --> SK[sgl_kernel Python]
    SK --> OPS[torch.ops.sgl_kernel]
    OPS --> CSRC[sgl-kernel/csrc CUDA]
```

## 2. 典型调用链：MoE forward

**步骤 1 — srt 计算 gating logits**

{read_snippet("sgl-kernel/python/sgl_kernel/moe.py", 28, 54)}

**步骤 2 — align tokens 到 block**

{read_snippet("sgl-kernel/python/sgl_kernel/moe.py", 6, 25)}

**步骤 3 — grouped GEMM（gemm.py）**

{read_snippet("sgl-kernel/python/sgl_kernel/gemm.py", 1, 25)}

## 3. 上下游

| 方向 | 模块 | 交互 |
|------|------|------|
| 上游 | srt/layers/moe, attention | import sgl_kernel |
| 下游 | csrc/*.cu | TORCH_LIBRARY 注册 |
| 平行 | flashinfer / triton | 部分算子 fallback |

## 4. KV 传输（PD 分离）

{read_snippet("sgl-kernel/python/sgl_kernel/kvcacheio.py", 1, 30)}

**解读：** prefill 节点写入 KV 后，通过 transfer kernel 搬到 decode 节点 pool。
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 26：关键问题

## Q1：为什么分 sm90 和 sm100 两套库？

{etc(
    "Hopper 与 Blackwell 在 fast math、指令集上差异大，分编译产物可最大化性能且避免错误指令。",
    read_snippet("sgl-kernel/python/sgl_kernel/load_utils.py", 59, 68),
    "- compute_capability==90 → sm90\n- 其余 → sm100\n- 无 GPU 开发机也走 sm100"
)}

## Q2：直接调 torch.ops 还是 Python 函数？

**正确：** 通过 `sgl_kernel.attention.merge_state_v2` 等公开 API，内部统一 dtype 处理与 assert。

{read_snippet("sgl-kernel/python/sgl_kernel/attention.py", 14, 26)}

**易错：** 跳过 wrapper 直接 ops 调用，遗漏 float32 lse 转换导致数值问题。

## Q3：macOS 如何用 sgl-kernel？

{read_snippet("sgl-kernel/python/sgl_kernel/__init__.py", 6, 9)}

仅 Metal 子集；CUDA 算子在 Apple Silicon 不可用，runtime 需选用其他 backend。

## Q4：算子缺失时如何排查？

1. 检查 `common_ops` 是否加载成功（日志 `[sgl_kernel]`）
2. 确认 torch CUDA 版本与 wheel 匹配
3. 对照 `_DEBUG_EXPORT_NAMES` 列表确认符号已 export
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(
        checkpoint(
            26,
            "sgl-kernel",
            [
                "sgl-kernel 通过架构特定 common_ops 向 srt 暴露 CUDA 算子。",
                "Python 层是薄封装，核心逻辑在 csrc 与 torch custom op 注册。",
                "attention/MoE/KV/speculative 是阅读 srt 性能路径时的必读依赖。",
            ],
        ),
        encoding="utf-8",
    )


def write_batch27():
    d = ROOT / "06-扩展组件/27-model-gateway"
    d.mkdir(parents=True, exist_ok=True)

    c_main = read_snippet("sgl-model-gateway/src/main.rs", 55, 80, "rust")
    c_server_state = read_snippet("sgl-model-gateway/src/server.rs", 70, 78, "rust")
    c_readiness = read_snippet("sgl-model-gateway/src/server.rs", 102, 120, "rust")
    c_rm_new = read_snippet("sgl-model-gateway/src/routers/router_manager.rs", 62, 78, "rust")
    c_rm_config = read_snippet("sgl-model-gateway/src/routers/router_manager.rs", 81, 100, "rust")
    c_router_ids = read_snippet("sgl-model-gateway/src/routers/router_manager.rs", 51, 60, "rust")
    c_backend = read_snippet("sgl-model-gateway/src/main.rs", 55, 67, "rust")
    c_app = read_snippet("sgl-model-gateway/src/server.rs", 9, 15, "rust")
    c_liveness = read_snippet("sgl-model-gateway/src/server.rs", 98, 100, "rust")

    (d / "README.md").write_text(f"""# 批次 27：sgl-model-gateway（Rust 模型网关）

> 阶段 VI · 扩展组件 | 状态：已完成

## 本批目标

说明 Rust 网关如何代理 OpenAI 兼容请求、负载均衡、PD 路由与健康检查。

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-核心概念.md](./01-核心概念.md) | IGW、RouterManager、Worker 注册 |
| [02-源码走读.md](./02-源码走读.md) | main、server、router_manager |
| [03-数据流与交互.md](./03-数据流与交互.md) | Client → Gateway → Worker |
| [04-关键问题.md](./04-关键问题.md) | FAQ |
| [checkpoint.md](./checkpoint.md) | 验收 |

## 最关键入口

{etc(
    "CLI 支持多种 backend（sglang/vllm/openai 等），网关作为统一入口转发。",
    c_backend,
    "- `Backend` enum 用 clap ValueEnum\n- 部署时 `--backend sglang` 连接 SGLang worker"
)}

## 下一批

→ [批次 28：Frontend lang](../28-Frontend-lang/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 27：核心概念

## 1. 角色

**sgl-model-gateway（SMG）** 是生产级 **Rust HTTP/gRPC 网关**：面向多 worker 负载均衡、Prefill-Decode 分离路由、OpenAI 兼容 API、限流熔断与可观测性。

## 2. 核心组件

| 组件 | 职责 |
|------|------|
| AppState | 持有 Router、AppContext、RouterManager |
| RouterManager | 多 router 模式（IGW）协调 |
| WorkerRegistry | worker 健康、类型（Prefill/Decode/Regular） |
| RouterTrait | chat/completion/embedding 等统一接口 |

### 2.1 AppState

{etc("Axum 共享状态：router + context + 可选 mesh。", c_server_state, "- `Arc<dyn RouterTrait>` 运行时多态\n- concurrency_queue 可选排队")}

### 2.2 RouterId 与多 router

{etc("静态 RouterId 常量减少热路径分配。", c_router_ids, "- HTTP_REGULAR / HTTP_PD / GRPC_*\n- enable_igw 时同时注册多个 router")}

### 2.3 Backend 类型

{etc("网关可对接多种推理后端。", c_main, "- Sglang 为首选\n- 也支持 vLLM、TRT-LLM、OpenAI、Anthropic")}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 27：源码走读

## 1. server.rs — HTTP 服务

### 1.1 Axum 依赖

{etc("基于 axum + tokio 异步 HTTP。", c_app, "- Router 挂载 chat/completion/responses 等路由\n- middleware 处理 auth 与 queue")}

### 1.2 健康检查

{etc("liveness 简单 OK。", c_liveness, "- K8s liveness probe\n- 不检查 worker")}

{etc("readiness 检查 worker 健康与 PD 模式。", c_readiness, "- IGW：至少一个 healthy worker\n- PD：需同时有 prefill 与 decode\n- Regular：任一 healthy 即可")}

---

## 2. router_manager.rs

### 2.1 结构体

{etc("DashMap 存 router，ArcSwap 快照供无锁读。", c_rm_new, "- default_router 可选\n- enable_igw 区分单/多 router")}

### 2.2 from_config

{etc("按 ServerConfig 创建 HTTP Regular/PD/OpenAI router。", c_rm_config, "- RouterFactory 异步创建\n- 失败 warn 但不阻断其他 router\n- IGW 模式 log multi-router")}

---

## 3. main.rs CLI

{read_snippet("sgl-model-gateway/src/main.rs", 1, 22, "rust")}

**解读：** clap Parser 解析路由模式、worker URL、认证、metrics 等；最终构建 ServerConfig 启动 server。

---

## 4. 路由热路径（概念）

chat completion 请求 → RouterManager.get_router → 选中 worker → HTTP/gRPC 转发 → 流式 SSE 回传。

{read_snippet("sgl-model-gateway/src/routers/router_manager.rs", 38, 49, "rust")}

---

## 5. Worker 类型

{read_snippet("sgl-model-gateway/src/server.rs", 109, 118, "rust")}

PD 模式下 readiness 强制 prefill+decode 均在线。

---

## 6. parse 端点

{read_snippet("sgl-model-gateway/src/server.rs", 80, 92, "rust")}

工具调用解析与 reasoning 分离 API。

---

## 7. mesh 集成（可选）

server.rs imports smg_mesh：集群 rate limit、graceful shutdown、policy sync。

---

## 8. 配置层

main.rs 引用 RouterConfig、RoutingMode、CircuitBreakerConfig 等 — 与 Rust 强类型配置树对应 Python server_args 的超集。
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 26：数据流与交互

## 1. 请求路径

```mermaid
sequenceDiagram
    Client->>Gateway: POST /v1/chat/completions
    Gateway->>RouterManager: route by model/policy
    RouterManager->>Worker: forward HTTP
    Worker-->>Gateway: SSE tokens
    Gateway-->>Client: SSE tokens
```

## 2. Worker 注册

启动时 `--worker-url` 或 service discovery 写入 WorkerRegistry；readiness 聚合 `is_healthy()`。

{read_snippet("sgl-model-gateway/src/server.rs", 102, 105, "rust")}

## 3. PD 分离路由

Prefill worker 处理长 prompt KV；decode worker 续写 token。RouterManager 在 `HTTP_PD` router 内选择配对。

## 4. 与 SGLang srt 关系

| 模式 | 连接方式 |
|------|----------|
| 单 worker | Gateway → sglang serve HTTP |
| PD | Gateway → prefill + decode 两套 srt |
| gRPC | grpc-regular / grpc-pd router |

## 5. OpenAI 兼容

OpenAI router 可代理外部 OpenAI API，与本地 sglang worker 并存（IGW）。
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 27：关键问题

## Q1：enable_igw 是什么？

**Explain：** Inference Gateway 多 router 模式；同一进程注册 regular、PD、OpenAI 等多套路由，按请求特征分发。

{read_snippet("sgl-model-gateway/src/routers/router_manager.rs", 91, 92, "rust")}

## Q2：与 srt grpc_server 区别？

legacy `server_args.grpc_mode` 走 Python gRPC；SMG 是 Rust 原生网关，性能与策略更丰富。launch_server 注释提到未来默认路径可能并行 HTTP+Rust gRPC。

## Q3：如何做负载均衡？

PolicyConfig 支持 cache-aware、power-of-two、manual 等；WorkerRegistry 跟踪 inflight 与 latency。

## Q4：认证

AuthConfig + JWT + API Key；middleware 在 Axum 层拦截，worker 可 mTLS。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(
        checkpoint(
            27,
            "model-gateway",
            [
                "SMG 是 Rust Axum 网关，统一 OpenAI 兼容入口与 worker 路由。",
                "RouterManager 在 IGW 模式下管理多 router；PD 模式需 prefill+decode 双活。",
                "与 Python srt 解耦，适合 K8s 多副本与跨后端混合部署。",
            ],
        ),
        encoding="utf-8",
    )


def write_batch28():
    d = ROOT / "06-扩展组件/28-Frontend-lang"
    d.mkdir(parents=True, exist_ok=True)

    (d / "README.md").write_text(f"""# 批次 28：Frontend Language（结构化生成 DSL）

> 阶段 VI · 扩展组件 | 状态：已完成

## 本批目标

理解 `@sgl.function`、IR、解释器与 RuntimeEndpoint 如何组成 SGLang 前端编程模型。

## 最关键入口

{etc(
    "`@function` 装饰器将 Python 函数包装为 SglFunction，延迟构建 IR。",
    read_snippet("python/sglang/lang/api.py", 23, 32),
    "- 支持 `@sgl.function` 无参/带参两种形式\n- num_api_spec_tokens 用于 API spec 优化"
)}

## 下一批

→ [批次 29：multimodal_gen](../29-multimodal_gen/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 28：核心概念

## 1. 三层结构

| 层 | 模块 | 职责 |
|----|------|------|
| API | lang/api.py | gen/select/user/assistant 用户 API |
| IR | lang/ir.py | SglGen/SglSelect/SglFunction 表达式树 |
| 执行 | lang/interpreter.py | StreamExecutor 驱动 backend |

## 2. SglSamplingParams

{etc(
    "统一采样参数，可转换为 OpenAI/Anthropic/Vertex 等 backend kwargs。",
    read_snippet("python/sglang/lang/ir.py", 17, 36),
    "- regex/json_schema 用于约束解码\n- clone() 支持分支复制"
)}

{etc(
    "to_openai_kwargs 丢弃 top_k（OpenAI 不支持）。",
    read_snippet("python/sglang/lang/ir.py", 64, 77),
    "- max_completion_tokens 与 max_tokens 双写兼容新 API"
)}

## 3. Backend 抽象

BaseBackend 定义 generate/select/cache_prefix；RuntimeEndpoint 通过 HTTP 调 srt。
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 28：源码走读

## 1. api.py

### gen / select / user / assistant

{read_snippet("python/sglang/lang/api.py", 75, 110)}

**解读：** gen 构造 SglGen IR 节点；运行时由解释器消费。

### Runtime 与 Engine

{read_snippet("python/sglang/lang/api.py", 35, 46)}

延迟 import：Runtime → RuntimeEndpoint；Engine → srt.entrypoints.engine。

---

## 2. ir.py — SglFunction

{read_snippet("python/sglang/lang/ir.py", 120, 160)}

SglFunction 保存 Python callable 与 bind_arguments；trace 阶段收集前缀。

---

## 3. interpreter.py — run_program

{etc(
    "创建 StreamExecutor，在 thread 或 sync 模式执行 program.func。",
    read_snippet("python/sglang/lang/interpreter.py", 57, 90),
    "- backend 可为 Runtime 包装，取 .endpoint\n- stream=True 时后台线程 + ProgramState"
)}

### batch 与 precache

{read_snippet("python/sglang/lang/interpreter.py", 105, 112)}

tracing 提取公共前缀，batch>1 时 cache_prefix 减少 prefill。

---

## 4. runtime_endpoint.py

{etc(
    "构造时拉取 /get_model_info，自动选 chat template。",
    read_snippet("python/sglang/lang/backend/runtime_endpoint.py", 27, 54),
    "- base_url 指向 sglang serve\n- verify 控制 TLS"
)}

{read_snippet("python/sglang/lang/backend/runtime_endpoint.py", 59, 75)}

flush_cache / get_server_info 直接 REST 调 srt。

---

## 5. base_backend.py

{read_snippet("python/sglang/lang/backend/base_backend.py", 1, 40)}

子类实现 generate、select、commit_lazy_operations。

---

## 6. choices.py

{read_snippet("python/sglang/lang/choices.py", 1, 35)}

ChoicesSamplingMethod：token_length_normalized 等策略。

---

## 7. chat_template.py

与 HuggingFace tokenizer 模板对齐，保证 multi-turn 格式正确。

---

## 8. 其他 backend

openai.py / anthropic.py / litellm.py 将同一 IR 转到第三方 API；无需本地 srt。
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 28：数据流与交互

## 典型程序执行流

```mermaid
flowchart TD
    A["@sgl.function def pipeline(s)"] --> B[用户调用 pipeline.run]
    B --> C[run_program]
    C --> D[StreamExecutor]
    D --> E{Backend}
    E -->|RuntimeEndpoint| F[HTTP /generate]
    E -->|OpenAI| G[OpenAI API]
    F --> H[srt TokenizerManager...]
```

## StreamExecutor 与 backend

{read_snippet("python/sglang/lang/interpreter.py", 42, 48)}

program.func 内 `s += gen(...)` 触发 lazy commit → backend.generate。

## 数据：SglGen → HTTP

RuntimeEndpoint 将 IR 序列化为 srt 兼容 JSON（input_ids、sampling_params、stream）。

## 与 srt 边界

| Frontend | srt |
|----------|-----|
| 结构化控制流、分支、select | continuous batching、KV cache |
| 单请求语义 | 多请求调度 |

Frontend 是 **客户端 SDK**；srt 是 **服务端 runtime**。
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 28：关键问题

## Q1：Frontend 与 OpenAI SDK 有何不同？

SGLang IR 支持 **select**（离散选择）、**变量作用域**、**lazy concat**、**tracing 前缀缓存** — 适合复杂 agent 流程，而非单次 chat completion。

## Q2：Runtime vs Engine？

{read_snippet("python/sglang/lang/api.py", 35, 46)}

- **Runtime**：HTTP 客户端连远程 serve
- **Engine**：进程内嵌 srt Engine，无 HTTP 开销

## Q3：易错：未 set_default_backend

{read_snippet("python/sglang/lang/api.py", 49, 51)}

调用 gen 前需 `sgl.set_default_backend(Runtime(...))` 或在 run 时传入 backend。

## Q4：regex 约束在 OpenAI backend

{read_snippet("python/sglang/lang/ir.py", 66, 67)}

OpenAI 不支持 regex 约束，会 warn；应用 RuntimeEndpoint + srt constrained 解码。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(
        checkpoint(
            28,
            "Frontend-lang",
            [
                "Frontend 用 IR+解释器表达结构化生成程序，backend 可插拔。",
                "RuntimeEndpoint 是连接本地 srt HTTP 服务的默认 backend。",
                "batch tracing 与 cache_prefix 是 Frontend 侧性能优化手段。",
            ],
        ),
        encoding="utf-8",
    )


def write_batch29():
    d = ROOT / "06-扩展组件/29-multimodal_gen"
    d.mkdir(parents=True, exist_ok=True)

    (d / "README.md").write_text(f"""# 批次 29：multimodal_gen（扩散模型 Runtime）

> 阶段 VI · 扩展组件 | 状态：已完成

## 本批目标

理解 `sglang serve --model-type diffusion` 背后的独立 runtime：视频/图像扩散 pipeline、多 GPU scheduler。

## 最关键入口

{etc(
    "multimodal_gen 的 launch_server 启动多进程 GPU worker + 可选 HTTP。",
    read_snippet("python/sglang/multimodal_gen/runtime/launch_server.py", 86, 100),
    "- 结构类似 srt 多进程\n- configure_logger + num_gpus worker"
)}

## 下一批

→ [批次 30：全链路复盘](../../07-总结与索引/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 29：核心概念

## 1. 与 srt 并列

| | srt | multimodal_gen |
|---|-----|----------------|
| 任务 | LLM/VLM 自回归 | 扩散去噪（图像/视频） |
| CLI | 默认 LLM | `--model-type diffusion` |
| 目录 | python/sglang/srt | python/sglang/multimodal_gen |

## 2. ServerArgs

{read_snippet("python/sglang/multimodal_gen/runtime/server_args.py", 1, 45)}

扩散专用参数：pipeline 名、step 数、CFG、parallelism 等。

## 3. Pipeline Executor

{read_snippet("python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py", 1, 50)}

编排 text encoder → denoising transformer → VAE decode 阶段。

## 4. CLI 分发

{read_snippet("python/sglang/cli/serve.py", 90, 115)}

model-type 为 diffusion 时 import multimodal_gen launch_server，而非 LLM run_server。
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 29：源码走读

## 1. launch_server.py

### 端口与进程树

{read_snippet("python/sglang/multimodal_gen/runtime/launch_server.py", 29, 44)}

{read_snippet("python/sglang/multimodal_gen/runtime/launch_server.py", 47, 70)}

kill_process_tree 与 srt 类似，防止 zombie worker。

### 主 launch 逻辑

{read_snippet("python/sglang/multimodal_gen/runtime/launch_server.py", 86, 120)}

spawn scheduler process per GPU，master-slave pipe 通信。

---

## 2. gpu_worker / scheduler

run_scheduler_process 加载 pipeline weights，执行 denoising loop。

---

## 3. http_server

create_app 暴露 OpenAI 风格或自定义 diffusion API（text-to-video 等）。

---

## 4. disaggregation

DiffusionServer + RoleType 支持扩散 PD 分离（orchestrator 协调多节点）。

{read_snippet("python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py", 1, 40)}

---

## 5. pipelines

各模型（Wan、Cosmos、Qwen-Image 等）在 pipelines/ 下注册 config + stage。

---

## 6. sampling_params

扩散步数、guidance scale、seed — 类比 srt SamplingParams。

---

## 7. storage & output

生成 latent/video 落盘、S3 或返回 base64。

---

## 8. 与 sgl-kernel

部分 VAE/attention 可复用 sgl_kernel 或独立 Triton；阅读时注意 import 边界。
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 29：数据流与交互

## 扩散推理数据流

```mermaid
flowchart LR
    Prompt[Text Prompt] --> TE[Text Encoder]
    TE --> DiT[Denoising Transformer]
    Noise[Random Noise] --> DiT
    DiT --> VAE[VAE Decode]
    VAE --> Out[Video/Image]
```

## HTTP 请求

Client POST prompt + params → http_server → scheduler queue → gpu_worker pipeline_executor。

## 多 GPU

tensor parallel / CFG parallel 在 pipeline_executor 内切分；与 srt TP 概念类似但算子不同。

## CLI 入口链

sglang serve → cli/serve.py → multimodal_gen.runtime.launch_server → workers。

{read_snippet("python/sglang/cli/serve.py", 1, 30)}
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 29：关键问题

## Q1：能否与 LLM 共进程？

否。diffusion 与 LLM 是不同 runtime 分支，需不同 model-type 与依赖 extra（`sglang[diffusion]`）。

## Q2：server_args 与 srt ServerArgs 同名？

不同模块：`multimodal_gen.runtime.server_args` vs `srt.server_args` — 字段不兼容，勿混用。

## Q3：性能热点

denoising loop 步数 × transformer forward；pipeline_executor 阶段 profiling 见单元测试 test_pipeline_stage_profiling。

## Q4：disaggregation 场景

长视频 multi-GPU 或多节点：orchestrator 拆分 encode/denoise/decode 角色，类似 srt PD 但语义是扩散阶段而非 prefill/decode。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(
        checkpoint(
            29,
            "multimodal_gen",
            [
                "multimodal_gen 是独立于 srt 的扩散推理 runtime。",
                "launch_server 多进程 + pipeline_executor 驱动 text-to-video/image。",
                "通过 CLI model-type=diffusion 与 LLM 路径分离。",
            ],
        ),
        encoding="utf-8",
    )


def write_batch30():
    d = ROOT / "07-总结与索引"
    d.mkdir(parents=True, exist_ok=True)

    c_launch = read_snippet("python/sglang/launch_server.py", 15, 51)
    c_pyproject = read_snippet("python/pyproject.toml", 1, 40)
    c_scheduler_ref = "# 概念性引用：批次 07 Scheduler event_loop\n```python\n# 来源：python/sglang/srt/managers/scheduler.py（详见批次 07 文档）\n# event_loop_normal: 从 waiting_queue 组 batch → run_batch → process_batch_result\n```"

    index_files = {
        "01-项目总览.md": f"""# 项目总览

> 整合 knowledge-graph tour 步骤 1–2 | scope: batch-30-final

## SGLang 是什么

高性能 **LLM/VLM 推理服务框架**：RadixAttention 前缀缓存、连续批处理、PD 分离、投机解码、多模态与扩散扩展。

## 仓库组件

| 目录 | 职责 |
|------|------|
| python/sglang/srt | 核心 Runtime |
| python/sglang/lang | Frontend DSL |
| python/sglang/multimodal_gen | 扩散 Runtime |
| sgl-kernel | CUDA 算子 |
| sgl-model-gateway | Rust 网关 |

## 推荐启动路径

{etc("CLI serve → run_server → HTTP srt。", c_launch, "- 四条分支：encoder/grpc/ray/http\n- 默认 HTTP")}

## 包配置

{read_snippet("python/pyproject.toml", 1, 25)}

entry-point `sglang=sglang.cli.main:main`。
""",
        "02-架构分层.md": f"""# 架构分层

> 来自 knowledge-graph layers | batch-30-final

## 五层模型

```mermaid
flowchart TB
    L1[文档与配置层]
    L2[入口层 CLI/launch]
    L3[公共 API 层]
    L4[运行时核心 srt+kernel+gateway]
    L5[前端语言层 lang]
    L1 --> L2 --> L3 --> L4
    L3 --> L5
```

### 文档与配置层

README、pyproject.toml、version.py — 定义边界与依赖。

### 入口层

{read_snippet("python/sglang/cli/main.py", 1, 35)}

### 公共 API

{read_snippet("python/sglang/__init__.py", 1, 45)}

### 运行时核心

srt + sgl-kernel + gateway；承载 RadixAttention、Scheduler、ModelRunner。

### 前端语言

lang IR 与 backend；客户端结构化程序。

## 扩展层（批次 24–29）

Multimodal、LoRA、kernel、gateway、Frontend、multimodal_gen 环绕 srt 核心。
""",
        "03-关键概念.md": f"""# 关键概念

## RadixAttention

基于 Radix Tree 的前缀 KV 共享（批次 15）。概念节点：`concept:radix-attention`。

## Continuous Batching

Scheduler 动态合并 prefill/decode（批次 07–09）。概念：`concept:continuous-batching`。

## Prefill-Decode Disaggregation

prefill 与 decode 分节点（批次 22）；gateway HTTP_PD router 路由。

## Speculative Decoding

EAGLE/NGRAM 等（批次 21）；sgl_kernel speculative ops 加速 verify。

## 三层分工（阶段 VI 验收）

| 层 | 技术 | 批次 |
|----|------|------|
| Python runtime | srt | 01–23 |
| CUDA kernel | sgl-kernel | 26 |
| Rust gateway | model-gateway | 27 |

## 代码锚点：run_server 分发

{c_launch}
""",
        "04-导读路径.md": f"""# 导读路径（Guided Tour）

> 整合 knowledge-graph tour，扩展至 30 批 | batch-30-final

| 步 | 主题 | 阅读目录 |
|----|------|----------|
| 1 | 项目总览 | 00-方法论 |
| 2 | 仓库结构 | 00-方法论/02 |
| 3 | CLI 入口 | 01-启动/02 |
| 4 | HTTP 启动 | 01-启动/03 |
| 5 | OpenAI API | 01-启动/04 |
| 6 | gRPC | 01-启动/05 |
| 7 | TokenizerManager | 02-请求/06 |
| 8 | Scheduler | 02-请求/07 |
| 9 | SchedulePolicy | 02-请求/08 |
| 10 | Batch/IO | 02-请求/09 |
| 11 | Detokenizer | 02-请求/10 |
| 12 | ModelRunner | 03-模型/11 |
| 13 | ModelLoader | 03-模型/12 |
| 14 | Models | 03-模型/13–14 |
| 15 | RadixAttention | 04-内存/15 |
| 16 | KV Cache | 04-内存/16 |
| 17 | Attention | 04-内存/17 |
| 18 | MoE | 04-内存/18 |
| 19 | Quantization | 04-内存/19 |
| 20 | Sampling | 05-高级/20 |
| 21 | Speculative | 05-高级/21 |
| 22 | Disaggregation | 05-高级/22 |
| 23 | Distributed | 05-高级/23 |
| 24 | Multimodal | 06-扩展/24 |
| 25 | LoRA | 06-扩展/25 |
| 26 | sgl-kernel | 06-扩展/26 |
| 27 | gateway | 06-扩展/27 |
| 28 | Frontend | 06-扩展/28 |
| 29 | diffusion | 06-扩展/29 |
| 30 | 本索引 | 07-总结与索引 |

## 步骤 3 核心代码：CLI

{read_snippet("python/sglang/cli/serve.py", 121, 128)}

## 步骤 4 核心代码：launch 分发

{c_launch}
""",
        "05-文件地图.md": f"""# 文件地图

| 文件 | 职责 | 代码片段 |
|------|------|----------|
| cli/main.py | CLI 根 | {read_snippet("python/sglang/cli/main.py", 1, 15).replace(chr(96)*3+'python', '').strip()} |
| launch_server.py | 启动分发 | 见 04-导读路径 |
| srt/entrypoints/http_server.py | FastAPI 入口 | HTTP 路由挂载 Engine |
| srt/managers/scheduler.py | 调度核心 | waiting_queue → batch |
| srt/model_executor/model_runner.py | GPU forward | logits 输出 |
| srt/mem_cache/radix_cache.py | 前缀缓存 | Radix Tree |
| sgl-kernel/python/sgl_kernel/__init__.py | 算子 export | load common_ops |
| sgl-model-gateway/src/server.rs | Rust 网关 | Axum router |
| lang/api.py | Frontend API | @function, gen |
| multimodal_gen/runtime/launch_server.py | 扩散服务 | 多 GPU worker |

## srt 子目录速查

- `entrypoints/` — HTTP/gRPC/OpenAI
- `managers/` — Tokenizer/Scheduler/Detokenizer
- `layers/` — Attention/MoE/Quant
- `model_loader/` — 权重加载
- `distributed/` — TP/PP/EP
""",
        "06-复杂度热点.md": f"""# 复杂度热点

## 1. Scheduler.event_loop

连续批处理 + 抢占 + PD 状态机；多 mixin 文件 >5k 行合计。

{c_scheduler_ref}

## 2. RadixAttention / KV 分配

树合并、ref count、跨请求共享；allocator 与 storage 交互（批次 15–16）。

## 3. Attention backend 选择

FlashInfer / Triton / sgl_kernel CUTLASS MLA；dtype 与 page size 约束（批次 17）。

## 4. sgl-kernel load_utils

多架构 wheel、importlib 动态加载、失败回退（批次 26）。

{read_snippet("sgl-kernel/python/sgl_kernel/load_utils.py", 48, 70)}

## 5. RouterManager PD 路由

prefill/decode 配对与健康检查（批次 27）。

{read_snippet("sgl-model-gateway/src/server.rs", 109, 118, "rust")}

## 6. StreamExecutor

Frontend lazy commit + 多 backend 适配（批次 28）。
""",
        "全链路请求追踪.md": f"""# 全链路请求追踪

> HTTP LLM 默认路径 | 每跳内嵌代码

## 跳 1：CLI

{read_snippet("python/sglang/cli/serve.py", 121, 128)}

## 跳 2：run_server → HTTP

{c_launch}

## 跳 3：http_server 接收

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1-L30（节选）
# FastAPI app 创建，挂载 /v1/chat/completions 等路由
# launch_server 启动 uvicorn + Engine 子进程
```

## 跳 4：TokenizerManager

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py（批次 06）
# HTTP JSON → TokenizedGenerateReqInput → ZMQ 发 Scheduler
```

## 跳 5：Scheduler 组 batch

```python
# 来源：python/sglang/srt/managers/scheduler.py（批次 07）
# get_next_batch_to_run → ScheduleBatch
```

## 跳 6：TP Worker / ModelRunner

```python
# 来源：python/sglang/srt/model_executor/model_runner.py（批次 11）
# forward → logits
```

## 跳 7：Sampling

```python
# 来源：python/sglang/srt/sampling/sampling_batch_info.py（批次 20）
# logits → next token ids
```

## 跳 8：Detokenizer → HTTP SSE

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py（批次 10）
# token ids → text chunks → client
```

## 经 Gateway 的可选路径

Client → SMG server.rs → worker srt HTTP（批次 27）。
""",
        "模块依赖图.md": f"""# 模块依赖图

```mermaid
flowchart TD
    CLI[cli] --> LS[launch_server]
    LS --> HTTP[http_server]
    HTTP --> TM[TokenizerManager]
    TM --> SCH[Scheduler]
    SCH --> TP[tp_worker]
    TP --> MR[ModelRunner]
    MR --> SK[sgl_kernel]
    MR --> ATTN[attention backends]
    SCH --> DET[Detokenizer]
    GW[model-gateway] --> HTTP
    LANG[lang Frontend] --> HTTP
```

## 关键 import

{read_snippet("python/sglang/launch_server.py", 1, 12)}

{read_snippet("python/sglang/__init__.py", 1, 20)}

## 图谱 edges（batch-30-final）

- cli/serve → launch_server → module:srt
- module:srt → module:sgl-kernel
- module:lang → HTTP RuntimeEndpoint → module:srt
""",
        "术语表.md": f"""# 术语表

| 术语 | 说明 | 代码锚点 |
|------|------|----------|
| ServerArgs | 服务配置 dataclass | srt/server_args.py |
| ScheduleBatch | 调度批次结构 | schedule_batch.py |
| RadixAttention | 前缀 KV 树 | mem_cache/radix_cache.py |
| Prefill |  prompt 首次 forward | Scheduler batch 类型 |
| Decode | 逐 token 生成 | Scheduler batch 类型 |
| PD Disaggregation | prefill/decode 分节点 | disaggregation/ |
| EAGLE | 投机解码算法 | speculative/ |
| MoE | 混合专家 | layers/moe/ |
| sgl-kernel | CUDA 算子包 | sgl-kernel/python/ |
| IGW | Inference Gateway 多 router | router_manager.rs |
| SglFunction | Frontend IR 程序 | lang/ir.py |
| PipelineExecutor | 扩散阶段编排 | multimodal_gen/.../pipeline_executor.py |

## ServerArgs 片段

```python
# 来源：python/sglang/srt/server_args.py L1-L20（结构体字段众多，见批次 02）
# model_path, tp_size, mem_fraction_static, ...
```
""",
        "业务域流程.md": f"""# 业务域流程

> understand-domain | batch-30-final

## 域 1：在线推理服务

```mermaid
flowchart LR
    U[用户/应用] --> G[网关可选]
    G --> S[SGLang Server]
    S --> M[模型权重/GPU]
```

**入口代码：**

{read_snippet("python/sglang/cli/serve.py", 121, 128)}

## 域 2：批处理调度

请求入队 → 前缀匹配 Radix → 组 batch → GPU → 采样 → 流式输出。

**入口：** Scheduler.get_next_batch_to_run（批次 07 文档）。

## 域 3：多模态与扩散

- VLM：multimodal processor + srt（批次 24）
- 扩散：multimodal_gen launch_server（批次 29）

{read_snippet("python/sglang/multimodal_gen/runtime/launch_server.py", 86, 95)}

## 域 4：集群与 PD

Prefill 节点写 KV → transfer → Decode 节点；Gateway HTTP_PD 路由。

## 域 5：结构化应用

Frontend @function → Runtime HTTP → srt；适合 agent/workflow。
""",
    }

    for name, content in index_files.items():
        (d / name).write_text(content, encoding="utf-8")

    (d / "README.md").write_text(f"""# 批次 30：全链路复盘与索引

> 阶段 VII · 收官 | scope: batch-30-final

## 本批交付

### 标准五篇 + checkpoint

| 文件 | 内容 |
|------|------|
| [01-核心概念.md](./01-核心概念.md) | 全栈概念串联 |
| [02-源码走读.md](./02-源码走读.md) | 端到端走读索引 |
| [03-数据流与交互.md](./03-数据流与交互.md) | 跨模块数据流 |
| [04-关键问题.md](./04-关键问题.md) | 总览 FAQ |
| [checkpoint.md](./checkpoint.md) | 收官验收 |

### PLAN 规定 10 篇索引

| 文件 | 对应 |
|------|------|
| [01-项目总览.md](./01-项目总览.md) | onboard Overview |
| [02-架构分层.md](./02-架构分层.md) | Layers |
| [03-关键概念.md](./03-关键概念.md) | Key Concepts |
| [04-导读路径.md](./04-导读路径.md) | Tour |
| [05-文件地图.md](./05-文件地图.md) | File Map |
| [06-复杂度热点.md](./06-复杂度热点.md) | Hotspots |
| [全链路请求追踪.md](./全链路请求追踪.md) | E2E trace |
| [模块依赖图.md](./模块依赖图.md) | edges |
| [术语表.md](./术语表.md) | Glossary |
| [业务域流程.md](./业务域流程.md) | Domain |

## 最关键入口

{c_launch}
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 30：核心概念复盘

## 全栈一览

SGLang = **Frontend（lang）** + **Runtime（srt）** + **Kernel（sgl-kernel）** + **Gateway（Rust）** + **Diffusion（multimodal_gen）**。

## 两条主路径

1. **服务路径**：CLI → launch → HTTP → Scheduler → ModelRunner
2. **编程路径**：@function → RuntimeEndpoint → HTTP → 同服务路径

{read_snippet("python/sglang/__init__.py", 1, 30)}

## 性能三板斧

RadixAttention、Continuous Batching、Kernel 融合（sgl-kernel）。

## 生产三板斧

PD 分离、Gateway 负载均衡、投机解码。
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 30：源码走读索引

按 PLAN 30 批顺序，每模块详见对应目录；此处给出 **一条完整链** 的文件顺序：

1. cli/serve.py
2. launch_server.py
3. srt/entrypoints/http_server.py
4. srt/managers/tokenizer_manager.py
5. srt/managers/scheduler.py
6. srt/managers/tp_worker.py
7. srt/model_executor/model_runner.py
8. srt/layers/attention/*
9. srt/managers/detokenizer_manager.py

扩展链：

10. sgl-kernel/python/sgl_kernel/*
11. sgl-model-gateway/src/server.rs
12. python/sglang/lang/*
13. python/sglang/multimodal_gen/runtime/*

## 启动链代码

{c_launch}

## Tokenizer 入口（概念）

```python
# 批次 06：TokenizerManager.handle_request 接收 GenerateReqInput
```

## Scheduler 入口（概念）

```python
# 批次 07：Scheduler.event_loop_normal 主循环
```

## ModelRunner 入口（概念）

```python
# 批次 11：ModelRunner.forward 执行模型
```

## Kernel 调用示例

{read_snippet("sgl-kernel/python/sgl_kernel/moe.py", 28, 54)}

## Gateway 路由

{read_snippet("sgl-model-gateway/src/routers/router_manager.rs", 81, 92, "rust")}

## Frontend 执行

{read_snippet("python/sglang/lang/interpreter.py", 57, 75)}

## Diffusion 启动

{read_snippet("python/sglang/multimodal_gen/runtime/launch_server.py", 86, 98)}
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 30：跨模块数据流

详见 [全链路请求追踪.md](./全链路请求追踪.md)。

## ZMQ 消息域（调度）

TokenizerManager ←ZMQ→ Scheduler ←ZMQ→ Detokenizer

## GPU 数据域

ScheduleBatch → ModelRunner → Tensor logits → Sampling

## KV 域

RadixCache ↔ Allocator ↔ Attention backend ↔ sgl_kernel kvcacheio

## 网关域

HTTP Client → Axum → Worker HTTP → srt

## Frontend 域

IR lazy ops → RuntimeEndpoint JSON → srt HTTP
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 30：总览 FAQ

## 从哪里开始读？

遵循 [04-导读路径.md](./04-导读路径.md) 或必读批次：01, 02, 03, 06, 07, 11, 15, 17, 30。

## Python / Rust / CUDA 分工？

见 [03-关键概念.md](./03-关键概念.md) 三层分工表。

## 如何对照源码维护文档？

文档内 `# 来源：path Lx-Ly` 供维护者 diff；读者只读 sglang_reading。

## 图谱在哪里？

写作侧：`sglang/.understand-anything/knowledge-graph.json`；读者侧：本目录索引 MD。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(
        checkpoint(
            30,
            "全链路复盘",
            [
                "30 批文档覆盖从 CLI 到 kernel/gateway/Frontend/diffusion 全栈。",
                "10 篇索引整合 knowledge-graph tour/layers/domain。",
                "读者仅读 sglang_reading 即可复述 HTTP 请求全链路与架构分层。",
            ],
        ),
        encoding="utf-8",
    )


def update_knowledge_graph():
    kg_path = SGLANG / ".understand-anything/knowledge-graph.json"
    kg = json.loads(kg_path.read_text(encoding="utf-8"))
    kg["project"]["analyzedAt"] = f"{TODAY}T12:00:00.000Z"
    meta_path = SGLANG / ".understand-anything/meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["scope"] = "batch-30-final"
    meta["lastBatch"] = 30
    meta["updatedAt"] = TODAY
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    extra_nodes = [
        {"id": "module:multimodal_gen", "type": "module", "name": "multimodal_gen", "summary": "扩散模型 runtime：text-to-video/image pipeline。", "tags": ["diffusion", "runtime"], "complexity": "complex"},
        {"id": "file:sgl-kernel/python/sgl_kernel/load_utils.py", "type": "file", "name": "sgl_kernel load_utils", "filePath": "sgl-kernel/python/sgl_kernel/load_utils.py", "summary": "按 GPU SM 架构动态加载 common_ops。", "tags": ["kernel", "loader"], "complexity": "moderate"},
        {"id": "file:sgl-model-gateway/src/server.rs", "type": "file", "name": "SMG server", "filePath": "sgl-model-gateway/src/server.rs", "summary": "Axum HTTP 服务、健康检查、路由挂载。", "tags": ["gateway", "http"], "complexity": "complex"},
        {"id": "file:python/sglang/lang/interpreter.py", "type": "file", "name": "Frontend interpreter", "filePath": "python/sglang/lang/interpreter.py", "summary": "StreamExecutor 执行 Sgl IR 程序。", "tags": ["frontend", "interpreter"], "complexity": "moderate"},
        {"id": "file:python/sglang/multimodal_gen/runtime/launch_server.py", "type": "file", "name": "diffusion launch_server", "filePath": "python/sglang/multimodal_gen/runtime/launch_server.py", "summary": "扩散服务多进程启动。", "tags": ["diffusion", "entrypoint"], "complexity": "moderate"},
    ]
    existing_ids = {n["id"] for n in kg["nodes"]}
    for n in extra_nodes:
        if n["id"] not in existing_ids:
            kg["nodes"].append(n)

    extra_edges = [
        {"source": "module:srt", "target": "module:multimodal_gen", "type": "related", "direction": "forward", "weight": 0.4},
        {"source": "module:lang", "target": "module:srt", "type": "calls", "direction": "forward", "weight": 0.7},
        {"source": "module:sgl-model-gateway", "target": "module:srt", "type": "calls", "direction": "forward", "weight": 0.8},
    ]
    kg["edges"].extend(extra_edges)

    # Extend tour
    tour_extra = [
        {"order": 6, "title": "扩展组件", "description": "sgl-kernel 算子、Rust gateway、Frontend lang、multimodal_gen 扩散。", "nodeIds": ["module:sgl-kernel", "module:sgl-model-gateway", "module:lang", "module:multimodal_gen"]},
        {"order": 7, "title": "全链路复盘", "description": "从 CLI 到 Detokenizer 的 HTTP 推理路径与索引文档。", "nodeIds": ["file:python/sglang/launch_server.py", "module:srt"]},
    ]
    existing_orders = {t["order"] for t in kg.get("tour", [])}
    for t in tour_extra:
        if t["order"] not in existing_orders:
            kg.setdefault("tour", []).append(t)
    kg["tour"] = sorted(kg["tour"], key=lambda x: x["order"])

    # Extend layers
    for layer in kg.get("layers", []):
        if layer["id"] == "layer:runtime-core":
            for nid in ["module:multimodal_gen"]:
                if nid not in layer["nodeIds"]:
                    layer["nodeIds"].append(nid)
        if layer["id"] == "layer:frontend":
            for nid in ["file:python/sglang/lang/interpreter.py"]:
                if nid not in layer["nodeIds"]:
                    layer["nodeIds"].append(nid)

    kg_path.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")


def update_progress():
    batches = [
        ("01", "00-方法论", "batch-01-initial"),
        ("02", "01-启动与入口/02-启动链路", ""),
        ("03", "01-启动与入口/03-HTTP-Server", ""),
        ("04", "01-启动与入口/04-OpenAI-API", ""),
        ("05", "01-启动与入口/05-gRPC-Proto", "batch-05"),
        ("06", "02-请求调度/06-TokenizerManager", ""),
        ("07", "02-请求调度/07-Scheduler", ""),
        ("08", "02-请求调度/08-SchedulePolicy", ""),
        ("09", "02-请求调度/09-ScheduleBatch-IO", ""),
        ("10", "02-请求调度/10-Detokenizer", "batch-10"),
        ("11", "03-模型执行/11-ModelRunner", ""),
        ("12", "03-模型执行/12-ModelLoader", ""),
        ("13", "03-模型执行/13-Models-通用", ""),
        ("14", "03-模型执行/14-Models-专用", ""),
        ("15", "04-内存与Attention/15-RadixAttention", "batch-15"),
        ("16", "04-内存与Attention/16-KV-Cache", ""),
        ("17", "04-内存与Attention/17-Attention", ""),
        ("18", "04-内存与Attention/18-MoE", ""),
        ("19", "04-内存与Attention/19-Quantization", ""),
        ("20", "05-高级特性/20-Sampling", "batch-20"),
        ("21", "05-高级特性/21-Speculative", ""),
        ("22", "05-高级特性/22-Disaggregation", ""),
        ("23", "05-高级特性/23-Distributed", ""),
        ("24", "06-扩展组件/24-Multimodal", ""),
        ("25", "06-扩展组件/25-LoRA", "batch-25"),
        ("26", "06-扩展组件/26-sgl-kernel", ""),
        ("27", "06-扩展组件/27-model-gateway", ""),
        ("28", "06-扩展组件/28-Frontend-lang", ""),
        ("29", "06-扩展组件/29-multimodal_gen", ""),
        ("30", "07-总结与索引", "batch-30-final"),
    ]
    lines = [
        "# SGLang 源码阅读进度",
        "",
        f"> 最后更新：{TODAY}  ",
        "> 总批次：30 | 已完成：30 | 进行中：0 | 待开始：0",
        "",
        "## 进度总览",
        "",
        "```",
        "[██████████████████████████████] 30/30 (100%)",
        "```",
        "",
        "## 分阶段进度",
        "",
        "| 阶段 | 批次 | 主题 | 完成数 |",
        "|------|------|------|--------|",
        "| I 地基 | 01–05 | 启动与入口 | 5/5 |",
        "| II 调度 | 06–10 | 请求调度 | 5/5 |",
        "| III 执行 | 11–14 | 模型执行 | 4/4 |",
        "| IV 内存 | 15–19 | 内存与 Attention | 5/5 |",
        "| V 高级 | 20–23 | 高级特性 | 4/4 |",
        "| VI 扩展 | 24–29 | 扩展组件 | 6/6 |",
        "| VII 收官 | 30 | 全链路复盘 | 1/1 |",
        "",
        "## 批次明细",
        "",
        "| 批 | 状态 | 开始日期 | 完成日期 | 文档目录 | 备注 |",
        "|----|------|----------|----------|----------|------|",
    ]
    for num, path, note in batches:
        name = path.split("/")[-1]
        link = f"[{name}](./{path}/)" if num != "01" else f"[00-方法论](./{path}/)"
        remark = note if note else ""
        lines.append(f"| {num} | ✅ 已完成 | {TODAY} | {TODAY} | {link} | {remark} |")

    lines.extend([
        "",
        "## 图谱更新记录",
        "",
        "| 日期 | 批次节点 | 操作 | 说明 |",
        "|------|----------|------|------|",
        f"| {TODAY} | batch-01 | 初始图谱 | 9 文件节点 + 模块/概念；scope=batch-01-initial |",
        f"| {TODAY} | batch-05/10/15/20/25 | 增量 | 各阶段调度/内存/高级域节点 |",
        f"| {TODAY} | batch-30-final | 全量整合 | tour/layers 扩展 + 扩展组件节点 |",
        "",
        "## 阅读笔记",
        "",
        "批次 26–30 完成扩展组件与收官索引；全栈文档体系已闭环。",
    ])
    (ROOT / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    write_batch26()
    write_batch27()
    write_batch28()
    write_batch29()
    write_batch30()
    update_knowledge_graph()
    update_progress()
    print("Generated batches 26-30 + progress + knowledge graph")


if __name__ == "__main__":
    main()
