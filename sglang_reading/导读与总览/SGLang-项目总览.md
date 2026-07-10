---
title: "SGLang 项目总览"
type: concept
framework: sglang
topic: "导读与总览"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# SGLang 项目总览

> 对应源码基线 `70df09b`；先看系统边界，再按任务进入源码证据。

## 你为什么要读

SGLang 是 monorepo：Python runtime、CUDA 算子、Rust 网关和多模态生成住在同一屋檐下，却不是同一条执行链。本篇先给你一张“园区地图”，说明每块代码解决什么系统问题；读完后，再看到一个目录名时，你应该能判断它属于请求入口、GPU 执行、集群路由还是独立生成运行时。

## 如果你是完全新手

若你**从未部署过大模型推理服务**，不清楚 prefill 与 decode、TTFT、KV Cache 或「为什么需要三个进程」，请先读 **[[SGLang-零基础先修]]**。该篇用餐厅类比讲清 LLM serving 的核心直觉，约 30 分钟，无需 PyTorch 或 CUDA 背景。

读完先修篇后，再回到本篇建立 monorepo 与启动链的全局视图；然后沿 **[[SGLang-HTTP请求全链路]]** 把一次 HTTP 请求从进入到流式响应走通，再按 **[[SGLang-学习路径]]** 深入入口、调度与模型执行。术语卡住时查 **[[SGLang-术语表#零基础速查生活类比版|术语表 · 零基础速查]]**。

若你已有 vLLM、TensorRT-LLM 等 serving 经验，可跳过先修篇，直接从下文「SGLang 是什么」开始；框架差异见 [[SGLang-框架对比与设计决策]]。

---

## SGLang 是什么

SGLang（Structured Generation Language）是 LMSYS 团队主导的开源**大模型推理服务框架**。它同时提供：

- **Runtime（SRT）**：`python/sglang/srt` — 高性能服务引擎，对标 vLLM、TensorRT-LLM
- **Frontend Language**：`python/sglang/lang` — 结构化生成 DSL，用于复杂 prompt 逻辑
- **扩展生态**：`sgl-kernel`（CUDA 算子）、`sgl-model-gateway`（Rust 路由网关）、`multimodal_gen`（扩散/视频）

**读法：** 日常部署说的「跑 SGLang 服务」，核心是 Runtime。用户执行 `sglang serve --model-path ...` 后，请求经 HTTP/OpenAI API 进入调度器，在 GPU 上连续批处理执行，最终流式返回 token。

**源码锚点：**

```python
## 来源：python/sglang/cli/serve.py L121-L128
        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            server_args = prepare_server_args(dispatch_argv)

            run_server(server_args)
```

**要点：**

- `prepare_server_args` 将 CLI 参数解析为统一的 `ServerArgs` 对象（模型路径、并行度、量化、disaggregation 等）。
- `run_server` 根据 flags 在 HTTP / gRPC / Ray / Encoder 四条路径间分发（见启动链路–HTTP Server）。
- 首次学习可以先用短主线建立全局模型；要修改实现或确认版本差异时，必须回到 `sglang/` 源码与测试核对。

---

## Monorepo 顶层结构

| 目录 | 语言 | 职责 | 阅读专题 |
|------|------|------|----------|
| `python/sglang/srt/` | Python | 推理运行时核心 | [[SGLang-请求调度]] · [[SGLang-模型执行]] |
| `python/sglang/lang/` | Python | Frontend DSL | [[SGLang-前端语言]] |
| `python/sglang/multimodal_gen/` | Python | 扩散/视频生成 | [[SGLang-多模态生成]] |
| `sgl-kernel/` | CUDA/C++ | 高性能算子 | [[SGLang-sgl-kernel]] |
| `sgl-model-gateway/` | Rust | 模型路由网关 | [[SGLang-model-gateway]] |
| `test/`, `benchmark/` | — | 测试与基准（概念引用） | — |

**读法：** Python 包只是 monorepo 的一部分；性能关键路径会下沉到 `sgl-kernel` CUDA 扩展，生产多节点部署可能前置 `sgl-model-gateway`。

**源码锚点：**

```toml
## 来源：python/pyproject.toml L178-L180
[project.scripts]
sglang = "sglang.cli.main:main"
killall_sglang = "sglang.cli.killall:main"
```

**要点：** pip 安装后在 PATH 生成 `sglang` 可执行文件；`killall_sglang` 用于清理残留进程。

---

## 核心能力（官方定位）

| 能力 | 源码锚点 | 阅读专题 |
|------|----------|----------|
| RadixAttention 前缀缓存 | `mem_cache/radix_cache.py` | [[SGLang-RadixAttention]] · [[SGLang-KV-Cache]] |
| Continuous Batching | `managers/scheduler.py` | [[SGLang-Scheduler]] · [[SGLang-SchedulePolicy]] |
| PD Disaggregation | `disaggregation/` | [[SGLang-PD分离]] |
| Speculative Decoding | `speculative/eagle_worker_v2.py` | [[SGLang-Speculative]] |
| 多 LoRA 批处理 | `lora/lora_manager.py` | [[SGLang-LoRA]] |
| OpenAI 兼容 API | `entrypoints/openai/` | OpenAI API |

---

## 技术栈

- **语言**：Python（主）、Rust（gRPC/gateway）、CUDA/C++（kernel）
- **框架**：FastAPI（HTTP）、PyTorch（模型）、FlashInfer/Triton（Attention）
- **通信**：ZMQ IPC（Tokenizer ↔ Scheduler ↔ Detokenizer）

---

## 怎么继续阅读

1. 零基础先读 [[SGLang-零基础先修]]，再从 [[SGLang-阅读方法]] 了解 阅读方法与全专题地图
2. 按 [[SGLang-学习路径]] 的语义路线阅读（建议路径：先修 → 本篇总览 → 全链路 → 入口、调度与模型执行）
3. 遇到术语查 [[SGLang-术语表|术语表]]；需要全局定位查 [[SGLang-源码地图]]
4. 追踪端到端请求见 [[SGLang-HTTP请求全链路|全链路请求追踪]]
