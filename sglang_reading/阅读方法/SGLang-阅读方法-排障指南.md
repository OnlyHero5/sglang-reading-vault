---
title: "阅读方法 · 排障指南"
type: troubleshooting
framework: sglang
topic: "阅读方法"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 阅读方法 · 排障指南

## 你为什么要读

这页不是生产故障手册，而是“阅读卡住时的排障”：SGLang 与 vLLM 怎么比较、入口文件怎么找、安装包和源码树为什么对不上。先把阅读方法校准，后续遇到真正的 Scheduler、KV 或 kernel 问题，才不会从错误目录开始推理。

## Q1：SGLang 和 vLLM 有什么区别？

**读法：** 两者都是 LLM **推理服务引擎**，但 SGLang 额外强调：

1. **RadixAttention** 前缀缓存（vLLM 以 PagedAttention 为主）
2. **Frontend DSL**（vLLM 无等价物）
3. **PD 分离、投机解码、结构化输出** 的一等公民支持

**源码锚点（官方自我定位）：**

```markdown
## 来源：README.md L66-L67
- **Fast Runtime**: Provides efficient serving with RadixAttention for prefix caching, a zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-LoRA batching.
- **Broad Model Support**: Supports a wide range of language models (Llama, Qwen, DeepSeek, Kimi, GLM, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse), reward models (Skywork), and diffusion models (WAN, Qwen-Image), with easy extensibility for adding new models. Compatible with most Hugging Face models and OpenAI APIs.
```

**中文释义：** 官方自我定位强调两点：一是 Runtime 侧的调度、缓存、并行、量化和 LoRA 能力；二是模型覆盖面广，既支持主流语言模型，也覆盖 embedding、reward 和 diffusion 模型，并兼容 Hugging Face 与 OpenAI API 生态。

**要点：** 性能对比随硬件/模型变化，不宜绝对化；读源码时关注**调度 + 缓存 + 内核**三条线即可。

---

## Q2：`srt` 和 `lang` 各干什么？能否只用其中一个？

| 包 | 用途 | 单独使用？ |
|----|------|------------|
| `srt` | 模型推理、批调度、KV 缓存 | ✅ `sglang serve` 仅依赖 srt |
| `lang` | 结构化程序（分支、约束、多轮） | ⚠️ 需连接已运行的 Runtime（HTTP） |

**源码锚点：**

```markdown
## 来源：python/sglang/README.md L4-L6
- `lang`: The frontend language.
- `multimodal_gen`: Inference framework for accelerated image/video generation.
- `srt`: The backend engine for running local models. (SRT = SGLang Runtime).
```

**中文释义：** `lang` 是前端语言，`multimodal_gen` 是加速图像/视频生成的推理框架，`srt` 是运行本地模型的后端引擎。

**要点：** 生产部署通常 **只跑 srt**；研究/复杂 prompt 逻辑才用 lang。

---

## Q3：为什么有两个 `Engine`？

**读法：** `__init__.py` 先从 `lang.api` 导入 `Engine`，又用 LazyImport 从 `srt.entrypoints.engine` 覆盖同名符号。

**源码锚点：**

```python
## 来源：python/sglang/__init__.py L36-L37 vs L79
from sglang.lang.api import (
    Engine,
```

**要点：**

- 实际 `from sglang import Engine` 得到的是 **Runtime Engine**（LazyImport 后者覆盖）。
- 这是常见易错点：读类型提示或文档时需确认是哪一个 Engine。

---

## Q4：`python -m sglang.launch_server` 和 `sglang serve` 差在哪？

**源码锚点（launch_server 自己承认）：**

```python
## 来源：python/sglang/launch_server.py L55-L58
    warnings.warn(
        "'python -m sglang.launch_server' is still supported, but "
        "'sglang serve' is the recommended entrypoint.\n"
        "  Example: sglang serve --model-path <model> [options]",
```

**对比：**

| 特性 | `sglang serve` | `python -m sglang.launch_server` |
|------|----------------|-------------------------------------|
| Diffusion 自动检测 | ✅ | ❌ |
| `--model-type` | ✅ | ❌ |
| 参数解析 | cli/serve + prepare_server_args | 直接 prepare_server_args |

---

## Q5：import sglang 很慢 / 副作用？

**读法：** `__init__.py` 会执行 HF patches、可能安装 stub，**不适合**在轻量脚本中随意 import。

**源码锚点：**

```python
## 来源：python/sglang/__init__.py L29-L32
from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches

_apply_hf_patches()
del _apply_hf_patches
```

**要点：** 若仅需版本号，用 `sglang version` CLI 而非 import。

---

## Q6：Monorepo 里的目录应该从哪里读？

| 目录 | 阅读专题 |
|------|----------|
| `python/sglang/srt/` | [[SGLang-请求调度]] · [[SGLang-模型执行]] · [[SGLang-内存与Attention]] |
| `sgl-kernel/` | [[SGLang-sgl-kernel]] |
| `sgl-model-gateway/` | [[SGLang-model-gateway]] |
| `python/sglang/lang/` | [[SGLang-前端语言]] |
| `python/sglang/multimodal_gen/` | [[SGLang-多模态生成]] |
| `test/`, `benchmark/` | 从对应专题的运行验证与性能实验进入 |

---

## Q7：本 sglang_reading 文档与源码不一致怎么办？

**读法：** 文档内嵌代码标注了 `# 来源：path Lxx-Lyy` 与 git commit `70df09b`。上游更新后，维护者应：

1. 对照 `sglang/` diff 更新内嵌块
2. 运行 `node maintenance/audit_source_evidence.mjs` 检查引用文件和行号范围
3. 用对应单测或最小运行实验重新验证行为

**要点：** 笔记负责解释，upstream 源码和运行结果负责裁决；两者冲突时以当前源码与验证为准。

---

## 验证建议（零基础可试）

以下三条**不要求 GPU**，可在读完本模块后立刻动手，确认概念与真实环境一致。

### 1. 查看 CLI 子命令

```bash
sglang --help
```

**预期结果：** 输出中包含 `serve`、`generate`、`version` 等子命令说明；`serve` 一节列出 `--model-path` 等常用参数。这对应 [[SGLang-阅读方法-源码走读]] 中的 `cli/main.py` 路由。

### 2. 确认安装包与版本

```bash
pip show sglang
```

**预期结果：** 显示 `Name: sglang`、`Version: …`、`Location: …/site-packages`。若未安装，先执行 `pip install sglang`（或按 [[SGLang-零基础先修]] 指引）。版本号可与文档标注的 git commit `70df09b` 对照，维护者 diff 时以 commit 为准。

### 3. 在官方 README 中定位 SRT

打开 SGLang 仓库或 PyPI 项目页的 `README.md`，搜索 **`srt`** 或 **Code Structure**。

**预期结果：** 找到类似「`srt`: The backend engine for running local models. (SRT = SGLang Runtime).」的条目——这与 [[SGLang-阅读方法-核心概念]] 的 Monorepo 表一致。若你只有 pip 包而无源码树，可在 `site-packages/sglang/README.md` 或 GitHub 网页版 README 中完成此步。
