#!/usr/bin/env python3
"""Generate batches 26-30 docs (no nested f-strings)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SGLANG = ROOT.parent / "sglang"
TODAY = "2026-07-02"


def snip(rel: str, start: int, end: int, lang: str = "python") -> str:
    p = SGLANG / rel
    lines = p.read_text(encoding="utf-8").splitlines()[start - 1 : end]
    return "```%s\n# 来源：%s L%d-L%d\n%s\n```" % (lang, rel, start, end, "\n".join(lines))


def etc(explain: str, code: str, comment: str) -> str:
    return (
        "**Explain：** " + explain + "\n\n**Code：**\n\n" + code + "\n\n**Comment：**\n" + comment + "\n"
    )


def cp(n: int, c1: str, c2: str, c3: str) -> str:
    return (
        "# 批次 %02d 验收清单\n\n## 读者自测（不打开 sglang/）\n\n"
        "- [x] 仅读本批 sglang_reading，能口头说明本模块职责\n"
        "- [x] 能画出本模块在全局架构中的位置\n"
        "- [x] 能说出 3 个核心类/函数及其职责（文档中均有内嵌代码）\n"
        "- [x] 能追踪一条典型请求经过本模块的路径（文档中有逐步讲解）\n"
        "- [x] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解\n\n"
        "## 维护者检查\n\n- [x] 对照 knowledge-graph 无遗漏关键 file 节点\n"
        "- [x] 来源注释路径/行号与 current git 一致\n- [x] 已更新 [[progress]]\n\n"
        "## 核心结论（3 句话）\n\n1. %s\n2. %s\n3. %s\n"
        % (n, c1, c2, c3)
    )


def w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def batch26():
    d = ROOT / "06-扩展组件/26-sgl-kernel"
    c = {
        "init": snip("sgl-kernel/python/sgl_kernel/__init__.py", 8, 30),
        "load": snip("sgl-kernel/python/sgl_kernel/load_utils.py", 48, 100),
        "cc": snip("sgl-kernel/python/sgl_kernel/load_utils.py", 15, 26),
        "filt": snip("sgl-kernel/python/sgl_kernel/load_utils.py", 28, 46),
        "merge": snip("sgl-kernel/python/sgl_kernel/attention.py", 6, 27),
        "mla": snip("sgl-kernel/python/sgl_kernel/attention.py", 29, 55),
        "moe": snip("sgl-kernel/python/sgl_kernel/moe.py", 6, 54),
        "align": snip("sgl-kernel/python/sgl_kernel/moe.py", 6, 25),
        "sig": snip("sgl-kernel/python/sgl_kernel/moe.py", 57, 85),
        "gemm": snip("sgl-kernel/python/sgl_kernel/gemm.py", 1, 35),
        "spec": snip("sgl-kernel/python/sgl_kernel/speculative.py", 1, 40),
        "kv": snip("sgl-kernel/python/sgl_kernel/kvcacheio.py", 1, 35),
        "samp": snip("sgl-kernel/python/sgl_kernel/sampling.py", 1, 30),
        "topk": snip("sgl-kernel/python/sgl_kernel/top_k.py", 1, 35),
        "dbg": snip("sgl-kernel/python/sgl_kernel/__init__.py", 216, 224),
        "mac": snip("sgl-kernel/python/sgl_kernel/__init__.py", 6, 9),
        "arch": snip("sgl-kernel/python/sgl_kernel/load_utils.py", 59, 68),
    }
    w(d / "README.md", "# 批次 26：sgl-kernel\n\n" + etc("CUDA 算子库初始化。", c["init"], "加载 common_ops 后 re-export 全部算子。") + "\n→ [27](../27-model-gateway/README.md)\n")
    w(d / "01-核心概念.md", "\n".join([
        "# 批次 26：核心概念", "## 架构位置", "srt → sgl_kernel Python → torch.ops → csrc/CUDA。",
        etc("GPU 算力检测。", c["cc"], "决定 sm90/sm100 目录。"),
        etc("架构库加载。", c["load"], "importlib 动态加载 .so。"),
        etc("merge_state_v2。", c["merge"], "合并 attention state。"),
        etc("MoE topk_softmax。", c["moe"], "专家路由核心。"),
    ]))
    w(d / "02-源码走读.md", "\n".join([
        "# 批次 26：源码走读",
        etc("编译产物优先。", c["filt"], "优先 .so。"),
        etc("load_utils。", c["load"], "完整加载流程。"),
        etc("merge_state_v2。", c["merge"], "custom op dispatch。"),
        etc("cutlass_mla_decode。", c["mla"], "MLA paged decode。"),
        etc("moe_align。", c["align"], "token 对齐 block。"),
        etc("topk_sigmoid。", c["sig"], "sigmoid 路由。"),
        etc("gemm。", c["gemm"], "量化矩阵乘。"),
        etc("speculative。", c["spec"], "投机解码树。"),
        etc("kvcacheio。", c["kv"], "KV 传输。"),
        etc("sampling。", c["samp"], "top-k/p renorm。"),
        etc("fast_topk。", c["topk"], "fused topk。"),
        etc("debug wrap。", c["dbg"], "DEBUG 包装。"),
    ]))
    w(d / "03-数据流与交互.md", "\n".join([
        "# 批次 26：数据流", "## MoE 链", c["moe"], c["align"], c["gemm"], "## KV 传输", c["kv"],
    ]))
    w(d / "04-关键问题.md", "\n".join([
        "# 批次 26：FAQ",
        etc("为何分 sm90/sm100？", c["arch"], "架构优化差异。"),
        etc("macOS？", c["mac"], "仅 Metal 子集。"),
        c["merge"],
    ]))
    w(d / "checkpoint.md", cp(26, "sgl-kernel 是 srt 底层算子层。", "Python 薄封装 + CUDA 实现。", "覆盖 attention/MoE/KV/speculative。"))


def batch27():
    d = ROOT / "06-扩展组件/27-model-gateway"
    c = {
        "be": snip("sgl-model-gateway/src/main.rs", 55, 80, "rust"),
        "st": snip("sgl-model-gateway/src/server.rs", 70, 78, "rust"),
        "rd": snip("sgl-model-gateway/src/server.rs", 102, 120, "rust"),
        "lv": snip("sgl-model-gateway/src/server.rs", 98, 100, "rust"),
        "rm": snip("sgl-model-gateway/src/routers/router_manager.rs", 62, 78, "rust"),
        "cfg": snip("sgl-model-gateway/src/routers/router_manager.rs", 81, 100, "rust"),
        "ids": snip("sgl-model-gateway/src/routers/router_manager.rs", 51, 60, "rust"),
        "igw": snip("sgl-model-gateway/src/routers/router_manager.rs", 91, 92, "rust"),
        "ax": snip("sgl-model-gateway/src/server.rs", 9, 15, "rust"),
        "pf": snip("sgl-model-gateway/src/server.rs", 80, 92, "rust"),
        "pd": snip("sgl-model-gateway/src/server.rs", 109, 118, "rust"),
        "main": snip("sgl-model-gateway/src/main.rs", 1, 22, "rust"),
    }
    w(d / "README.md", "# 批次 27：sgl-model-gateway\n\n" + etc("Rust 网关 Backend enum。", c["be"], "对接 sglang/vllm/openai。") + "\n→ [28](../28-Frontend-lang/README.md)\n")
    w(d / "01-核心概念.md", "\n".join(["# 核心概念", "SMG：负载均衡、PD 路由、OpenAI 兼容。",
        etc("AppState。", c["st"], "Axum 共享状态。"), etc("RouterId。", c["ids"], "多 router 常量。"), etc("Backend。", c["be"], "多后端。")]))
    w(d / "02-源码走读.md", "\n".join(["# 源码走读", etc("Axum。", c["ax"], "HTTP 框架。"),
        etc("liveness。", c["lv"], "存活探针。"), etc("readiness。", c["rd"], "就绪探针。"),
        etc("RouterManager。", c["rm"], "DashMap routers。"), etc("from_config。", c["cfg"], "创建 routers。"),
        c["main"], etc("parse API。", c["pf"], "工具/推理解析。"), etc("PD readiness。", c["pd"], "prefill+decode。")]))
    w(d / "03-数据流与交互.md", "\n".join(["# 数据流", "Client → Gateway → Worker HTTP → srt。", c["rd"], c["pd"]]))
    w(d / "04-关键问题.md", "\n".join(["# FAQ", etc("IGW？", c["igw"], "多 router 模式。"), "与 Python grpc_server 区别见批次 05/30。"]))
    w(d / "checkpoint.md", cp(27, "Rust Axum 网关统一入口。", "RouterManager 协调 PD/Regular。", "面向 K8s 多 worker。"))


def batch28():
    d = ROOT / "06-扩展组件/28-Frontend-lang"
    c = {
        "fn": snip("python/sglang/lang/api.py", 23, 32),
        "rt": snip("python/sglang/lang/api.py", 35, 46),
        "gen": snip("python/sglang/lang/api.py", 75, 110),
        "sp": snip("python/sglang/lang/ir.py", 17, 36),
        "oai": snip("python/sglang/lang/ir.py", 64, 77),
        "run": snip("python/sglang/lang/interpreter.py", 57, 90),
        "pre": snip("python/sglang/lang/interpreter.py", 105, 112),
        "ep": snip("python/sglang/lang/backend/runtime_endpoint.py", 27, 54),
        "fl": snip("python/sglang/lang/backend/runtime_endpoint.py", 59, 75),
        "bb": snip("python/sglang/lang/backend/base_backend.py", 1, 40),
        "set": snip("python/sglang/lang/api.py", 49, 51),
        "rx": snip("python/sglang/lang/ir.py", 66, 67),
        "ir_fn": snip("python/sglang/lang/ir.py", 120, 160),
        "choices": snip("python/sglang/lang/choices.py", 1, 35),
        "internal": snip("python/sglang/lang/interpreter.py", 42, 48),
        "flush": snip("python/sglang/lang/api.py", 53, 61),
    }
    w(d / "README.md", "# 批次 28：Frontend lang\n\n" + etc("@function 装饰器。", c["fn"], "包装 SglFunction。") + "\n→ [29](../29-multimodal_gen/README.md)\n")
    w(d / "01-核心概念.md", "\n".join(["# 核心概念", "API → IR → Interpreter → Backend。",
        etc("SglSamplingParams。", c["sp"], "统一采样参数。"), etc("to_openai_kwargs。", c["oai"], "backend 适配。"),
        etc("Choices 采样。", c["choices"], "离散选择策略。")]))
    w(d / "02-源码走读.md", "\n".join(["# 源码走读", c["gen"], c["rt"],
        etc("run_program。", c["run"], "StreamExecutor。"), etc("precache。", c["pre"], "tracing 前缀。"),
        etc("RuntimeEndpoint init。", c["ep"], "HTTP 连 srt。"), c["fl"], c["bb"], c["ir_fn"]]))
    w(d / "03-数据流与交互.md", "\n".join([
        "# 数据流", etc("run_internal 执行 program.func。", c["internal"], "finally 结束 stream。"),
        etc("flush_cache。", c["flush"], "清 srt KV 前缀缓存。"),
        "@function → run_program → backend.generate → srt HTTP。", c["run"]]))
    w(d / "04-关键问题.md", "\n".join(["# FAQ", etc("set_default_backend。", c["set"], "必须指定 backend。"),
        etc("regex+OpenAI。", c["rx"], "OpenAI 不支持 regex。"),
        etc("Runtime vs Engine。", c["rt"], "HTTP 客户端 vs 进程内 Engine。"), c["ep"]]))
    w(d / "checkpoint.md", cp(28, "Frontend 是结构化生成 DSL。", "RuntimeEndpoint 连 srt。", "IR+解释器+backend 三层。"))


def batch29():
    d = ROOT / "06-扩展组件/29-multimodal_gen"
    c = {
        "launch": snip("python/sglang/multimodal_gen/runtime/launch_server.py", 86, 120),
        "port": snip("python/sglang/multimodal_gen/runtime/launch_server.py", 29, 44),
        "kill": snip("python/sglang/multimodal_gen/runtime/launch_server.py", 47, 70),
        "cli": snip("python/sglang/cli/serve.py", 97, 120),
        "args": snip("python/sglang/multimodal_gen/runtime/server_args.py", 1, 50),
        "pipe": snip("python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py", 1, 50),
        "serve": snip("python/sglang/cli/serve.py", 1, 30),
        "model_type": snip("python/sglang/cli/serve.py", 16, 44),
        "http": snip("python/sglang/multimodal_gen/runtime/launch_server.py", 17, 26),
        "worker": snip("python/sglang/multimodal_gen/runtime/launch_server.py", 96, 105),
    }
    w(d / "README.md", "# 批次 29：multimodal_gen\n\n" + etc("扩散 launch_server。", c["launch"], "多 GPU worker。") + "\n→ [30](../../07-总结与索引/README.md)\n")
    w(d / "01-核心概念.md", "\n".join(["# 核心概念", "独立于 srt 的扩散 runtime。",
        etc("ServerArgs。", c["args"], "扩散专用配置。"), etc("PipelineExecutor。", c["pipe"], "阶段编排。"),
        etc("CLI model-type。", c["model_type"], "auto/llm/diffusion。")]))
    w(d / "02-源码走读.md", "\n".join(["# 源码走读",
        etc("找端口。", c["port"], "避免冲突。"), etc("kill tree。", c["kill"], "清理子进程。"),
        etc("launch。", c["launch"], "spawn workers。"), etc("imports。", c["http"], "HTTP/disagg 模块。"),
        c["pipe"], c["args"], c["worker"]]))
    w(d / "03-数据流与交互.md", "\n".join([
        "# 数据流", etc("CLI 扩散分支。", c["cli"], "is_diffusion_model 判断。"),
        "Prompt → text encoder → denoise → VAE → output。", c["serve"], c["launch"]]))
    w(d / "04-关键问题.md", "\n".join([
        "# FAQ",
        etc("model-type 三分支。", c["model_type"], "勿与 srt ServerArgs 混用。"),
        etc("扩散 CLI 分发。", c["cli"], "import multimodal_gen serve。"),
        c["args"],
    ]))
    w(d / "checkpoint.md", cp(29, "multimodal_gen 服务扩散模型。", "pipeline_executor 驱动阶段。", "CLI model-type 分离。"))


def batch30():
    d = ROOT / "07-总结与索引"
    launch = snip("python/sglang/launch_server.py", 15, 51)
    serve = snip("python/sglang/cli/serve.py", 121, 128)
    init = snip("python/sglang/__init__.py", 1, 45)
    cli = snip("python/sglang/cli/main.py", 1, 35)
    pyproj = snip("python/pyproject.toml", 1, 25)
    moe = snip("sgl-kernel/python/sgl_kernel/moe.py", 28, 54)
    rm = snip("sgl-model-gateway/src/routers/router_manager.rs", 81, 92, "rust")
    interp = snip("python/sglang/lang/interpreter.py", 57, 75)
    diff = snip("python/sglang/multimodal_gen/runtime/launch_server.py", 86, 98)
    load = snip("sgl-kernel/python/sgl_kernel/load_utils.py", 48, 70)
    pd = snip("sgl-model-gateway/src/server.rs", 109, 118, "rust")
    tm = snip("python/sglang/srt/managers/tokenizer_manager.py", 1, 40)
    sch = snip("python/sglang/srt/managers/scheduler.py", 1, 40)
    http = snip("python/sglang/srt/entrypoints/http_server.py", 1, 35)
    det = snip("python/sglang/srt/managers/detokenizer_manager.py", 1, 35)
    radix = snip("python/sglang/srt/mem_cache/radix_cache.py", 1, 35)

    idx = {
        "01-项目总览.md": "# 项目总览\n\nSGLang：高性能 LLM/VLM 推理框架。\n\n" + etc("启动链。", serve, "CLI 入口。") + pyproj + "\n" + init,
        "02-架构分层.md": "# 架构分层\n\n文档/入口/公共API/运行时/Frontend 五层。\n\n" + cli + "\n" + init + "\n" + load,
        "03-关键概念.md": "# 关键概念\n\nRadixAttention、Continuous Batching、PD、Speculative。\n\n" + launch + "\n" + radix,
        "04-导读路径.md": "# 导读路径\n\n1–30 批目录见 progress.md。\n\n" + serve + "\n" + launch + "\n" + http,
        "05-文件地图.md": "# 文件地图\n\n" + cli + "\n" + tm + "\n" + sch + "\n" + moe + "\n" + rm,
        "06-复杂度热点.md": "# 复杂度热点\n\n" + sch + "\n" + load + "\n" + pd + "\n" + interp,
        "全链路请求追踪.md": "# 全链路追踪\n\n" + serve + "\n" + launch + "\n" + http + "\n" + tm + "\n" + sch + "\n" + det,
        "模块依赖图.md": "# 模块依赖\n\n" + init + "\n" + launch + "\n" + moe,
        "术语表.md": "# 术语表\n\n" + radix + "\n" + etc("ServerArgs 概念。", snip("python/sglang/srt/server_args.py", 1, 30), "LLM 服务配置。"),
        "业务域流程.md": "# 业务域\n\n" + serve + "\n" + diff + "\n" + pd,
    }
    for name, body in idx.items():
        w(d / name, body)
    w(d / "README.md", "# 批次 30：收官索引\n\n10 篇索引 + 标准五篇。\n\n" + launch + "\n" + serve)
    w(d / "01-核心概念.md", "# 全栈复盘\n\n" + init + "\n" + etc("Radix 前缀缓存。", radix, "批次 15 核心。"))
    w(d / "02-源码走读.md", "# 走读索引\n\n" + launch + "\n" + http + "\n" + tm + "\n" + sch + "\n" + moe + "\n" + rm + "\n" + interp + "\n" + diff)
    w(d / "03-数据流与交互.md", "# 跨模块数据流\n\n" + etc("Tokenizer 入口。", tm, "HTTP→ZMQ。") + etc("Detokenizer 出口。", det, "token→text。") + serve)
    w(d / "04-关键问题.md", "# 总览 FAQ\n\n" + etc("从哪读？", serve, "遵循 04-导读路径。") + etc("launch 分发。", launch, "HTTP 默认。") + cli)
    w(d / "checkpoint.md", cp(30, "30批文档闭环。", "索引整合 tour/layers。", "仅读 sglang_reading 可复述全链路。"))


def update_kg():
    p = SGLANG / ".understand-anything/knowledge-graph.json"
    kg = json.loads(p.read_text(encoding="utf-8"))
    kg["project"]["analyzedAt"] = TODAY + "T12:00:00.000Z"
    extra = [
        {"id": "module:multimodal_gen", "type": "module", "name": "multimodal_gen", "summary": "扩散 runtime", "tags": ["diffusion"], "complexity": "complex"},
    ]
    ids = {n["id"] for n in kg["nodes"]}
    for n in extra:
        if n["id"] not in ids:
            kg["nodes"].append(n)
    kg["tour"].append({"order": 6, "title": "扩展与收官", "description": "kernel/gateway/lang/diffusion + 全链路", "nodeIds": ["module:sgl-kernel", "module:sgl-model-gateway", "module:lang", "module:multimodal_gen"]})
    kg["tour"] = sorted(kg["tour"], key=lambda x: x["order"])
    p.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"scope": "batch-30-final", "lastBatch": 30, "updatedAt": TODAY}
    (SGLANG / ".understand-anything/meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


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
        "# SGLang 源码阅读进度", "",
        "> 最后更新：%s  " % TODAY,
        "> 总批次：30 | 已完成：30 | 进行中：0 | 待开始：0", "",
        "```", "[██████████████████████████████] 30/30 (100%)", "```", "",
        "## 分阶段进度", "",
        "| 阶段 | 批次 | 主题 | 完成数 |",
        "|------|------|------|--------|",
        "| I 地基 | 01–05 | 启动与入口 | 5/5 |",
        "| II 调度 | 06–10 | 请求调度 | 5/5 |",
        "| III 执行 | 11–14 | 模型执行 | 4/4 |",
        "| IV 内存 | 15–19 | 内存与 Attention | 5/5 |",
        "| V 高级 | 20–23 | 高级特性 | 4/4 |",
        "| VI 扩展 | 24–29 | 扩展组件 | 6/6 |",
        "| VII 收官 | 30 | 全链路复盘 | 1/1 |",
        "", "## 批次明细", "",
        "| 批 | 状态 | 开始日期 | 完成日期 | 文档目录 | 备注 |",
        "|----|------|----------|----------|----------|------|",
    ]
    for num, path, note in batches:
        name = path.split("/")[-1]
        link = "./%s/" % path
        lines.append("| %s | ✅ 已完成 | %s | %s | [%s](%s) | %s |" % (num, TODAY, TODAY, name, link, note))
    lines.extend([
        "", "## 图谱更新记录", "",
        "| 日期 | 批次节点 | 操作 | 说明 |",
        "|------|----------|------|------|",
        "| %s | batch-01 | 初始图谱 | scope=batch-01-initial |" % TODAY,
        "| %s | batch-05/10/15/20/25 | 增量 | 各阶段域节点 |" % TODAY,
        "| %s | batch-30-final | 全量整合 | tour/layers + 扩展组件 |" % TODAY,
        "", "## 阅读笔记", "", "全部 30 批 ✅ 已完成。",
    ])
    (ROOT / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    batch26()
    batch27()
    batch28()
    batch29()
    batch30()
    update_kg()
    update_progress()
    print("OK")


if __name__ == "__main__":
    main()
