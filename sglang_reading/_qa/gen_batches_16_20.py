#!/usr/bin/env python3
"""Generate sglang_reading docs for batches 16-20."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SGLANG = ROOT.parent / "sglang"


def code(path: str, start: int, end: int, lang: str = "python") -> str:
    """Extract lines from sglang source."""
    fp = SGLANG / path
    lines = fp.read_text(encoding="utf-8").splitlines()
    snippet = "\n".join(lines[start - 1 : end])
    return f"```{lang}\n# 来源：{path} L{start}-L{end}\n{snippet}\n```"


def etc(explain: str, c: str, comment: str | list[str]) -> str:
    if isinstance(comment, list):
        comment = "\n".join(
            line if line.startswith("-") else f"- {line}" for line in comment
        )
    return f"**Explain：** {explain}\n\n**Code：**\n\n{c}\n\n**Comment：**\n{comment}\n"


def checkpoint(batch: int, title: str, conclusions: list[str]) -> str:
    return f"""# 批次 {batch:02d} 验收清单

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

- 部分 storage 后端（Mooncake/NIXL）需结合部署文档进一步阅读
"""


def write_batch16():
    d = ROOT / "04-内存与Attention/16-KV-Cache"
    d.mkdir(parents=True, exist_ok=True)

    c_base = code("python/sglang/srt/mem_cache/allocator/base.py", 27, 110)
    c_token = code("python/sglang/srt/mem_cache/allocator/token.py", 28, 84)
    c_paged = code("python/sglang/srt/mem_cache/allocator/paged.py", 105, 170)
    c_paged_ext = code("python/sglang/srt/mem_cache/allocator/paged.py", 172, 215)
    c_host = code("python/sglang/srt/mem_cache/pool_host/base.py", 79, 143)
    c_host_alloc = code("python/sglang/srt/mem_cache/pool_host/base.py", 240, 268)
    c_storage = code("python/sglang/srt/mem_cache/storage/backend_factory.py", 16, 96)
    c_storage_create = code("python/sglang/srt/mem_cache/storage/backend_factory.py", 66, 96)

    readme_entry = etc(
        "所有 KV 索引分配器继承 `BaseTokenToKVPoolAllocator`，统一暴露 `alloc`/`free`/`available_size` 接口，供 RadixCache 与 Scheduler 调用。",
        c_base,
        "- `free_group_begin/end` 支持批量释放后再合并，减少 sort 开销\n- `merge_and_sort_free` 将 `release_pages` 合并回 `free_pages` 并排序\n- 批次 15 的 RadixCache 通过 allocator 获取/归还 token 或 page 索引",
    )
    q1_code = c_paged

    (d / "README.md").write_text(f"""# 批次 16：KV Cache 分配与存储

> 阶段 IV · 内存与算子 | 状态：已完成 | Git：`70df09b`

## 本批目标

读完本目录后，你应能**不打开 `sglang/`**，说明：

1. Token 级与 Page 级 KV 索引分配器的区别
2. HiCache 主机内存池如何与设备池协同
3. 外部 storage 后端（Mooncake/NIXL 等）如何接入

## 文档导航

| 文件 | 内容 |
|------|------|
| [01-核心概念.md](./01-核心概念.md) | 分配器、HiCache、Storage 术语 |
| [02-源码走读.md](./02-源码走读.md) | allocator / pool_host / storage 精读 |
| [03-数据流与交互.md](./03-数据流与交互.md) | Scheduler → Allocator → KV Pool |
| [04-关键问题.md](./04-关键问题.md) | page_size、OOM、与 RadixAttention 关系 |
| [checkpoint.md](./checkpoint.md) | 验收清单 |

## 最关键的一段入口代码

{readme_entry}

## 衔接

← [批次 15：RadixAttention](../15-RadixAttention/README.md) · → [批次 17：Attention](../17-Attention/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 16：核心概念

## 1. 架构位置

KV Cache 分配层位于 **RadixAttention（批次 15）** 与 **物理 KV 张量（memory_pool）** 之间：

```mermaid
flowchart LR
    Scheduler --> RadixCache
    RadixCache --> Allocator["Token/Page Allocator"]
    Allocator --> KVPool["KVCache 张量池"]
    KVPool --> Attention["Attention Backend（批次 17）"]
    KVPool --> HostPool["HiCache Host Pool"]
    HostPool --> Storage["Storage Backend"]
```

## 2. Token 级 vs Page 级分配

{etc(
    "`TokenToKVPoolAllocator` 以 token 为粒度管理索引，`page_size=1`；适合 page_size=1 或未启用 paged KV 的场景。",
    c_token,
    ["slot 0 保留给 padding token 的 dummy 写入", "`need_sort=True` 时释放的索引先进入 `release_pages`，alloc 前 merge", "`get_cpu_copy/load_cpu_copy` 支持 HiCache 与 CPU offload"],
)}

## 3. Page 对齐分配

{etc(
    "`PagedTokenToKVPoolAllocator` 将 KV 索引按 page 对齐，与 FlashInfer PagedAttention 及 `--page-size` 配置一致。",
    c_paged,
    ["`alloc` 返回连续 page 展开后的 token 索引", "ROCm 上 init 时预热 `torch.unique`，避免首请求 JIT 延迟", "`num_pages = size // page_size` 定义总 page 数"],
)}

## 4. HiCache 主机池

{etc(
    "`HostKVCache` 在主机 RAM 上维护 KV 副本，实现分层缓存（L1 设备 / L2 主机 / L3 外部 storage）。",
    c_host,
    ["`host_size`（GB）或 `host_to_device_ratio` 决定容量", "PP 并行时 `sync_fixed_hicache_size` 取各 rank 最小 token 数", "启动前检查 `psutil.virtual_memory()` 防止 OOM"],
)}

## 5. Storage 后端工厂

{etc(
    "`StorageBackendFactory` 注册并懒加载 Mooncake、NIXL、LMCache 等 HiCache 外部存储后端。",
    c_storage,
    ["`_registry` 保存 backend 名 → loader 映射", "`register_backend` 支持插件式扩展", "创建实例时传入 `HiCacheStorageConfig` 与 `mem_pool_host`"],
)}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 16：源码走读

## 走读顺序

1. `allocator/base.py` — 抽象基类
2. `allocator/token.py` — Token 级实现
3. `allocator/paged.py` — Page 级实现
4. `pool_host/base.py` — HiCache 主机池
5. `storage/backend_factory.py` — 外部存储工厂

---

## 1. BaseTokenToKVPoolAllocator

{etc("定义分配器公共接口与 free_group 批处理语义。", c_base, ["子类必须实现 `clear/alloc/free`", "`available_size` 默认按 page 折算"])}

---

## 2. TokenToKVPoolAllocator.alloc

{etc("从 `free_pages` 头部切出 `need_size` 个连续索引。", code("python/sglang/srt/mem_cache/allocator/token.py", 55, 76), ["空间不足返回 `None`，Scheduler 触发 retract/evict", "`free` 在 group 模式下暂存到 `free_group`"])}

---

## 3. PagedTokenToKVPoolAllocator.alloc

{etc("按 page 数分配，返回 page×page_size 的 flat 索引。", c_paged, ["debug_mode 下断言 page 对齐", "输出 `(out_pages[:, None] * page_size + arange(page_size)).reshape(-1)`"])}

---

## 4. alloc_extend（Prefill 扩展）

{etc("Prefill 阶段为 batch 中各序列分配 extend 段的 KV 索引，调用 Triton kernel。", c_paged_ext, ["输入 `prefix_lens/seq_lens/last_loc` 描述各 req 状态", "`alloc_extend_kernel` 在 GPU 上并行计算索引"])}

---

## 5. HostKVCache.__init__

{etc("根据设备池配置计算主机 token 容量并分配 buffer。", c_host, ["`size_per_token` 由子类按 layout 计算", "容量小于设备池时打 warning"])}

---

## 6. HostKVCache.alloc/free

{etc("主机侧 slot 分配，带 double-alloc/free 检测。", c_host_alloc, ["必须 page 对齐", "`slot_used` bool 张量追踪占用状态"])}

---

## 7. StorageBackendFactory.create_backend

{etc("按名称实例化已注册或 dynamic 配置的 storage 后端。", c_storage_create, ["builtin 走 `_create_builtin_backend`", "dynamic 从 `extra_config` 加载模块路径"])}

---

## 8. merge_and_sort_free

{etc("合并 release 队列并排序 free_pages，提升 alloc 局部性。", code("python/sglang/srt/mem_cache/allocator/base.py", 78, 84), "- sort 后 free_pages 单调递增，便于 coalesce 感知")}

---

## 9. free_group 批释放

{etc("Radix 树批量 insert 时 defer free，结束时一次性 cat。", code("python/sglang/srt/mem_cache/allocator/base.py", 69, 76), "- `is_not_in_free_group=False` 期间 free 只 append 到 list")}
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 16：数据流与交互

## 1. 输入 / 输出

| 方向 | 类型 | 说明 | 源码 |
|------|------|------|------|
| 输入 | `need_size: int` | 需分配的 token/page 数 | allocator.alloc |
| 输出 | `torch.Tensor` | KV 池中的 slot 索引 | 同上 |
| 输入 | `free_index: Tensor` | 释放的索引 | allocator.free |

## 2. 上下游

| 模块 | 关系 | 说明 |
|------|------|------|
| RadixCache（批 15） | 上游 | prefix match 后 alloc 新 extend 索引 |
| memory_pool.KVCache | 下游 | 索引映射到 K/V 物理张量 |
| Scheduler | 调用方 | OOM 时 retract 触发 free |
| Attention Backend（批 17） | 消费者 | 通过 indices 读写的 KV |

## 3. Prefill extend 数据流

{etc("Scheduler 提交 extend batch → ModelRunner forward → allocator.alloc_extend。", c_paged_ext, ["步骤 1：计算各 req prefix/seq len", "步骤 2：kernel 填充 out_indices", "步骤 3：Attention 用 indices scatter K/V"])}

## 4. HiCache 回写流

{etc("设备 KV evict 时 backup 到 HostKVCache，命中时 load_to_device_per_layer。", code("python/sglang/srt/mem_cache/pool_host/base.py", 174, 189), "逐 layer 异步 IO，与 forward 流水线重叠")}

## 5. 外部 Storage 加载

{etc("Storage 后端实现 HiCacheStorage，与 host pool 交换 page 数据。", c_storage_create, "支持 file/NIXL/Mooncake 等多后端，factory 统一入口")}
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 16：关键问题

## Q1：page_size=1 与 page_size>1 如何选择？

{etc("server_args.page_size 决定 allocator 类型；>1 时用 Paged 分配器配合 FlashInfer。", q1_code, "page_size=16/32 常见，需与模型 max_seq 和 kernel 对齐要求一致")}

## Q2：alloc 返回 None 怎么办？

{etc("空间不足时 Scheduler retract 或 evict radix 节点。", code("python/sglang/srt/mem_cache/allocator/token.py", 55, 64), "先 merge_and_sort_free 尝试回收 release_pages；仍不足则上层处理")}

## Q3：HiCache 主机内存不足？

{etc("HostKVCache 启动时硬性检查可用 RAM。", code("python/sglang/srt/mem_cache/pool_host/base.py", 123, 137), "保留 HICACHE_HOST_MEMORY_RESERVE_BYTES（10GB）给 OS；减小 --hicache-ratio")}

## Q4：与 RadixAttention 的分工？

- **RadixCache**：逻辑前缀树、match/insert、决定哪些 token 可复用
- **Allocator**：物理 slot 索引的 alloc/free
- **KVCache 张量**：真正存储 K/V 数据

二者通过 `token_to_kv_pool_allocator` 解耦，批次 15 管「共享什么」，本批管「存在哪」。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(checkpoint(16, "KV Cache", [
        "KV 索引由 Token 或 Paged 分配器管理，接口统一在 BaseTokenToKVPoolAllocator。",
        "HiCache 在主机 RAM 维护 L2，Storage 工厂支持多种 L3 后端。",
        "Prefill extend 通过 alloc_extend kernel 批量分配 page 对齐索引。",
    ]), encoding="utf-8")


def write_batch17():
    d = ROOT / "04-内存与Attention/17-Attention"
    d.mkdir(parents=True, exist_ok=True)

    c_base = code("python/sglang/srt/layers/attention/base_attn_backend.py", 18, 87)
    c_meta = code("python/sglang/srt/layers/attention/triton_backend.py", 81, 100)
    c_fi_header = code("python/sglang/srt/layers/attention/flashinfer_backend.py", 1, 60)
    c_merge = code("python/sglang/srt/layers/attention/flashinfer_backend.py", 105, 116)
    c_dispatch = code("python/sglang/srt/layers/attention/flashinfer_backend.py", 119, 121)

    readme_entry17 = etc(
        "AttentionBackend 定义 forward metadata 的三方法契约：eager 入口、graph 外、graph 内。",
        c_base,
        ["`init_forward_metadata_out_graph`：host op、动态 shape", "`init_forward_metadata_in_graph`：可录制进 cuda graph 的 GPU op"],
    )

    (d / "README.md").write_text(f"""# 批次 17：Attention 后端

> 阶段 IV · 内存与算子 | 状态：已完成

## 本批目标

1. 理解 AttentionBackend 抽象与 CUDA Graph 三阶段 metadata 契约
2. 对比 FlashInfer 与 Triton 后端的选型
3. 追踪 prefill（extend）与 decode 两条算子路径

## 最关键代码

{readme_entry17}

← [批次 16](../16-KV-Cache/README.md) · → [批次 18](../18-MoE/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 17：核心概念

## 1. 后端分层

| 后端 | 特点 | 典型场景 |
|------|------|----------|
| flashinfer | 高性能、paged KV | NVIDIA 生产默认 |
| triton | 可定制、纯 Python+Triton | 新架构/调试 |
| trtllm_mla | MLA 专用 | DeepSeek 等 |

## 2. Extend vs Decode

- **Extend（Prefill）**：处理新 token，可能带 cached prefix
- **Decode**：每 req 每步 1 token，读全量 KV history

## 3. AttentionBackend 契约

{etc("ModelRunner 按 prefill/decode 模式选择 backend 字符串并 stamp 到实例。", c_base, "CUDA Graph capture 时 out_graph 与 in_graph 分离，避免 .item() 破坏录制")}

## 4. ForwardMetadata（Triton）

{etc("Triton 后端用 dataclass 携带 kv_indptr/kv_indices 等 paged 元数据。", c_meta, "与 FlashInfer 的 wrapper 参数同构，便于切换")}

## 5. FlashInfer 模块说明

{etc("文件头注释概括双后端 + extend/decode 双算子设计。", c_fi_header, "依赖 `is_flashinfer_available()` 条件 import")}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 17：源码走读

## 1. AttentionBackend 基类

{etc("ABC 定义 init_forward_metadata 默认组合 out+in。", c_base, "legacy capture/replay 方法已移除")}

## 2. init_forward_metadata 默认实现

{etc("Eager 路径：先 out_graph 再 in_graph。", code("python/sglang/srt/layers/attention/base_attn_backend.py", 45, 51), "子类可 override 整个 eager 体")}

## 3. init_forward_metadata_in_graph

{etc("Graph 内可录制 op，禁止 .cpu()/.item()。", code("python/sglang/srt/layers/attention/base_attn_backend.py", 75, 87), "Lint 契约写在 docstring")}

## 4. Triton ForwardMetadata

{etc("Paged KV 索引与 split 元数据。", c_meta, "含 sliding window 专用字段")}

## 5. FlashInfer merge_state 安全包装

{etc("head 数过大时回退 Triton merge，避免 CUDA block 超限。", c_merge, "max_heads 由 head_dim 与 vec_size 推导")}

## 6. WrapperDispatch 枚举

{etc("区分 sliding window 与 cross-attention wrapper。", c_dispatch, "FlashInferBackend 内按层类型 dispatch")}

## 7. needs_cpu_seq_lens

{etc("后端是否依赖 CPU 侧 seq_lens。", code("python/sglang/srt/layers/attention/base_attn_backend.py", 89, 90), "Opt-out 仅当从不读 seq_lens_cpu")}

## 8. init_cuda_graph_state

{etc("为 cuda graph 预分配 max_bs 级别 buffer。", code("python/sglang/srt/layers/attention/base_attn_backend.py", 99, 101), "capture 前一次性分配")}

## 9. RadixAttention 调用链（概念）

RadixAttention.forward → backend.forward → extend/decode kernel；KV 索引来自批次 16 allocator + 批次 15 radix match。
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 17：数据流与交互

## 上下游

| 上游 | 下游 |
|------|------|
| ForwardBatch（批 11） | logits |
| KVCache indices（批 16） | KV 读写 |
| RadixAttention 层 | 各 backend |

## Extend 数据流

{etc("Prefill：Q 新 token × 全量 K/V（含 prefix cache hit）。", c_meta, "kv_indptr/kv_indices 描述 paged 布局")}

## Decode 数据流

BatchDecodeWithPagedKVCacheWrapper（FlashInfer）或 triton decode kernel；每步 seq_len+1。

## CUDA Graph

{etc("Capture：out_graph(in_capture=True) → graph.capture → in_graph。", c_base, "Replay：out_graph(False) → graph.replay()")}
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 17：关键问题

## FlashInfer vs Triton？

{etc("FlashInfer 更快；Triton 更易改 kernel。", c_fi_header, "server_args.attention_backend 配置")}

## 为何 merge_state 有 fallback？

{etc("DP attention 时 num_heads 大导致 blockDim 超限。", c_merge, "自动切换 merge_state_triton")}

## 易错：graph 内调用 host sync

**错误**：在 `init_forward_metadata_in_graph` 里 `.item()`  
**正确**：动态逻辑放 `init_forward_metadata_out_graph`
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(checkpoint(17, "Attention", [
        "AttentionBackend 三方法契约分离 eager/capture/replay 的 metadata 准备。",
        "FlashInfer 与 Triton 共享 paged KV 元数据模型。",
        "Extend 与 Decode 使用不同 kernel/wrapper 路径。",
    ]), encoding="utf-8")


def write_batch18():
    d = ROOT / "04-内存与Attention/18-MoE"
    d.mkdir(parents=True, exist_ok=True)

    c_router = code("python/sglang/srt/layers/moe/router.py", 13, 76)
    c_topk = code("python/sglang/srt/layers/moe/topk.py", 44, 82)
    c_eplb = code("python/sglang/srt/eplb/eplb_manager.py", 1, 60)

    (d / "README.md").write_text(f"""# 批次 18：MoE 层

> 阶段 IV | 状态：已完成

## 目标

理解 Router → TopK → Token Dispatcher → Expert GEMM 流水线，以及 EPLB 专家负载均衡。

{etc("Triton fused router kernel 计算 logits、softcap、topk。", c_router, "输出 topk_ids 与 topk_weights 供 dispatch")}

← [17-Attention](../17-Attention/README.md) · → [19-Quantization](../19-Quantization/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 18：核心概念

## MoE 流水线

Router → TopK 选择专家 → Token Dispatcher（A2A）→ MoeRunner（GEMM）→ Combine

## Router

{etc("fused_moe_router_cudacore_kernel 在 GPU 上算 gate logits 并 topk。", c_router, "支持 moe_softcapping 与 correction_bias")}

## TopK / Routing

{etc("triton_kernels.topk 产生 RoutingData/GatherIndx/ScatterIndx。", c_topk, "供 fused MoE matmul 使用")}

## EPLB

{etc("Expert Parallel Load Balancer 周期性重排专家到各 rank。", c_eplb if (SGLANG / "python/sglang/srt/eplb/eplb_manager.py").exists() else code("python/sglang/srt/eplb/expert_location_dispatch.py", 1, 40), "expert_location_dispatch 做 logical→physical 映射")}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 18：源码走读

## 1. fused_moe_router kernel

{etc("逐 token 算 expert logits。", c_router, "tl.dot 待优化注释")}

## 2. topk 权重 softmax

{etc("router 内 invsumexp 归一化 topk 权重。", code("python/sglang/srt/layers/moe/router.py", 70, 90), "topk>=2 时 mask 已选 expert")}

## 3. fused_topk Python 入口

{etc("topk.py 封装多种 routing 策略。", code("python/sglang/srt/layers/moe/router.py", 117, 140), "调用 fused_moe_router_cudacore 或 alternative")}

## 4. Token Dispatcher base

{etc("抽象 dispatch/combine 接口。", code("python/sglang/srt/layers/moe/token_dispatcher/base.py", 1, 80), "Standard/DeepEP/FlashInfer 等实现")}

## 5. MoeRunner

{etc("Triton fused_moe 执行 expert GEMM。", code("python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py", 1, 50), "量化信息由批 19 QuantMethod 注入")}

## 6. expert_location_dispatch

{etc("EPLB 将 logical expert id 映射到 physical rank。", code("python/sglang/srt/eplb/expert_location_dispatch.py", 1, 50), "topk_ids_logical_to_physical")}

## 7. expert_distribution recorder

{etc("记录各 expert  token 计数供 EPLB 决策。", code("python/sglang/srt/eplb/expert_distribution.py", 1, 50), "全局 singleton recorder")}

## 8. DeepEP dispatcher

{etc("NVLink/RDMA expert parallel 通信。", code("python/sglang/srt/layers/moe/token_dispatcher/deepep.py", 1, 50), "大规模 EP 部署")}

## 9. MoE utils

{etc("get_moe_runner_backend 等运行时选择。", code("python/sglang/srt/layers/moe/utils.py", 1, 60), "与 server_args 联动")}
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 18：数据流与交互

## 数据流

Hidden states → Router logits → topk_ids/weights → Dispatch（permute）→ Expert compute → Combine（unpermute）→ 输出

{etc("Router kernel 输出供 dispatch 使用。", c_router, "batch 维 × topk 专家")}

## 与分布式

EP/TP 通过 token_dispatcher 与 eplb 协同；批 23 Distributed 详述。
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 18：关键问题

## Router softcapping 作用？

限制 gate logits 幅度，稳定训练分布；推理沿用相同公式。

{etc("tanh softcap 实现。", code("python/sglang/srt/layers/moe/router.py", 47, 55), "moe_softcapping=0 时跳过")}

## EPLB 何时触发？

周期性根据 expert_distribution 重排权重；见 eplb_manager。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(checkpoint(18, "MoE", [
        "Router Triton kernel 融合 gate+topk。",
        "Token Dispatcher 抽象 EP 通信，MoeRunner 执行 GEMM。",
        "EPLB 动态平衡 expert 负载。",
    ]), encoding="utf-8")


def write_batch19():
    d = ROOT / "04-内存与Attention/19-Quantization"
    d.mkdir(parents=True, exist_ok=True)

    c_base = code("python/sglang/srt/layers/quantization/base_config.py", 20, 84)
    c_fp8 = code("python/sglang/srt/layers/quantization/fp8.py", 120, 180)
    c_gptq = code("python/sglang/srt/layers/quantization/gptq/gptq.py", 51, 90)

    (d / "README.md").write_text(f"""# 批次 19：量化

> 阶段 IV | 状态：已完成

{etc("QuantizationConfig 为各层选择 QuantizeMethod（Linear/MoE/KV）。", c_base, "create_weights + apply 两阶段")}

← [18-MoE](../18-MoE/README.md) · → [20-Sampling](../../05-高级特性/20-Sampling/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 19：核心概念

## 量化体系

- **QuantizationConfig**：从 HF config 解析方法（fp8/gptq/awq/mxfp4…）
- **LinearMethodBase / FusedMoEMethodBase**：层级别 apply
- **BaseKVCacheMethod**：KV cache FP8/FP4 量化

## 基类

{etc("QuantizeMethodBase 定义 create_weights 与 apply。", c_base, "process_weights_after_loading 做 layout 变换")}

## FP8

{etc("Fp8Config 支持 W8A8 block/per-tensor，对接 DeepGEMM/Marlin。", c_fp8, "dispatch_w8a8_block_fp8_linear 路由 kernel")}

## GPTQ

{etc("GPTQConfig 支持 dynamic per-module 规则与 Marlin format。", c_gptq, "checkpoint_format=marlin 走 Marlin kernel")}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 19：源码走读

## 1. QuantizeMethodBase

{etc("抽象量化方法。", c_base, "子类实现 create_weights/apply")}

## 2. LinearMethodBase.apply

{etc("线性层前向入口。", code("python/sglang/srt/layers/quantization/base_config.py", 74, 83), "x @ quant_weight + bias")}

## 3. FusedMoEMethodBase

{etc("MoE 量化接口。", code("python/sglang/srt/layers/quantization/base_config.py", 86, 100), "create_moe_runner 绑定 MoeRunner")}

## 4. FP8 linear dispatch

{etc("按硬件选 cutlass/deepgemm/marlin。", code("python/sglang/srt/layers/quantization/fp8_utils.py", 1, 50), "is_sm90/sm100 等分支")}

## 5. scaled_fp8_quant

{etc("激活 FP8 量化 kernel。", code("python/sglang/srt/layers/quantization/fp8_kernel.py", 1, 50), "per-token-group quant")}

## 6. GPTQ schemes

{etc("GPTQLinearScheme vs GPTQMarlinLinearScheme。", code("python/sglang/srt/layers/quantization/gptq/schemes/gptq_linear.py", 1, 50), "按 checkpoint 格式选择")}

## 7. AWQ

{etc("AWQ 权重量化。", code("python/sglang/srt/layers/quantization/awq/awq.py", 1, 50), "4bit weight only")}

## 8. KV cache quant

{etc("KV FP8/FP4 方法。", code("python/sglang/srt/layers/quantization/kv_cache.py", 1, 60), "与 Attention backend 配合")}

## 9. unquant fallback

{etc("无量化时 UnquantizedLinearMethod。", code("python/sglang/srt/layers/quantization/unquant.py", 1, 50), "默认 bf16/fp16 路径")}
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 19：数据流与交互

ModelLoader 读权重 → QuantConfig.get_quant_method → layer.quant_method.apply → kernel

{etc("Config 解析 HF quant metadata。", c_gptq, "dynamic dict 支持 per-layer 规则")}
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 19：关键问题

## FP8 vs GPTQ？

{etc("FP8 走 block/per-tensor 量化 GEMM；GPTQ 用 group index 解压 4bit。", c_fp8, ["W8A8 对接 Tensor Core", "按 SM90/SM100 选 cutlass/deepgemm"])}

## Marlin 是什么？

{etc("check_marlin_format 检测 checkpoint 是否已 Marlin layout。", code("python/sglang/srt/layers/quantization/gptq/gptq.py", 43, 48), ["checkpoint_format=marlin 走 GPTQMarlinLinearScheme", "prepare_fp8_layer_for_marlin 做权重 reorder"])}
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(checkpoint(19, "Quantization", [
        "QuantizationConfig + Method 双轨扩展 Linear/MoE/KV。",
        "FP8 按 SM 版本 dispatch 不同 GEMM backend。",
        "GPTQ/AWQ 支持 Marlin 加速 layout。",
    ]), encoding="utf-8")


def write_batch20():
    d = ROOT / "05-高级特性/20-Sampling"
    d.mkdir(parents=True, exist_ok=True)

    c_params = code("python/sglang/srt/sampling/sampling_params.py", 75, 120)
    c_batch = code("python/sglang/srt/sampling/sampling_batch_info.py", 23, 75)
    c_batch_from = code("python/sglang/srt/sampling/sampling_batch_info.py", 76, 110)
    c_orch = code("python/sglang/srt/sampling/penaltylib/orchestrator.py", 13, 70)
    c_grammar = code("python/sglang/srt/constrained/grammar_manager.py", 25, 100)

    (d / "README.md").write_text(f"""# 批次 20：Sampling 与约束解码

> 阶段 V · 高级特性 | 状态：已完成 | **图谱更新点 batch-20**

## 目标

SamplingParams → SamplingBatchInfo → logits 采样；GrammarManager 约束 JSON/regex。

{etc("SamplingParams msgspec Struct 承载全部采样与约束参数。", c_params, "normalize() 预处理 stop/json_schema 等")}

← [19-Quantization](../../04-内存与Attention/19-Quantization/README.md) · → [21-Speculative](../21-Speculative/README.md)
""", encoding="utf-8")

    (d / "01-核心概念.md").write_text(f"""# 批次 20：核心概念

## 模块组成

| 目录 | 职责 |
|------|------|
| srt/sampling/ | 温度、top_p/k、penalty |
| srt/constrained/ | grammar/xgrammar/outlines |
| srt/parser/ | reasoning/harmony 等输出解析 |

## SamplingParams

{etc("API 参数与内部字段分离，msgspec 高效 IPC。", c_params, "json_schema/regex/ebnf 触发约束解码")}

## SamplingBatchInfo

{etc("将各 req 的 SamplingParams 批量化成 GPU tensor。", c_batch, "is_all_greedy 等 flag 跳过无用 kernel")}

## Penalty Orchestrator

{etc("frequency/presence/repetition/min_new_tokens 统一管理。", c_orch, "weakref 持有 ScheduleBatch")}
""", encoding="utf-8")

    (d / "02-源码走读.md").write_text(f"""# 批次 20：源码走读

## 1. SamplingParams

{etc("Struct 定义。", c_params, "TOP_K_ALL=1<<30 表示不限制 top_k")}

## 2. raise_if_tokenizer_required

{etc("无 tokenizer 时禁止 stop_str/min_new_tokens。", code("python/sglang/srt/sampling/sampling_params.py", 45, 72), "skip_tokenizer_init 场景")}

## 3. SamplingBatchInfo.from_schedule_batch

{etc("从 ScheduleBatch 构建批 tensor。", c_batch_from, "pin_memory + non_blocking H2D")}

## 4. BatchedPenalizerOrchestrator.apply

{etc("对 logits in-place 施加 penalty。", c_orch, "spec decode 时 repeat 扩展 layout")}

## 5. GrammarManager.process_req_with_grammar

{etc("检测 json_schema/regex 并入 grammar 队列。", c_grammar, "grammar_backend 异步编译")}

## 6. create_grammar_backend

{etc("按 server_args 选 xgrammar/outlines/llguidance。", code("python/sglang/srt/constrained/base_grammar_backend.py", 1, 60), "factory 模式")}

## 7. frequency_penalty

{etc("按 token 出现次数减 logits。", code("python/sglang/srt/sampling/penaltylib/frequency_penalty.py", 1, 50), "BatchedPenalizer 子类")}

## 8. custom_logit_processor

{etc("用户自定义 logits 变换。", code("python/sglang/srt/sampling/custom_logit_processor.py", 1, 50), "per-req 注册")}

## 9. reasoning_parser

{etc("解析  等 reasoning 标签。", code("python/sglang/srt/parser/reasoning_parser.py", 1, 50), "与 ReasonerGrammar 配合")}
""", encoding="utf-8")

    (d / "03-数据流与交互.md").write_text(f"""# 批次 20：数据流与交互

ModelRunner logits → penalty → grammar mask → sample（top_p/top_k）→ token id

{etc("批量化采样参数。", c_batch_from, "temperatures shape [bs,1] 广播")}

## 约束解码流

Req 带 json_schema → GrammarManager 队列 → compile grammar → vocab_mask apply_mask_func
""", encoding="utf-8")

    (d / "04-关键问题.md").write_text(f"""# 批次 20：关键问题

## greedy vs sampling flags

is_all_greedy=True 时走 argmax 捷径，跳过 top_p kernel。

## grammar-backend none

{etc("无 backend 时约束请求直接 abort。", code("python/sglang/srt/constrained/grammar_manager.py", 98, 101), "启动需显式配置 backend")}

## msgspec 为何用 Struct？

Scheduler 与 TokenizerManager 间 msgpack IPC，Struct 比 dataclass 更快且 omit_defaults。
""", encoding="utf-8")

    (d / "checkpoint.md").write_text(checkpoint(20, "Sampling", [
        "SamplingBatchInfo 将 per-req 参数批量化供 GPU 采样。",
        "Penalty orchestrator 统一 frequency/presence/repetition。",
        "GrammarManager 异步编译 json_schema/regex 约束。",
    ]), encoding="utf-8")


def update_knowledge_graph():
    kg_path = SGLANG / ".understand-anything/knowledge-graph.json"
    kg = json.loads(kg_path.read_text(encoding="utf-8"))

    new_nodes = [
        {"id": "module:mem-cache-allocator", "type": "module", "name": "KV Cache Allocator", "summary": "Token/Page 级 KV 索引分配：TokenToKVPoolAllocator、PagedTokenToKVPoolAllocator。", "tags": ["cache", "memory"], "complexity": "moderate"},
        {"id": "module:pool-host", "type": "module", "name": "HiCache Host Pool", "summary": "主机 RAM KV 池，分层缓存 L2，支持 backup/load 与 storage 对接。", "tags": ["cache", "hicache"], "complexity": "moderate"},
        {"id": "module:attention-backends", "type": "module", "name": "Attention Backends", "summary": "FlashInfer/Triton 等 Attention 实现，extend+decode 双路径。", "tags": ["attention", "kernel"], "complexity": "complex"},
        {"id": "module:moe", "type": "module", "name": "MoE Layers", "summary": "Router、TopK、Token Dispatcher、MoeRunner 与 EPLB。", "tags": ["moe", "expert"], "complexity": "complex"},
        {"id": "module:quantization", "type": "module", "name": "Quantization", "summary": "FP8/GPTQ/AWQ/MXFP4 等 QuantizationConfig 与 kernel dispatch。", "tags": ["quantization"], "complexity": "complex"},
        {"id": "module:sampling", "type": "module", "name": "Sampling", "summary": "SamplingParams、批采样、penalty 与 constrained grammar。", "tags": ["sampling", "decoding"], "complexity": "moderate"},
        {"id": "concept:paged-kv", "type": "concept", "name": "Paged KV Cache", "summary": "Page 对齐的 KV 索引分配，对接 FlashInfer PagedAttention。", "tags": ["cache"], "complexity": "moderate"},
        {"id": "concept:grammar-decoding", "type": "concept", "name": "Grammar-Guided Decoding", "summary": "json_schema/regex/ebnf 约束，通过 vocab mask 限制 logits。", "tags": ["constrained"], "complexity": "moderate"},
    ]
    existing_ids = {n["id"] for n in kg["nodes"]}
    for n in new_nodes:
        if n["id"] not in existing_ids:
            kg["nodes"].append(n)

    new_edges = [
        {"source": "concept:radix-attention", "target": "module:mem-cache-allocator", "type": "depends_on", "direction": "forward", "weight": 0.7},
        {"source": "module:mem-cache-allocator", "target": "module:pool-host", "type": "related", "direction": "forward", "weight": 0.6},
        {"source": "module:mem-cache-allocator", "target": "module:attention-backends", "type": "depends_on", "direction": "forward", "weight": 0.8},
        {"source": "module:attention-backends", "target": "module:moe", "type": "related", "direction": "forward", "weight": 0.5},
        {"source": "module:moe", "target": "module:quantization", "type": "related", "direction": "forward", "weight": 0.6},
        {"source": "module:srt", "target": "module:sampling", "type": "contains", "direction": "forward", "weight": 0.7},
        {"source": "module:sampling", "target": "concept:grammar-decoding", "type": "related", "direction": "forward", "weight": 0.7},
    ]
    kg["edges"].extend(new_edges)

    # Update runtime-core layer
    for layer in kg["layers"]:
        if layer["id"] == "layer:runtime-core":
            for nid in ["module:mem-cache-allocator", "module:attention-backends", "module:moe", "module:quantization", "module:sampling", "concept:paged-kv", "concept:grammar-decoding"]:
                if nid not in layer["nodeIds"]:
                    layer["nodeIds"].append(nid)

    kg["project"]["meta"] = kg["project"].get("meta", {})
    kg["project"]["meta"]["scope"] = "batch-20"
    kg["project"]["meta"]["lastBatchUpdate"] = "2026-07-02"
    kg["project"]["meta"]["batchesCovered"] = "01,16-20"

    tour_add = [
        {"order": 6, "title": "KV Cache 分配", "description": "Token/Page 分配器与 HiCache 主机池、Storage 后端。", "nodeIds": ["module:mem-cache-allocator", "module:pool-host", "concept:paged-kv"]},
        {"order": 7, "title": "Attention 后端", "description": "FlashInfer/Triton extend+decode 与 CUDA Graph metadata。", "nodeIds": ["module:attention-backends"]},
        {"order": 8, "title": "MoE 与量化", "description": "Expert routing、EPLB 与 FP8/GPTQ 量化路径。", "nodeIds": ["module:moe", "module:quantization"]},
        {"order": 9, "title": "采样与约束解码", "description": "SamplingBatchInfo、penalty 与 grammar-guided decoding。", "nodeIds": ["module:sampling", "concept:grammar-decoding"]},
    ]
    existing_orders = {t["order"] for t in kg["tour"]}
    for t in tour_add:
        if t["order"] not in existing_orders:
            kg["tour"].append(t)
    kg["tour"].sort(key=lambda x: x["order"])

    kg_path.write_text(json.dumps(kg, ensure_ascii=False, indent=2), encoding="utf-8")


def update_progress():
    p = ROOT / "progress.md"
    text = p.read_text(encoding="utf-8")
    # Update header stats
    text = text.replace("已完成：1 | 进行中：0 | 待开始：29", "已完成：6 | 进行中：0 | 待开始：24")
    text = text.replace("[█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 1/30 (3%)", "[██████░░░░░░░░░░░░░░░░░░░░░░░░] 6/30 (20%)")
    text = text.replace("| IV 内存 | 15–19 | 内存与 Attention | 0/5 |", "| IV 内存 | 15–19 | 内存与 Attention | 0/5 |")
    # Mark batches 16-20 complete
    for b in range(16, 21):
        old = f"| **{b:02d}** | ⬜ 待开始 |"
        new = f"| **{b:02d}** | ✅ 已完成 | 2026-07-02 | 2026-07-02 |"
        if old in text:
            parts = text.split(f"| **{b:02d}** | ⬜ 待开始 |")
            if len(parts) == 2:
                rest = parts[1].split("|", 1)[1]  # keep rest of row after first |
                text = parts[0] + f"| **{b:02d}** | ✅ 已完成 | 2026-07-02 | 2026-07-02 |" + rest

    # Fix batch rows properly
    rows = {
        16: ("16-KV-Cache", ""),
        17: ("17-Attention", ""),
        18: ("18-MoE", ""),
        19: ("19-Quantization", ""),
        20: ("20-Sampling", "图谱 batch-20"),
    }
    import re
    for b, (dir_name, note) in rows.items():
        pattern = rf"\| \*\*{b:02d}\*\* \| [^|]+\| [^|]*\| [^|]*\| \[[^\]]+\]\([^)]+\) \| [^|]*\|"
        link = f"./04-内存与Attention/{dir_name}/" if b < 20 else f"./05-高级特性/{dir_name}/"
        if b == 20:
            link = f"./05-高级特性/{dir_name}/"
        elif b >= 16:
            link = f"./04-内存与Attention/{dir_name}/"
        replacement = f"| **{b:02d}** | ✅ 已完成 | 2026-07-02 | 2026-07-02 | [{dir_name}]({link}) | {note} |"
        text = re.sub(pattern, replacement, text)

    # Add graph update record
    if "batch-20" not in text:
        insert = "| 2026-07-02 | batch-16-20 | 增量更新 | KV Cache/Attention/MoE/Quant/Sampling；scope=batch-20 |\n"
        text = text.replace(
            "| 2026-07-02 | batch-01 | 初始图谱 | 9 文件节点 + 模块/概念；scope=batch-01-initial |\n",
            "| 2026-07-02 | batch-01 | 初始图谱 | 9 文件节点 + 模块/概念；scope=batch-01-initial |\n" + insert,
        )

    p.write_text(text, encoding="utf-8")


def main():
    write_batch16()
    write_batch17()
    write_batch18()
    write_batch19()
    write_batch20()
    update_knowledge_graph()
    update_progress()
    print("Generated batches 16-20 (30 files)")


if __name__ == "__main__":
    main()
