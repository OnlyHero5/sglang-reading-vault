---
type: batch-doc
module: 16-KV-Cache
batch: "16"
doc_type: checkpoint
title: "KV Cache 验收清单"
tags:
 - sglang/batch/16
 - sglang/module/kv-cache
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# KV Cache 验收清单

## 读者自测（不打开 sglang/）

- [x] 能解释 Token 级与 Page 级 KV 索引分配器的区别与选型
- [x] 能画出 prefill extend / decode 两条 alloc 路径
- [x] 能说出 BaseTokenToKVPoolAllocator、PagedTokenToKVPoolAllocator、HostKVCache 职责
- [x] 能说明 alloc 返回 None 时 Scheduler 如何处理
- [x] 能解释 HiCache L2/L3 分层与 Storage 工厂角色
- [x] 五篇正文 ≥ 15 段内嵌源码

## 维护者检查

- [x] 覆盖 `allocator/base.py`、`token.py`、`paged.py`、`pool_host/base.py`、`storage/backend_factory.py`
- [x] 行号对齐 git `70df09b`（2026-07-02）
- [ ] [[progress]] 由 P8 更新

## 核心结论（3 句话）

1. **KV 索引由 Token 或 Paged 分配器管理，接口统一在 BaseTokenToKVPoolAllocator**，RadixCache 通过 alloc/free 获取/归还 slot，不直接操作物理 KV 张量。
2. **HiCache 在主机 RAM 维护 L2，Storage 工厂支持 Mooncake/NIXL 等多种 L3 后端**，逐 layer IO 与 forward 流水线重叠。
3. **Prefill extend 通过 alloc_extend kernel 批量分配 page 对齐索引；decode 通过 alloc_decode 每 req 单 token 分配**，OOM 时返回 None 触发 retract/evict。

## 遗留问题

- `_match_prefix_helper` / `_insert_helper` 内部 split 算法 → 可与RadixAttention RadixCache 联读
- `HybridCacheController` prefetch 状态机 → HiCache 专批
- Mooncake/NIXL storage 后端部署细节 → 运维文档

## 内嵌源码统计（维护者）

| 文档 | ETC 段数（约） |
|------|----------------|
| README.md | 2 |
| 01-核心概念.md | 5 |
| 02-源码走读.md | 11 |
| 03-数据流与交互.md | 6 |
| 04-关键问题.md | 6 |
| **合计** | **30 段** |

合计内嵌源码行数：**约 280+ 行**

## 建议补充 KG 节点

- `BaseTokenToKVPoolAllocator` / `TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator`
- `alloc_extend` / `alloc_decode` / `alloc_extend_kernel`
- `HostKVCache` / `StorageBackendFactory`
- `free_group_begin` / `merge_and_sort_free`
