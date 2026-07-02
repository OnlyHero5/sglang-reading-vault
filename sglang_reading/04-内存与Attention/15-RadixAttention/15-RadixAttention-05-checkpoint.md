---
type: batch-doc
module: 15-RadixAttention
batch: "15"
doc_type: checkpoint
title: "RadixAttention 验收清单"
tags:
 - sglang/batch/15
 - sglang/module/radix-attention
 - sglang/doc/checkpoint
aliases:
 - "checkpoint"
updated: 2026-07-02
---
# RadixAttention 验收清单

## 读者自测（不打开 sglang/）

- [x] 能解释 RadixCache（树）与 RadixAttention（算子）的分工
- [x] 能画出 prefill：match_prefix → extend forward → cache_unfinished/finished
- [x] 能说出 3 个核心 API：`match_prefix`、`insert`、`RadixAttention.forward`
- [x] 能说明 extra_key、page_size、lock_ref 的作用
- [x] 能对比 RadixCache 与 UnifiedRadixCache 扩展点
- [x] 五篇正文 ≥ 15 段内嵌源码

## 维护者检查

- [x] 覆盖 `radix_cache.py`、`unified_radix_cache.py`、`radix_attention.py` 主路径
- [x] 行号对齐 git `70df09b`（2026-07-02）
- [ ] [[progress]] 由 P8 更新

## 核心结论（3 句话）

1. **RadixCache 用 RadixKey+TreeNode 管理 token 前缀到 KV pool indices 的映射**，`match_prefix`/`insert`/lock/evict 构成请求生命周期管理。
2. **RadixAttention 是模型内 Attention 统一入口**，通过 `get_attn_backend()` 或 piecewise `unified_attention_with_output` 读写物理 KV，不直接操作 radix 树。
3. **UnifiedRadixCache 在相同树结构上叠加多 component、HiCache、StreamingSession**，API 兼容经典 RadixCache 并扩展 evict/lock 语义。

## 遗留问题

- `_match_prefix_helper` / `_insert_helper` 内部 split 算法 → 可与KV Cache KV pool 联读
- `HybridCacheController` prefetch 状态机 → HiCache 专批
- Attention backend 注册表 → `attention/` 目录

## 内嵌源码统计（维护者）

| 文档 | ETC 段数（约） |
|------|----------------|
| README.md | 2 |
| 01-核心概念.md | 10 |
| 02-源码走读.md | 13 |
| 03-数据流与交互.md | 10 |
| 04-关键问题.md | 12 |
| **合计** | **47 段** |

合计内嵌源码行数：**约 320+ 行**

## 建议补充 KG 节点

- `RadixKey` / `TreeNode` / `RadixCache`
- `UnifiedTreeNode` / `UnifiedRadixCache` / `ComponentType`
- `match_prefix` / `cache_unfinished_req` / `cache_finished_req`
- `RadixAttention` / `unified_attention_with_output`
- `StreamingSession` / `HybridCacheController`（边：extends UnifiedRadixCache）
