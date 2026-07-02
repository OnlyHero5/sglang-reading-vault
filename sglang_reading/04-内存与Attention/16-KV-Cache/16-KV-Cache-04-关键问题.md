---
type: batch-doc
module: 16-KV-Cache
batch: "16"
doc_type: faq
title: "KV Cache：关键问题"
tags:
 - sglang/batch/16
 - sglang/module/kv-cache
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# KV Cache：关键问题

## Q1：page_size=1 与 page_size>1 如何选择？

**Explain：** server_args.page_size 决定 allocator 类型；>1 时用 Paged 分配器配合 FlashInfer。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/paged.py L105-L170
class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")

        # Pre-warm the torch.unique HIP kernel used in free(). When a request
        # finishes with a prompt that already exists in the radix tree (e.g.
        # bench_serving sending the same warmup+measured prompt), the radix
        # cache's _insert_helper frees the duplicate KV indices via
        # token_to_kv_pool_allocator.free(value[start:prefix_len]). That call
        # path runs `torch.unique(free_index // self.page_size)` on a
        # ~prompt_len-sized int64 tensor. The first such call on AMD ROCm
        # JIT-compiles rocPRIM sort/unique kernels and costs ~200ms, which
        # shows up as a mysterious "second-request slow" (Run 1) for
        # repeated-prompt benchmarks. Running it once at init time moves
        # that JIT cost to startup. This is a ROCm-only JIT cost, so the
        # warm-up is gated on _is_hip and skipped on other platforms.
        if _is_hip and torch.cuda.is_available():
            try:
                _warmup = torch.arange(1024, dtype=torch.int64, device=device)
                _ = torch.unique(_warmup // page_size)
                torch.cuda.synchronize()
            except Exception:
                pass
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices
```

**Comment：**
page_size=16/32 常见，需与模型 max_seq 和 kernel 对齐要求一致


## Q2：alloc 返回 None 怎么办？

**Explain：** 空间不足时 Scheduler retract 或 evict radix 节点。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/token.py L55-L64
    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index
```

**Comment：**
先 merge_and_sort_free 尝试回收 release_pages；仍不足则上层处理


## Q3：HiCache 主机内存不足？

**Explain：** HostKVCache 启动时硬性检查可用 RAM。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/pool_host/base.py L123-L137
        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )
```

**Comment：**
保留 HICACHE_HOST_MEMORY_RESERVE_BYTES（10GB）给 OS；减小 --hicache-ratio


## Q4：与 RadixAttention 的分工？

- **RadixCache**：逻辑前缀树、match/insert、决定哪些 token 可复用
- **Allocator**：物理 slot 索引的 alloc/free
- **KVCache 张量**：真正存储 K/V 数据

二者通过 `token_to_kv_pool_allocator` 解耦，RadixAttention 管「共享什么」，本模块管「存在哪」。

---

## Q5：Paged free 如何回收 page？

**Explain：** Paged 分配器 free 时先将 token index 除以 page_size 做 `torch.unique` 得到 page index，再合并到 free_pages 或 release_pages。非 group 模式且 need_sort=True 时进 release_pages 延迟排序；group 模式暂存 free_group 待 end 时 cat 一次性 free。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/paged.py L261-L272
    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
        else:
            self.free_group.append(free_index)
```

**易错对比：**

```python
# ❌ Token 级 allocator 直接 free token index——无需 unique
# ✅ Paged allocator 必须 unique page index，否则同一 page 部分 token 释放会导致 double-free
```

---

## 验证建议（零基础可试）

1. **操作：** 启动 `--page-size 16`（或默认 paged 配置），发送超长 prompt 使 `--max-running-requests` 接近上限，观察是否出现 retract 或 evict 日志而非进程崩溃。 
 **预期现象：** `alloc` 返回 None 时 Scheduler retract/evict，服务存活；无 CUDA illegal memory access。 
 **对应文档节：** [[16-KV-Cache-01-核心概念|01-核心概念 § 用户故事]]、Q2 alloc 返回 None

2. **操作：** 同一 warmup prompt 连续发两条相同请求（ROCm 平台尤其明显），对比 Run1 vs Run2 第二条延迟。 
 **预期现象：** 若未预热，Run1 在 radix insert free 路径可能多 ~200ms（`torch.unique` JIT）；Run2 正常。NVIDIA 上差异较小。 
 **对应文档节：** §3 Page 对齐分配、paged.py init 预热注释

3. **操作：** 尝试 `--hicache-ratio 2.0` 在内存紧张机器上启动（或故意设极大 `--hicache-size`）。 
 **预期现象：** 启动时报 `ValueError: Not enough host memory available`；减小 ratio 后正常启动。 
 **对应文档节：** §4 HiCache 主机池、Q3 主机内存不足
