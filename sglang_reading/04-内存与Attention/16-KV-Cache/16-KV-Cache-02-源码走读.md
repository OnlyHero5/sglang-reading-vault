---
type: batch-doc
module: 16-KV-Cache
batch: "16"
doc_type: walkthrough
title: "KV Cache · 源码走读"
tags:
 - sglang/batch/16
 - sglang/module/kv-cache
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# KV Cache · 源码走读

## 走读顺序

1. `allocator/base.py` — 抽象基类
2. `allocator/token.py` — Token 级实现
3. `allocator/paged.py` — Page 级实现
4. `pool_host/base.py` — HiCache 主机池
5. `storage/backend_factory.py` — 外部存储工厂

---

## 1. BaseTokenToKVPoolAllocator

**Explain：** 所有 KV 索引分配器的抽象基类，RadixCache（RadixAttention）与 Scheduler 只依赖 `alloc`/`free`/`available_size` 三接口，不感知底层是 token 还是 page 粒度。`free_group_begin/end` 支持 Radix 树批量 insert 时 defer free，避免中间态频繁 sort；`merge_and_sort_free` 将 release 队列合并回 free 列表并排序，提升 alloc 局部性。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/base.py L27-L110
class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
```

**Comment：**
- 子类必须实现 `clear/alloc/free`
- `available_size` 默认按 page 折算


---

## 2. TokenToKVPoolAllocator.alloc

**Explain：** Token 级分配器以 `page_size=1` 运行，`free_pages` 实际存储 token slot 索引（slot 0 保留给 padding dummy）。空间不足返回 `None` 触发 Scheduler retract/evict；`need_sort=True` 时释放的索引先进 `release_pages`，alloc 前 merge 排序以减少碎片。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/token.py L55-L76
    def alloc(self, need_size: int):
        if self.need_sort and need_size > len(self.free_pages):
            self.merge_and_sort_free()

        if need_size > len(self.free_pages):
            return None

        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            if self.need_sort:
                self.release_pages = torch.cat((self.release_pages, free_index))
            else:
                self.free_pages = torch.cat((self.free_pages, free_index))
        else:
            self.free_group.append(free_index)
```

**Comment：**
- 空间不足返回 `None`，Scheduler 触发 retract/evict
- `free` 在 group 模式下暂存到 `free_group`


---

## 3. PagedTokenToKVPoolAllocator.alloc

**Explain：** Page 级分配器与 `--page-size` 及 FlashInfer PagedAttention 对齐；`alloc` 按 page 数切分 free 列表，输出 `(page_id * page_size + arange(page_size))` 展开的 token 索引。ROCm init 时预热 `torch.unique` 避免 radix 重复 prefix 场景下首请求 JIT 延迟 200ms+。

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
- debug_mode 下断言 page 对齐
- 输出 `(out_pages[:, None] * page_size + arange(page_size)).reshape(-1)`


---

## 4. alloc_extend（Prefill 扩展）

**Explain：** Prefill extend 阶段 Scheduler 提交 batch 后，ModelRunner 调用此函数为每个 req 的新 token 分配 KV 索引。Triton `alloc_extend_kernel` 在 GPU 上并行计算各 req 的 prefix/seq 边界，从 free_pages 分配新 page 并填充 out_indices；debug_mode 断言 last_loc 与 prefix_lens 的 page 对齐关系。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/paged.py L172-L215
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
```

**Comment：**
- 输入 `prefix_lens/seq_lens/last_loc` 描述各 req 状态
- `alloc_extend_kernel` 在 GPU 上并行计算索引


---

## 5. HostKVCache.__init__

**Explain：** 根据设备池配置计算主机 token 容量并分配 buffer。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/pool_host/base.py L79-L143
class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)
        self.can_use_write_back_jit = False

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = sync_fixed_hicache_size(
                int(host_size * 1e9 // self.size_per_token), host_size
            )
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        if self.size <= device_pool.size:
            logger.warning(
                "HiCache host KV pool (%d tokens) is smaller than the device pool (%d tokens);"
                "L2 cache effectiveness is reduced."
                "Consider increasing --hicache-ratio (or --hicache-size) for higher L2 cache hit rate.",
                self.size,
                device_pool.size,
            )

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

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()
```

**Comment：**
- `size_per_token` 由子类按 layout 计算
- 容量小于设备池时打 warning


---

## 6. HostKVCache.alloc/free

**Explain：** HiCache L2 主机池的 slot 分配接口，与设备侧 allocator 语义类似但运行在 CPU 内存。`@synchronized` 保证多线程 prefetch/backup 并发安全；`slot_used` bool 张量检测 double-alloc/free，违反时 assert 报错而非 silent corruption。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/pool_host/base.py L240-L268
    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        assert not self.slot_used[select_index].any(), (
            f"Double-alloc detected: slots already allocated: "
            f"{select_index[self.slot_used[select_index]].tolist()}."
        )
        self.slot_used[select_index] = True

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices_cpu = indices.cpu()
        assert self.slot_used[indices_cpu].all(), (
            f"Double-free detected: slots not currently allocated: "
            f"{indices_cpu[~self.slot_used[indices_cpu]].tolist()}."
        )
        self.slot_used[indices_cpu] = False
        self.free_slots = torch.cat([self.free_slots, indices_cpu])
        return len(indices)
```

**Comment：**
- 必须 page 对齐
- `slot_used` bool 张量追踪占用状态


---

## 7. StorageBackendFactory.create_backend

**Explain：** 按名称实例化已注册或 dynamic 配置的 storage 后端。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/storage/backend_factory.py L66-L96
    def create_backend(
        cls,
        backend_name: str,
        storage_config: HiCacheStorageConfig,
        mem_pool_host: Any,
        **kwargs,
    ) -> HiCacheStorage:
        """Create a storage backend instance.
        Args:
            backend_name: Name of the backend to create
            storage_config: Storage configuration
            mem_pool_host: Memory pool host object
            **kwargs: Additional arguments passed to external backends
        Returns:
            Initialized storage backend instance
        Raises:
            ValueError: If backend is not registered and cannot be dynamically loaded
            ImportError: If backend module cannot be imported
            Exception: If backend initialization fails
        """
        # First check if backend is already registered
        if backend_name in cls._registry:
            registry_entry = cls._registry[backend_name]
            backend_class = registry_entry["loader"]()
            logger.info(
                f"Creating storage backend '{backend_name}' "
                f"({registry_entry['module_path']}.{registry_entry['class_name']})"
            )
            return cls._create_builtin_backend(
                backend_name, backend_class, storage_config, mem_pool_host
            )
```

**Comment：**
- builtin 走 `_create_builtin_backend`
- dynamic 从 `extra_config` 加载模块路径


---

## 8. merge_and_sort_free

**Explain：** 合并 release 队列并排序 free_pages，提升 alloc 局部性。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/base.py L78-L84
    def merge_and_sort_free(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )
```

**Comment：**
- sort 后 free_pages 单调递增，便于 coalesce 感知


---

## 9. free_group 批释放

**Explain：** Radix 树批量 insert 时 defer free，结束时一次性 cat。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/base.py L69-L76
    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))
```

---

## 10. alloc_decode（Decode 单 token）

**Explain：** Decode 阶段每 req 只增 1 个 token，调用 `alloc_decode_kernel` 为 batch 中各 req 分配单个 KV 索引。与 extend 不同，输入只需 seq_lens/last_loc；同样检查 page 边界对齐，OOM 时返回 None 触发 retract。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/allocator/paged.py L222-L259
    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
```

**Comment：**
- decode 输出 shape `[bs]`，每 req 一个 index
- free 时用 `torch.unique(free_index // page_size)` 回收 page
