---
title: "KV-Cache · 核心概念"
type: concept
framework: sglang
topic: "KV-Cache"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# KV-Cache · 核心概念

## 读者任务

这一篇先建立心理模型：SGLang 的 KV Cache 不是单个大 tensor，而是三层对象协作：

1. 请求行：`ReqToTokenPool` 给每个活跃请求一行。
2. KV slot/page：allocator 负责发号、回收、延迟整理。
3. 物理张量：`KVCache` 按 layer 存 K/V，attention backend 在指定位置写入和读取。

只要分不清这三层，`prefix_indices`、`out_cache_loc`、`req_pool_idx`、`available_size()` 会混成一团。

## 心理模型：图书馆借阅系统

把 KV Cache 想成图书馆：

- `ReqToTokenPool` 是借阅登记表：每个请求一行，每列记录这个 token 借到了哪个书架位置。
- allocator 是空位管理员：知道哪些 slot/page 可用，什么时候把释放的空位重新排序。
- `KVCache` 是书架本身：每层 K/V tensor 按 slot 存放内容。
- RadixCache 是目录：知道哪些前缀已经存过，可以让新请求复用旧书架位置。
- HiCache 是楼下库房或外部仓库：把冷 KV 从 GPU 书架搬到主机内存或存储后端。

## 第一层：请求行不是 KV 内容

`ReqToTokenPool` 不存 K/V 向量，它存“请求 token → KV slot”的映射。slot 0 保留给 padding，真实请求从 1 开始拿行号。

```python
# 来源：python/sglang/srt/mem_cache/memory_pool.py L242-L301
class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""
    ...
    self.req_to_token = torch.zeros(
        (self._alloc_size, max_context_len), dtype=torch.int32, device=device
    )
    self.free_slots = list(range(1, self._alloc_size))
    ...
    def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
        ...
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        ...
        return [r.req_pool_idx for r in reqs]
```

这解释了两个常见现象：

- `max_running_requests` 不只影响调度，也影响请求行容量。
- chunked prefill 可能复用已有 `req_pool_idx`，因为同一个请求跨 chunk 仍然写同一行。

## 第二层：allocator 只管理号，不理解语义

`BaseTokenToKVPoolAllocator` 统一了 token allocator、paged allocator、SWA allocator、NPU allocator 等实现的接口。它关心的是可用 slot/page、释放队列和批量释放边界，不关心这个 token 是 prompt、decode、prefix hit 还是 user output。

```python
# 来源：python/sglang/srt/mem_cache/allocator/base.py L27-L110
class BaseTokenToKVPoolAllocator(abc.ABC):
    ...
    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

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
```

这里有三个不变量：

- `available_size()` 是调度容量判断的输入，但不等于“GPU 空闲字节数”。
- `release_pages` 是延迟释放区，必要时才并回 `free_pages`。
- `free_group_begin/end` 给批量释放一个事务边界，避免 Radix 或结果处理频繁排序。

## 第三层：token 粒度和 page 粒度

token allocator 以单个 slot 为最小单位。它简单、直观，但不满足 paged attention 对 page 边界的要求。

```python
# 来源：python/sglang/srt/mem_cache/allocator/token.py L42-L76
def clear(self):
    self.free_pages = torch.arange(
        1, self.size + 1, dtype=torch.int64, device=self.device
    )
    ...

def alloc(self, need_size: int):
    if self.need_sort and need_size > len(self.free_pages):
        self.merge_and_sort_free()

    if need_size > len(self.free_pages):
        return None

    select_index = self.free_pages[:need_size]
    self.free_pages = self.free_pages[need_size:]
    return select_index
```

paged allocator 内部管理 page id，返回时再展开成 token index。释放时必须把 token index 除以 `page_size` 并去重，否则同一个 page 里的多个 token 会被重复释放。

```python
# 来源：python/sglang/srt/mem_cache/allocator/paged.py L149-L170
def alloc(self, need_size: int):
    if self.debug_mode:
        assert (
            need_size % self.page_size == 0
        ), "The allocation size should be page-aligned"

    num_pages = need_size // self.page_size
    ...
    out_pages = self.free_pages[:num_pages]
    self.free_pages = self.free_pages[num_pages:]

    out_indices = (
        out_pages[:, None] * self.page_size
        + torch.arange(self.page_size, device=self.device)
    ).reshape(-1)

    return out_indices
```

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

## 第四层：物理 K/V 张量才是真的缓存

`KVCache` 是抽象基类，要求子类提供 `get_key_buffer`、`get_value_buffer`、`get_kv_buffer`、`set_kv_buffer`。MHA、MLA、FP4、SWA、稀疏、统一内存等差异都可以藏在子类里。

```python
# 来源：python/sglang/srt/mem_cache/memory_pool.py L1191-L1272
class KVCache(abc.ABC):
    ...
    @abc.abstractmethod
    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()
```

MHA pool 的默认物理形状是每层一组 K/V buffer；slot 0 的 padded page 负责吸收 dummy 写入。

```python
# 来源：python/sglang/srt/mem_cache/memory_pool.py L1446-L1531
def _create_buffers(self):
    ...
    # [size, head_num, head_dim] for each layer
    # The padded slot 0 is used for writing dummy outputs from padded tokens.
    self.k_buffer = [
        torch.zeros(
            (self.size + self.page_size, self.head_num, self.head_dim),
            dtype=self.store_dtype,
            device=self.device,
        )
        for _ in range(self.layer_num)
    ]
    self.v_buffer = [
        torch.zeros(
            (
                self.size + self.page_size,
                self.head_num,
                self.v_head_dim,
            ),
            dtype=self.store_dtype,
            device=self.device,
        )
        for _ in range(self.layer_num)
    ]
```

## 第五层：HiCache 是层级缓存，不是主路径必需条件

没有 HiCache 时，KV 主要在 GPU pool 中。开启层级缓存后，host pool 按同样的 page 语义管理 L2；storage backend 再提供 L3。

```python
# 来源：python/sglang/srt/mem_cache/pool_host/base.py L79-L143
class HostKVCache(abc.ABC):
    ...
    self.dtype = device_pool.store_dtype
    self.size_per_token = self.get_size_per_token()
    if host_size > 0:
        self.size = sync_fixed_hicache_size(
            int(host_size * 1e9 // self.size_per_token), host_size
        )
    else:
        self.size = int(device_pool.size * host_to_device_ratio)
    ...
    host_mem = psutil.virtual_memory()
    requested_bytes = self.size * self.size_per_token
    available_bytes = host_mem.available - HICACHE_HOST_MEMORY_RESERVE_BYTES
    if requested_bytes > available_bytes:
        raise ValueError(...)
```

这个检查解释了为什么 HiCache 配置过大时会在启动阶段失败，而不是服务运行后随机崩溃。

## 复盘

- `req_pool_idx` 是请求行，不是 KV slot。
- `out_cache_loc` 是本轮新 token 的写入位置。
- `prefix_indices` 是可复用前缀已经占用的 KV slot。
- allocator 管理 slot/page 生命周期，RadixCache 管 prefix 语义，attention backend 管真实 K/V 写入。
- page size 改变的是分配粒度和 page 边界成本，不是“一个 token 的 K/V 大小”。
