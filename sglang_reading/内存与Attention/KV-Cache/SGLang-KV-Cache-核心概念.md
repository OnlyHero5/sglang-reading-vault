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

这一篇先建立心理模型：SGLang 的 KV Cache 不是单个大 tensor。最小可用模型是三层对象协作：

1. 请求行：`ReqToTokenPool` 给每个活跃请求一行。
2. KV slot/page：allocator 负责发号、回收、延迟整理。
3. 物理存储：`KVCache` 子类按自己的 layout 存 K/V 或压缩状态，attention backend 在翻译后的目标位置写入和读取。

只要分不清这三层，`prefix_indices`、`cache_protected_len`、`out_cache_loc`、`req_pool_idx`、`available_size()` 会混成一团。反过来，也不能把三层模型当成所有配置的精确类图：Unified、SWA、Mamba、MLA/DSA 会在它之上增加地址翻译、并行物理池或状态池。

## 心理模型：图书馆借阅系统

把 KV Cache 想成图书馆：

- `ReqToTokenPool` 是借阅登记表：每个请求一行，每列记录这个 token 借到了哪个书架位置。
- allocator 是空位管理员：知道哪些 slot/page 可用，什么时候把释放的空位重新排序。
- `KVCache` 是书架本身：基线 MHA 是每层 K/V tensor，其他池可以是 combined MLA buffer、SWA 双池、稀疏索引池或特殊 layout。
- RadixCache 是目录：知道哪些前缀已经存过，可以让新请求复用旧书架位置。
- HiCache 是楼下库房或外部仓库：把冷 KV 从 GPU 书架搬到主机内存或存储后端。

## 第一层：请求行不是 KV 内容

`ReqToTokenPool` 不存 K/V 向量，它存“请求 token 位置 → 通用 KV id”的映射。这里保留的是**请求行 0**，真实请求从行 1 开始拿 `req_pool_idx`；设备 KV allocator 也通常保留自己的 page/slot 0，但二者是两套编号，不能混称为同一个“slot 0”。Unified pool 下表里的 id 还可能是 virtual id，而非可直接索引物理 tensor 的地址。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/memory_pool.py L242-L301
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

`BaseTokenToKVPoolAllocator` 统一了 token allocator、paged allocator、SWA allocator、NPU allocator 等实现的接口。它关心的是可调度 slot/page、释放队列和批量释放边界，不理解“这段 KV 属于哪个语义前缀”。prompt、decode、prefix match 等语义由 Scheduler、Req 与 tree cache 维护。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/allocator/base.py L27-L110
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
- `release_pages` 只在 `need_sort` 路径承担延迟合并；不能把它概括成所有 allocator 都必经的释放区。
- `free_group_begin/end` 给批量释放一个事务边界，避免 Radix 或结果处理频繁排序。

## 第三层：token 粒度和 page 粒度

token allocator 以单个 slot 为最小单位。它简单、直观，但不满足 paged attention 对 page 边界的要求。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/allocator/token.py L42-L76
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
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/allocator/paged.py L149-L170
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
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/memory_pool.py L1191-L1272
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

MHA pool 的 **NHD 基线**是每层一组 K/V buffer；额外 padded page 负责吸收 dummy 写入。这张卡只证明默认形态，不代表所有 `KVCache` 子类。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/memory_pool.py L1446-L1531
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

### 三层模型在哪些地方会扩展

| 配置/池 | 基线模型的扩展点 | 读源码时要追的对象 |
| ------ | ---------------- | ------------------ |
| Unified memory pool | `req_to_token` 与 `out_cache_loc` 可携带 virtual id，forward metadata 统一翻译成 physical id | `KVWriteLoc.loc`、`full_loc`、v2p 表 |
| SWA hybrid | 同一通用位置可能对应 full 与 SWA 子池的不同物理位置 | `swa_loc`、`full_loc`、`SWAKVPool` |
| Hybrid Mamba | 除请求行与 attention KV 外，还有 request-scoped state、ping-pong/track slot | `HybridReqToTokenPool` 与 mamba allocator |
| MLA / DSA / sparse | 物理内容不再是标准 MHA 的每层 K/V 二元组 | combined KV、index K、scale 等 buffer |
| HND / PageMajor / `vectorized_5d` | “slot”仍是逻辑分配单位，物理维度与写 kernel 已改变 | pool layout 与 `_store_kv_layer` |
| NoOp KV pool | 只保留容量契约和极小占位 buffer，真实读写会 fail loud | 严格 prefill-only 使用条件 |

地址翻译之所以必须显式建模，是因为当前代码把所有写目标放进 `KVWriteLoc`：

```python
# 来源：python/sglang/srt/mem_cache/memory_pool.py L1148-L1160
@dataclass
class KVWriteLoc:
    """Write target(s) for ``KVCache.set_kv_buffer``.

    All location info lives here (in the attention metadata), NOT in the pool:
    - ``loc``: the generic per-token write location (the allocated
      ``out_cache_loc``). VIRTUAL under the unified memory pool (it indexes the
      virtual slot space); already physical for a non-unified memory pool.
    - ``swa_loc``: the pre-translated SWA-sub-pool PHYSICAL location for hybrid
      SWA pools (``None`` otherwise).
    - ``full_loc``: the pre-translated full-attention-sub-pool PHYSICAL location
      for the unified memory pool (``None`` otherwise), computed once per forward in
      attention metadata (``ForwardMetadata.out_cache_loc_full_physical``). The
```

因此，“`out_cache_loc` 就是物理 tensor 下标”只在非 Unified 等基线路径成立。更稳妥的说法是：它是 allocator 返回的**通用写入位置**；attention metadata/backend 负责把它落实为当前物理池真正接受的地址。

## 第五层：HiCache 是层级缓存，不是主路径必需条件

没有 HiCache 时，KV 主要在 GPU pool 中。开启层级缓存后，host pool 按同样的 page 语义管理 L2；storage backend 再提供 L3。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/pool_host/base.py L79-L143
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

fixed-GB 配置还会在 PP ranks 之间同步为最小 token capacity，并向 page 对齐；容量小于 device pool 只产生 warning。真正的 fail-fast 来自扣除 10 GiB 预留后的可用主机内存不足。host pool 的 `alloc/free` 另有锁与 double alloc/free 检查，销毁时还必须注销 pinned buffer，所以 HiCache 不是“多一个 Python list”。

Storage backend 是再外一层的 L3 接口：内置 backend 延迟 import，且不同 backend 的构造参数并不统一；`dynamic` 配置必须提供 `backend_name/module_path/class_name`，当前构造契约是 `backend_class(storage_config, kwargs)`，不会自动把 `mem_pool_host` 当通用参数传入。

## 复盘

- `req_pool_idx` 是请求行，不是 KV slot。
- `out_cache_loc` 是本轮新 token 的通用写入位置；是否已经是 physical id 取决于 pool。
- `prefix_indices` 表示请求下一轮可从哪里继续计算所需的 KV 索引集合：刚做完 match 时是 device tree hit；chunk commit 后还可能包含 tree canonical indices 与请求私有 tail。它不等于 tree 已接管的长度。
- `cache_protected_len` 才回答“Radix tree 真正保护/拥有到哪里”。
- allocator 管理 slot/page 生命周期，RadixCache 管 prefix 语义，attention backend 管真实 K/V 写入。
- page size 改变的是分配粒度和 page 边界成本，不是“一个 token 的 K/V 大小”。
