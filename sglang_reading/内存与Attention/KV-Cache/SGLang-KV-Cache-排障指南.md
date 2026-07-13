---
title: "KV-Cache · 排障指南"
type: troubleshooting
framework: sglang
topic: "KV-Cache"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# KV-Cache · 排障指南

## 读者任务

这一篇按排障症状组织。遇到 KV 相关问题时，先判断它属于哪一层：请求行不足、KV slot/page 不足、逻辑/virtual/physical 地址混淆、paged 边界、decode retract/abort、prefix 复用失效、HiCache 主机内存，还是 storage backend 配置。

## 症状 1：请求刚要 prefill 就报请求行不足

现象：报错里出现 `alloc_req_slots runs out of memory`，并提示调小 `--max-running-requests`。

判断：这不是 KV 物理 slot 不足，而是 `ReqToTokenPool` 请求行不足。请求还没有进入真正的 K/V 写入阶段。

源码入口：`alloc_for_extend` 先分配请求行，失败后直接抛错。

```python
# 来源：python/sglang/srt/mem_cache/common.py L431-L438
req_pool_indices = req_to_token_pool.alloc(reqs)
if req_pool_indices is None:
    raise RuntimeError(
        "alloc_req_slots runs out of memory. "
        "Please set a smaller number for `--max-running-requests`. "
        f"{req_to_token_pool.available_size()=}, {num_reqs=}, "
    )
return req_pool_indices
```

验证：

- 看 `max_running_requests` 和实际并发请求数。
- 区分 `req_to_token_pool.available_size()` 与 `token_to_kv_pool_allocator.available_size()`。
- 如果只有请求行耗尽，调大 KV token 数不一定解决问题。

## 症状 2：prefill 或 decode 分配 KV slot 失败

现象：prefill 报 `Prefill out of memory`、decode 报 `Decode out of memory`，或上游进入 evict/retract 路径。

判断：这是 KV slot/page 不足，但要继续区分两个层次：`alloc_*` helper 会先 evict，仍拿不到位置就 fail loud 抛 `RuntimeError`；正常 decode 调度还会在真正分配前通过 `check_decode_mem` 估算下一轮需求并 retract。若 helper 仍在 decode forward 前失败，说明准入估算、特殊 allocator 状态或并发时序需要继续核对，不能把异常本身当成正常 retract。

源码入口：paged decode 会先保守估算每个请求可能需要一个新 page，然后 evict，再调用 allocator。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/common.py L527-L576
allocator = tree_cache.token_to_kv_pool_allocator
# Over estimate the number of tokens: assume each request needs a new page.
num_tokens = len(seq_lens) * allocator.page_size
evict_from_tree_cache(tree_cache, num_tokens)
...
out = allocator.alloc_decode(seq_lens, seq_lens_cpu, last_loc, **extra_alloc_kwargs)
...
if out_cache_loc is None:
    error_msg = (...)
    logger.error(error_msg)
    ...
    raise RuntimeError(error_msg)
```

验证：

- 查 batch size、最大输出长度、`max_total_num_tokens`、prefix cache evictable tokens。
- 如果错误来自 decode，先看 Scheduler 是否在 `check_decode_mem` 之前已经判断不足。
- 查看 `new_tokens_required_next_decode`：普通 decode 按新 page 估算；spec v2 则依据 committed/allocated 与 reserve 计算，二者不是同一公式。
- 压测时搜索 `KV cache pool is full`、`Retract requests`、`Decode out of memory`。

## 症状 3：KV pool 满后部分请求延迟升高，但服务没崩

现象：日志出现 KV pool 满或 retract，部分请求变慢，服务继续处理其他请求。

判断：这是 decode 侧容量保护。Scheduler 从 running batch 中撤回一部分请求，释放它们的 KV，再把它们重新调度。

源码入口：`check_decode_mem` 先 evict tree cache；仍不足时 `retract_decode` 调 `release_req`。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/schedule_batch.py L2453-L2532
def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
    num_tokens = self.new_tokens_required_next_decode(selected_indices)
    evict_from_tree_cache(self.tree_cache, num_tokens)
    return self.token_to_kv_pool_allocator.available_size() >= num_tokens

def retract_decode(
    self, server_args: ServerArgs
) -> Tuple[List[Req], float, List[Req]]:
    ...
    while first_iter or (
        not self.check_decode_mem(selected_indices=sorted_indices)
    ):
        if len(sorted_indices) == 1:
            break
        ...
        retracted_reqs.append(req)
        self.release_req(idx, len(sorted_indices), server_args)
```

`release_req` 负责释放请求占用的 KV，并重置请求状态：

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/schedule_batch.py L1590-L1612
def release_req(
    *,
    req: Req,
    remaing_req_count: int,
    server_args: ServerArgs,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    tree_cache: BasePrefixCache,
    hisparse_coordinator: Optional[HiSparseCoordinator],
) -> None:
    ...
    release_kv_cache(req, tree_cache, is_insert=False)
    ...
    evict_from_tree_cache(tree_cache, num_tokens)

    req.reset_for_retract()
```

验证：

- 如果 retract 高频出现，先降低并发或最大输出长度。
- 看 prefix cache 命中率和可 evict 空间；prefix 长但不可 evict 时，decode 更容易挤压。
- 区分“单次保护性 retract”和“持续容量配置不匹配”。
- 如果只剩一个请求也放不下，当前代码会给它设置 HTTP 500 的 `FINISH_ABORT`、释放资源并作为 `reqs_to_abort` 返回；这不是“继续 retract 等下次重试”。

## 症状 4：`page_size` 改了以后延迟或空间利用变化明显

现象：改 `--page-size` 后，prefill/decode 分配、fragmentation 或 attention backend 行为变化。

判断：page size 不是单纯性能开关，它改变 allocator 的最小管理单位。paged allocator 内部拿 page，返回 token index；decode 只有跨 page 才消耗新 page。

源码入口：ModelRunner 根据 page size 选择 token allocator 或 paged allocator。

```python
# 来源：python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py L1132-L1148
elif self.page_size == 1 and self.dcp_size == 1:
    self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        self.max_total_num_tokens,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kvcache=self.token_to_kv_pool,
        need_sort=need_sort,
    )
else:
    self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
        self.max_total_num_tokens * self.dcp_size,
        page_size=self.page_size * self.dcp_size,
        dtype=self.kv_cache_dtype,
        device=self.device,
        kvcache=self.token_to_kv_pool,
        need_sort=need_sort,
    )
```

paged free 必须按 page 去重：

```python
# 来源：python/sglang/srt/mem_cache/allocator/paged.py L265-L272
if self.is_not_in_free_group:
    free_page_indices = torch.unique(free_index // self.page_size)
    if self.need_sort:
        self.release_pages = torch.cat((free_page_indices, self.release_pages))
    else:
        self.free_pages = torch.cat((free_page_indices, self.free_pages))
else:
    self.free_group.append(free_index)
```

验证：

- page size 越大，跨 page 管理开销可能更低，但内部碎片可能更明显。
- DCP 下 effective page size 可能是 `page_size * dcp_size`。
- 如果关注 decode，每步是否跨 page 比单个 token 大小更重要。

## 症状 5：attention 写 KV 时疑似越界或 silent corruption

现象：CUDA illegal memory access、输出异常、或怀疑 `out_cache_loc` 指向了错误 slot。

判断：物理写入边界在 `KVCache.set_kv_buffer`，但不能假设 allocator 返回值始终能直接索引当前物理 tensor。Unified 下 `out_cache_loc` 可是 virtual id；SWA/full 子池还有各自 physical loc。attention metadata/backend 先构造 `KVWriteLoc`，pool 才消费属于自己的物理目标。

源码入口：MHA pool 写入前会解包 `KVWriteLoc`，并在 async assert 开启时检测越界。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/memory_pool.py L1669-L1730
def set_kv_buffer(
    self,
    layer: RadixAttention,
    loc_info,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    ...
):
    loc, _, _ = unwrap_write_loc(loc_info)
    maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MHA)")
    ...
    if self.use_hnd:
        pages = loc // self.page_size
        offs = loc % self.page_size
        k_buf[pages, :, offs, :] = cache_k
        v_buf[pages, :, offs, :] = cache_v
        return

    self._store_kv_layer(layer_id - self.start_layer, loc, cache_k, cache_v)
```

验证：

- 开启相关 async assert 或 debug memory pool 环境变量做定位。
- 检查普通 extend 中 `out_cache_loc` 长度是否等于本轮新增 token 数；普通 decode `token_per_req=1` 时才等于 batch size，推测/DSV4 不要套这个断言。
- Unified/SWA 路径同时检查 `KVWriteLoc.loc/swa_loc/full_loc` 的角色是否匹配，避免拿 virtual id 对 physical pool 做越界判断。
- 对 HND/page layout，确认代码使用 page/off 写法，而不是把 loc 当一维连续行。
- 对 MLA/DSA、PageMajor、ROCm `vectorized_5d` 与 NoOp pool，先确认实际 pool 类型和 layout；不要只盯 `k_buffer/v_buffer` 这对 MHA 字段。

## 症状 6：HiCache 启动时报主机内存不足

现象：启动阶段直接报 `Not enough host memory available`。

判断：Host pool 在初始化时就按每 token KV 字节数估算容量，并预留固定主机内存给系统。这个失败发生在 buffer 分配前，是预期保护。

源码入口：

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/pool_host/base.py L110-L143
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
    raise ValueError(...)
```

验证：

- 减小 `--hicache-ratio` 或 `--hicache-size`。
- 观察 host pool 是否小于 device pool；小于也能跑，但 L2 命中收益会下降。
- 多 PP stage 使用固定 GB 配置时，注意不同 stage 每 token 字节数可能不同。
- fixed-GB 模式会把各 PP rank 的 token capacity 同步为最小值并按 page 对齐；看到某 rank 容量比本地估算小，先检查这个全局一致性步骤。
- 若问题发生在服务重启或反复初始化后，检查 `destroy()` 是否执行并注销 pinned host buffers，而不只看初次容量。

## 症状 7：storage backend 名称不识别

现象：启用 HiCache storage 后报未知 backend，或动态后端类导入失败。

判断：`StorageBackendFactory` 先查内置注册表并 lazy import；`dynamic` 需要提供 `extra_config` 的后端名、模块路径和类名。内置 backend 的构造参数由工厂逐个分派，不能假设统一签名。

源码入口：

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/storage/backend_factory.py L66-L103
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

if backend_name == "dynamic" and storage_config.extra_config is not None:
    backend_config = storage_config.extra_config
    return cls._create_dynamic_backend(
        backend_config, storage_config, mem_pool_host, **kwargs
    )
```

内置后端在文件末尾注册：

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/mem_cache/storage/backend_factory.py L194-L239
StorageBackendFactory.register_backend(
    "file", "sglang.srt.mem_cache.hicache_storage", "HiCacheFile"
)
...
StorageBackendFactory.register_backend(
    "mooncake",
    "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",
    "MooncakeStore",
)
...
StorageBackendFactory.register_backend(
    "mori",
    "sglang.srt.mem_cache.storage.umbp.umbp_store",
    "UMBPStore",
)
```

验证：

- 先确认 backend 名是否在注册表中：`file`、`nixl`、`mooncake`、`hf3fs`、`aibrix`、`eic`、`simm`、`mori`。
- dynamic 后端确认 `backend_name/module_path/class_name` 三个字段。
- 当前 dynamic 实例化调用是 `backend_class(storage_config, kwargs)`；自定义类若期待裸 `**kwargs` 或 `mem_pool_host`，会发生签名错配，需按当前工厂契约实现。
- backend 初始化失败时区分 import 错误、类未继承 `HiCacheStorage`、运行时依赖缺失。

## 症状 8：相同 prompt 再请求仍慢，怀疑 prefix/KV 复用失效

现象：相同或高度相似的 prompt 第二次请求 TTFT/总延迟仍异常，但没有明确 OOM。

判断：先验证“是否真的形成 device hit、还剩多少 extend token、是否排队或进入不同 batch”，不能直接把问题归因到 allocator `free()`。`prefix_indices` 的长度描述下一轮从哪里继续算，`cache_protected_len` 才描述 tree ownership；chunked prefill 后二者可能不同。

操作与预期：

- 对比 prefix match 结果、extend length、排队时间与 batch 组成。预期：若复用有效，命中区间增加、实际 extend token 减少；端到端延迟是否下降还受排队和 workload 影响。
- 核对请求是否因 page 对齐、cache key、LoRA/多模态输入或配置差异无法共享。预期：先找到语义或 key 差异，再判断 allocator。
- 只有在 HIP/ROCm 且证据指向首次 duplicate-free 路径时，才考虑 `torch.unique` 的 warmup/JIT。当前 `PagedTokenToKVPoolAllocator.__init__` 已主动 warmup 该路径，因此它不是“相同 prompt 第二次慢”的通用首查项。

## 最小排障顺序

1. 先区分请求行不足和 KV slot/page 不足。
2. 看问题发生在 prefill、普通/推测 decode、attention 地址翻译与写入、retract，还是 HiCache 初始化。
3. 若是 decode 容量，查 `check_decode_mem` 和 `retract_decode`。
4. 若是 page 行为，查 `page_size`、DCP、paged allocator 的 `alloc_extend/alloc_decode/free`。
5. 若是写入异常，按 generic/virtual/physical 顺序查 `out_cache_loc`、`KVWriteLoc`、`set_kv_buffer`。
6. 若是 HiCache，先看 host pool 容量，再看 storage backend 注册和导入。
7. 若是复用收益异常，先核对 hit/extend/排队，不先猜性能原因。

## 运行验证

FAQ 场景很多，维护时先用一个源码检索确认排障入口仍在同一组模块里：decode 准入与 retract 在 `ScheduleBatch`，KV 写入在 memory pool / model runner，paged allocator 管 page，HiCache host 与 storage backend 管二级缓存。

```powershell
rg -n 'class KVCache|class BaseTokenToKVPool|def check_decode_mem|def retract_decode|def prepare_for_decode|def prepare_for_extend|def set_kv_buffer|class PagedTokenToKVPoolAllocator|class MHATokenToKVPoolHost|class StorageBackendFactory|register_backend|create_backend' sglang/python/sglang/srt/mem_cache/common.py sglang/python/sglang/srt/managers/schedule_batch.py sglang/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py sglang/python/sglang/srt/mem_cache/allocator/paged.py sglang/python/sglang/srt/mem_cache/memory_pool.py sglang/python/sglang/srt/mem_cache/pool_host/base.py sglang/python/sglang/srt/mem_cache/storage/backend_factory.py
```

命中结果要能对应上面的最小排障顺序。如果 `check_decode_mem`、`retract_decode` 或 `StorageBackendFactory.register_backend` 的位置变化，优先重读对应症状段，不要只按旧行号继续定位。
