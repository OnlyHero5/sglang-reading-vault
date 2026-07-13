---
title: "LoRA · 核心概念"
type: concept
framework: sglang
topic: "LoRA"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# LoRA · 核心概念

## 读者任务

这篇先不按源码文件顺序展开，而是把 LoRA serving 拆成五个对象。读完后，你应该能解释一条请求从 `lora_path` 到 GPU adapter 槽位之间经历了哪些转换，以及每个对象为什么不能互相替代。

## 心理模型：adapter 登机牌

可以把一次 LoRA 请求看成登机流程：

| 登机流程 | 源码对象 | 失效边界 |
|----------|----------|----------|
| 乘客报名字 | `lora_name` / `lora_path` | 用户可见名字不能直接当 GPU 槽位 |
| 换登机牌 | `LoRARegistry` 生成 `lora_id` | registry 不搬运权重 |
| 安检放行 | `Scheduler._can_schedule_lora_req` | 调度器只看本 batch adapter 容量 |
| 找座位 | `LoRAMemoryPool.uid_to_buffer_id` | slot 是执行面状态，不是用户 API 状态 |
| 机上服务 | `LoRABatchInfo.weight_indices` | backend 只消费槽位、rank、scaling 和 segment metadata |

这个类比只用于理解对象边界。真正的源码证据仍然是 `lora_id`、`extra_key`、`weight_indices` 和 GPU buffer 的流转。

## 五个核心对象

### 1. `LoRARef`：稳定身份

启动时通过 `--lora-paths` 传入的 adapter 会生成 deterministic `lora_id`。源码注释说明原因：多节点各自解析参数，如果用随机 ID，同一个 adapter 在不同节点会得到不同 ID，跨节点请求会失配。

```python
# 来源：python/sglang/srt/lora/lora_registry.py L46-L54
    @staticmethod
    def deterministic_id(lora_name: str, lora_path: str) -> str:
        """Stable ``lora_id`` for ``--lora-paths`` adapters.

        Each node in a multi-node launch parses ``--lora-paths`` independently;
        ``uuid4`` would mint a different id per node for the same adapter,
        breaking cross-node lookups when the master broadcasts a request id.
        """
        return uuid5(NAMESPACE_URL, f"{lora_name}\0{lora_path}").hex
```

运行时动态加载的 adapter 可以用新 ID；启动参数里的 adapter 则必须稳定，才能在各个 worker 之间保持一致。

### 2. `LoRARegistry`：请求控制面

`LoRARegistry` 住在 TokenizerManager 进程里。它维护用户可见 adapter 名称到内部 `LoRARef` 的映射，并用 counter 记录正在使用的请求。

```python
# 来源：python/sglang/srt/lora/lora_registry.py L65-L72
class LoRARegistry:
    """
    The central registry to keep track of available LoRA adapters and ongoing LoRA requests.

    The `LoRARegistry` resides in the tokenizer manager process and acts as the single source of truth for all
    available LoRA adapters. It supports concurrent inference and dynamic adapter updates through a two-phase
    update / eventual consistency model between the tokenizer manager process and the scheduler processes.
    """
```

这里的关键词是 single source of truth，但作用域只限“当前可用名称”。它回答“这个名字现在是否可 acquire、对应哪个 `lora_id`、是否还有请求在用”，但不回答“GPU 上哪个 buffer slot 装着它”。TokenizerManager 另有 `lora_ref_cache` 保存曾加载过的历史引用：显式 unload 后名称会从 registry 消失，但后续同名请求可以触发隐式 reload，并获得新的动态 `lora_id`。

### 3. `Req.extra_key`：prefix cache 隔离线

LoRA 改变同一段 token 后续的 hidden states。如果不同 adapter 共用同一份 prefix cache，就会出现语义串线。SGLang 在构造 `Req` 时把 `lora_id` 拼进 `extra_key`。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L782-L789
        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id
```

这不是性能优化，而是正确性边界。同一 prompt、不同 adapter 应该落在不同 cache namespace。

### 4. `LoRAManager`：执行面总管

`LoRAManager` 挂在 `ModelRunner` 上。它选择 backend、初始化 memory pool、包装 base model 里的目标层，并把 batch 中的 `lora_id` 转成 backend 可消费的 `weight_indices`、rank 和 scaling。

```python
# 来源：python/sglang/srt/lora/lora_manager.py L98-L112
        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
        backend_type = get_backend_from_name(lora_backend)
        self.lora_backend: BaseLoRABackend = backend_type(
            max_loras_per_batch=max_loras_per_batch,
            device=self.device,
            server_args=server_args,
        )

        # Initialize mutable internal state of the LoRAManager.
        self.init_state(
            max_lora_rank=max_lora_rank,
            target_modules=target_modules,
            lora_paths=lora_paths,
        )
```

`LoRAManager` 是执行面，不是 registry。动态加载的正常顺序是“后端成功，再注册控制面”，这只避免 backend 失败时出现可请求的半加载 adapter；它不是事务保证。backend 已成功后，registry register、`max_loaded_loras` LRU 卸载或控制面返回仍可能失败，源码没有向已经成功的 backend 发补偿回滚。

### 5. `LoRAMemoryPool`：GPU 槽位分配器

Memory pool 预分配 LoRA A/B buffer，并维护 `uid_to_buffer_id` 与 `buffer_id_to_uid`。注意 `None` 是 base model 的合法 uid，所以空槽不能用 `None` 表示，源码使用 `EMPTY_SLOT`。

```python
# 来源：python/sglang/srt/lora/mem_pool.py L194-L218
        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # Standard LoRA (3D): [num_loras, rank, hidden_dim]
        # MoE LoRA (4D): [num_loras, num_experts, rank, hidden_dim]
        # The dimensionality is determined by the module type (MoE vs standard)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        self.lm_head_A_buffer: Dict[str, torch.Tensor] = {}
        self.lm_head_B_buffer: Dict[str, torch.Tensor] = {}
        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}

        self.embedding_dim: int = self.base_hf_config.hidden_size

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as `EmptySlot` for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        self.buffer_id_to_uid: List[Union[str, None, EmptySlot]] = [
            EMPTY_SLOT
        ] * self.max_loras_per_batch
```

所以 `max_loras_per_batch` 不是“最多注册几个 adapter”，而是“一个执行 batch 的 GPU 槽位账本能同时容纳几个 adapter uid，包括 base model 的 `None` 路径”。

## 两条容易混淆的线

### registry LRU 与 memory pool eviction 不是一回事

`LoRARegistry` 的 LRU 用于 `max_loaded_loras` 控制面注册上限；`LoRAMemoryPool` 的 eviction 用于 GPU buffer slot 不够时换出冷 uid。前者发生在 TokenizerManager，后者发生在 model runner 执行面。开启 overlap loading 时，`max_loaded_loras` 还约束需要 pin 在 CPU 的 adapter 集合，启动期要求它不超过 `2 * max_loras_per_batch`，所以它也不是纯粹的“名称数量”参数。

### `lora_id` 与 `weight_indices` 不是一回事

`lora_id` 是跨请求、跨 worker 的 adapter 身份；`weight_indices` 是本 batch 内每个 request 使用的 GPU buffer slot。`LoRAManager.prepare_lora_batch` 才把二者连接起来。

```python
# 来源：python/sglang/srt/lora/lora_manager.py L330-L349
        weight_indices = [0] * len(forward_batch.lora_ids)
        lora_ranks = [0] * self.max_loras_per_batch
        scalings = [0] * self.max_loras_per_batch
        for i, uid in enumerate(forward_batch.lora_ids):
            if uid not in self.memory_pool.uid_to_buffer_id:
                continue
            weight_indices[i] = self.memory_pool.get_buffer_id(uid)
            if uid is not None:
                lora = self.loras[uid]
                lora_ranks[weight_indices[i]] = lora.config.r
                scalings[weight_indices[i]] = lora.scaling
        # Do in-place updates when CUDA graph is enabled and the batch forward mode
        # could use CUDA graph.
        self.lora_backend.prepare_lora_batch(
            forward_batch=forward_batch,
            weight_indices=weight_indices,
            lora_ranks=lora_ranks,
            scalings=scalings,
            use_cuda_graph=use_cuda_graph,
        )
```

如果你在 kernel 侧看到错 adapter，先不要回到 HTTP 参数找原因，而是先确认本 batch 的 `weight_indices` 是否对应正确的 `uid_to_buffer_id`。

这里还有一个防御性缺口：当某个 `forward_batch.lora_ids` 不在 `uid_to_buffer_id` 时，循环直接 `continue`，该请求的 `weight_indices` 保持默认值 `0`，不会立即报错。slot 0 并不由类型保证永远是 base model；因此控制面身份与 pool 映射一旦失配，可能表现为错误 adapter，而不是清晰的 “uid missing” 异常。

## 更新不是事务：四份状态可能短暂或永久分叉

动态 LoRA 至少涉及四份状态：registry/counter、`lora_ref_cache`、每个 worker 的 `configs/loras/lora_refs` CPU 字典、GPU pool 的 `uid_to_buffer_id` 与权重槽。当前基线没有统一 commit/rollback：

- `LoRAManager.load_lora_adapter` 在权重加载前写入 `configs`；后续异常只返回失败，不删除已经写入的局部条目。
- unload 在控制面先 unregister 并删除 counter，再通知 backend；backend unload 失败时，名称仍不可 acquire，但 worker 可能仍保留 adapter。
- backend unload 只删除 manager CPU 字典，不主动清除 GPU pool 映射；旧 slot 等后续 eviction 覆盖。
- `lora_ref_cache` 不随显式 unload 删除，所以未来请求会尝试重新加载该名称。

这不意味着正常请求必然串线，而是说明排障不能只看 `/load_lora_adapter` 的最终布尔值。必须同时核对名称、ID、worker loaded-adapter 列表和 pool 映射。

## 本篇结论

- `LoRARegistry` 是控制面事实源，负责 name、id、ref count。
- `Req.extra_key` 是 cache 正确性的隔离线。
- `Scheduler` 只做准入，不搬权重。
- `LoRAManager` 把 batch 的 adapter 身份转成执行 metadata。
- `LoRAMemoryPool` 是 GPU slot allocator，不是 adapter registry。
- `LoRABackend` 和 LoRA 包装层只消费 slot、rank、scaling 与 segment，不关心用户传入的名字。
- 动态更新是多状态最终一致流程，不具备跨 registry/backend/pool 的事务回滚。

下一篇 [[SGLang-LoRA-源码走读]] 会沿一条请求把这些对象串起来。

## 运行验证

LoRA 的验证要分清控制面身份、cache 隔离线和执行面 slot。下面的检索覆盖 `LoRARef/LoRARegistry`、`Req.extra_key`、`LoRAManager` 和 `LoRAMemoryPool`。

```powershell
rg -n 'class LoRARef|class LoRARegistry|ref_count|extra_key|class LoRAManager|class LoRAMemoryPool|uid_to_buffer_id|weight_indices|prepare_lora_batch|lora_id|lora_path' sglang/python/sglang/srt/lora/lora_registry.py sglang/python/sglang/srt/lora/lora_manager.py sglang/python/sglang/srt/lora/mem_pool.py sglang/python/sglang/srt/managers/schedule_batch.py
```

读输出时先看 `LoRARef` 的稳定 `lora_id`，再看 `schedule_batch.py` 如何把 `lora_id` 拼进 `extra_key`。执行面排查时重点看 `LoRAManager.prepare_lora_batch` 生成的 `weight_indices`，以及 `LoRAMemoryPool.uid_to_buffer_id` 是否把 adapter id 映射到正确 GPU slot。
