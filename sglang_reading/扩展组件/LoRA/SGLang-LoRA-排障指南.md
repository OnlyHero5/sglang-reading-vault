---
title: "LoRA · 排障指南"
type: troubleshooting
framework: sglang
topic: "LoRA"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# LoRA · 排障指南

## 读者任务

这篇按症状排障。先判断问题落在控制面、调度面、执行面还是 kernel metadata，再回到对应源码入口。

| 症状 | 优先看 |
|------|--------|
| 请求一进来就报 LoRA 未启用或 adapter 不存在 | TokenizerManager / LoRARegistry |
| adapter 动态加载失败 | TokenizerControlMixin / LoRAManager.validate_new_adapter |
| 请求长期排队或 batch 进不去 | Scheduler / LoRADrainer / LoRAOverlapLoader |
| 同 prompt 不同 adapter 输出疑似串线 | `Req.extra_key` 与 RadixCache namespace |
| GPU slot 满、eviction 异常、pinned adapter 饥饿 | LoRAMemoryPool / eviction policy |
| MoE LoRA shape mismatch | MemoryPool 的 EP/TP expert buffer 规则 |
| embedding 或 lm_head LoRA 行为不符合预期 | LoRAManager target modules 与 mem_pool 加载分支 |
| load/unload 返回失败后状态怪异 | registry、`lora_ref_cache`、worker CPU 字典、GPU pool 四方对账 |
| lm_head + input logprobs 才崩 | Triton per-pass `lm_head_batch_info` 构造 |

## 1. 请求带了 LoRA，为什么直接报没有启用

这是控制面配置问题。请求解析时如果 `obj.lora_path` 非空但 `enable_lora` 为假，会直接抛错，不会进入 scheduler。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L2778-L2794
        if not obj.lora_path:
            return

        if not self.server_args.enable_lora:
            first_adapter = (
                obj.lora_path
                if isinstance(obj.lora_path, str)
                else next((a for a in obj.lora_path if a), None)
            )

            raise ValueError(
                f"LoRA adapter '{first_adapter}' was requested, but LoRA is not enabled. "
                "Please launch the server with --enable-lora flag and preload adapters "
                "using --lora-paths or /load_lora_adapter endpoint."
            )

        await self._resolve_lora_path(obj)
```

验证方法：检查启动参数是否显式传了 `--enable-lora`，或是否传了 `--lora-paths` 让 ServerArgs 自动打开 LoRA。若用户显式设置 `enable_lora=False`，`--lora-paths` 会被忽略。

## 2. adapter 名字明明传了，为什么说没加载

`LoRARegistry.acquire` 会查 registry 里的 name。查到后会把这个 name 移到 LRU 末尾，并返回内部 `lora_id`；如果查不到，`_lookup` 会在同一个分支里抛出未加载错误。这说明 adapter 没有成功 register，或者被 unload 后不在 registry 中。

```python
# 来源：python/sglang/srt/lora/lora_registry.py L142-L150
            self._registry.move_to_end(name)
            return lora_ref.lora_id

        if isinstance(lora_name, str):
            async with self._registry_lock.writer_lock:
                lora_id = _lookup(lora_name)

            await self._counters[lora_id].increment(notify_all=False)
            return lora_id
```

验证方法：在 `TokenizerManager._resolve_lora_path` 看 `obj.lora_path` 是否和 `lora_ref.lora_name` 一致。注意请求字段使用的是 adapter 名称，不是内部 `lora_id`。

## 3. 动态加载为什么在 DP 下不可用

当前动态 load/unload 路径在 TokenizerManager 里有硬断言：`dp_size` 必须为 1。这是源码现状，不是文档漏写。还要注意该断言抛 `AssertionError`，而方法尾部只捕获 `ValueError`；因此不能保证 API 总能返回结构化 `success=false`，实际可能表现为未处理异常/5xx。

```python
# 来源：python/sglang/srt/managers/tokenizer_control_mixin.py L554-L558
            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
```

验证方法：如果 `/load_lora_adapter` 在多 DP 配置下失败，先确认 `server_args.dp_size`。生产上需要预加载 adapter，或等 upstream 放开动态 DP 路径。

## 4. 为什么有的 adapter load 失败，报 rank 或 target modules 不兼容

`LoRAManager.validate_new_adapter` 会在真正加载权重前校验 adapter 能否装进当前 memory pool。rank 超出 `--max-lora-rank`，或 adapter target module 不在 `--lora-target-modules` 内，都会失败。

```python
# 来源：python/sglang/srt/lora/lora_manager.py L230-L238
        # Check if the LoRA adapter shape is compatible with the current LoRA memory pool configuration.
        memory_pool = getattr(self, "memory_pool", None)
        incompatible = memory_pool and not memory_pool.can_support(lora_config)
        if incompatible:
            raise ValueError(
                f"LoRA adapter {lora_ref.lora_name} with rank {lora_config.r} is incompatible with the current "
                "LoRA memory pool configuration. Please ensure that the LoRA adapter's rank is within the configured "
                "`--max-lora-rank` and that the target modules are included in `--lora-target-modules`."
            )
```

更早的 target module 推断阶段也会检查 CLI 指定的 target modules 是否覆盖 adapter 自身声明。

```python
# 来源：python/sglang/srt/lora/lora_manager.py L594-L604
            if target_modules is not None:
                # When `--lora-target-modules` is provided, validate adapter target modules is a subset of the specified target modules.
                if not adapter_target_modules.issubset(self.target_modules):
                    unsupported_modules = adapter_target_modules - self.target_modules
                    lora_name = self.lora_refs[lora_id].lora_name
                    raise ValueError(
                        f"LoRA adapter '{lora_name}' contains target modules {sorted(unsupported_modules)} "
                        f"that are not included in the specified --lora-target-modules {sorted(self.target_modules)}. "
                        f"Please update --lora-target-modules to include all required modules: "
                        f"{sorted(self.target_modules | adapter_target_modules)}, or use 'all' to enable all supported modules."
                    )
```

验证方法：对比 adapter 的 PEFT config 里的 `r` 和 `target_modules`，以及服务启动参数的 `--max-lora-rank`、`--lora-target-modules`。如果要自动覆盖支持模块，可用 `--lora-target-modules all`，但要接受更大的 memory pool 形状。

## 5. added tokens 和 DoRA 为什么不能用

当前执行面明确拒绝 added tokens adapter 和 DoRA adapter。虽然 memory pool 里存在部分 extra embedding buffer 路径，但 load 前校验会先挡住 added tokens。

```python
# 来源：python/sglang/srt/lora/lora_manager.py L203-L215
    def validate_new_adapter(self, lora_config: LoRAConfig, lora_ref: LoRARef):
        """
        Validate if an adapter can be loaded into the current LoRA memory pool and generate error if it is incompatible.
        """
        if lora_config.lora_added_tokens_size > 0:
            raise ValueError(
                f"Failed to load {lora_ref.lora_name} because LoRA serving currently doesn't support adapters that add tokens to the vocabulary"
            )

        if lora_config.use_dora:
            raise ValueError(
                f"Failed to load {lora_ref.lora_name} because LoRA serving currently doesn't support DoRA adapters"
            )
```

验证方法：看 adapter config 里的 added tokens 和 DoRA 字段。不要仅凭 memory pool 有 extra-token 代码就判断 serving 已支持。

## 6. 同一个 prompt 用不同 adapter，会不会共享错误 prefix cache

不会，前提是请求走标准 `Req` 构造路径。`Req.__init__` 会把 `lora_id` 拼入 `extra_key`，从而隔离 RadixCache namespace。

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

验证方法：在 `Req.__init__` 或 cache lookup 前打断点，确认带 LoRA 请求的 `extra_key` 包含内部 `lora_id`。如果你改了请求构造路径，必须保持这个不变量。

## 7. 为什么 batch 明明还有 token 空间，LoRA 请求还是排不进去

LoRA 准入看的是 adapter 种类和 slot，不只是 token 或 request 数。`Scheduler._can_schedule_lora_req` 会把当前 running adapters 和新请求 adapter 合并后交给 `validate_lora_batch`。

```python
# 来源：python/sglang/srt/managers/scheduler.py L3011-L3024
        if req.lora_id in running_loras:
            return True

        if self.enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )
```

验证方法：打印 `running_loras` 和候选 request 的 `req.lora_id`。如果唯一 uid 数（包括 base 的 `None`）超过 `max_loras_per_batch`，它不会进入本轮 batch。

## 8. pinned adapter 为什么可能让普通 adapter 饿死

源码禁止 pinned adapter 占满所有 slot，因为必须给 unpinned adapter 和 base model 留空间。

```python
# 来源：python/sglang/srt/lora/lora_manager.py L240-L247
        # Ensure pinned LoRA adapters does not exceed maximal limit or cause starvation.
        if lora_ref.pinned and self.num_pinned_loras >= self.max_loras_per_batch - 1:
            raise ValueError(
                f"Failed to load LoRA adapter {lora_ref.lora_name} as a pinned adapter. It is not allowed to pin all slots "
                "in the LoRA memory pool to avoid starvation for unpinned adapters and base models. Please increase your "
                "`--max-loras-per-batch` or load it as unpinned LoRA adapters."
            )
```

验证方法：统计 `num_pinned_loras` 与 `max_loras_per_batch`。如果 pinned adapter 接近上限，普通 adapter 会频繁等待、换出或无法进入 batch。

## 9. memory pool eviction 会不会换出当前 batch 正在用的 adapter

不会。memory pool 选择 victim 时会跳过当前 batch 需要的 uid，也会跳过 pinned adapter。

```python
# 来源：python/sglang/srt/lora/mem_pool.py L689-L724
            # 2. Memory pool is full, need to evict using policy
            candidates = set()

            for buffer_id in range(self.max_loras_per_batch):
                uid = self.buffer_id_to_uid[buffer_id]

                # Skip if this adapter is needed by current batch
                if uid in cur_uids:
                    continue

                # Skip if this adapter is pinned
                if uid is not None:
                    lora_ref = lora_refs.get(uid)
                    if lora_ref and lora_ref.pinned:
                        continue

                candidates.add(uid)

            if not candidates:
                raise ValueError(
                    "No available buffer slots found. Please ensure the number of active (pinned) loras is less than max_loras_per_batch."
                )

            # Prefer evicting LoRA adapters over the base model (None).
            # Only evict None when the batch consists entirely of LoRA requests
            # and no other adapters can be evicted.
            non_none_candidates = candidates - {None}
            if non_none_candidates:
                # Prioritize evicting actual LoRA adapters
                candidates_to_use = non_none_candidates
            else:
                # Only None is available for eviction (batch is all LoRA requests)
                candidates_to_use = candidates

            # Select victim using eviction policy
            victim_uid = self.eviction_policy.select_victim(candidates_to_use)
```

验证方法：在 `prepare_lora_batch` 看 `cur_uids`、`buffer_id_to_uid` 和 `candidates`。如果报 no available slot，通常是当前 batch 加 pinned adapter 已占满可用槽位。

## 10. strict loading 到底控制什么

当 adapter 里有权重名匹配不上当前 target modules，memory pool 的预检会构造 warning 或 error。`lora_strict_loading=True` 时直接失败；非 strict 时虽然先 warning，但当前实现随后仍在真实加载循环中无条件调用 `get_target_module_name`。因此“warning 后可靠跳过”不是当前基线能兑现的契约，真正未知的 layer weight 仍可能导致 load 失败。

```python
# 来源：python/sglang/srt/lora/mem_pool.py L814-L826
        if skipped_weight_names:
            msg = (
                f"LoRA adapter '{uid}': {len(skipped_weight_names)} weight(s) "
                f"skipped because they did not match any target module in "
                f"{sorted(self.target_modules)}. Skipped weights: "
                f"{sorted(skipped_weight_names)}. This likely indicates a "
                f"mismatch between the adapter's target modules and the base "
                f"model architecture."
            )
            if self.strict_loading:
                raise ValueError(msg)
            else:
                logger.warning(msg)
```

验证方法：如果看到 skipped weights warning，继续观察最终 load 结果和 worker 状态，不要把 warning 当作已成功跳过。生产验收应优先修正 adapter/base target-module 对齐；strict 模式只负责更早失败，并不能修复非 strict 后续再次解析的问题。

## 11. MoE LoRA 在 EP/TP 下为什么容易 shape mismatch

MoE per-expert buffer 的切片依据不是外层 `tp_size`，而是 `moe_tp_size`。源码注释明确说明：在 `--tp 4 --ep 4` 下，每个 rank 可能持有 full-width expert weights，如果按外层 TP 切会得到错误宽度。

```python
# 来源：python/sglang/srt/lora/mem_pool.py L180-L189
        # Per-expert MoE weights are sharded by `moe_tp_size`, NOT the outer
        # `tp_size`: `moe_tp_size = tp_size // ep_size // dp_size`, so under
        # e.g. `--tp 4 --ep 4` each rank holds full-width expert weights
        # (`moe_tp_size == 1`). Sizing per-expert LoRA buffers by `tp_size`
        # here would yield a 4x-narrower inner dim than the adapter weight
        # (which `FusedMoEWithLoRA.slice_moe_lora_{a,b}_weights` correctly
        # skip-slices when `moe_tp_size <= 1`), producing a shape-mismatch
        # assert during weight load. Non-MoE modules still shard by
        # `tp_size` because attention TP is unchanged.
        self.moe_tp_size, self.moe_tp_rank = _get_moe_tp_context()
```

验证方法：遇到 MoE LoRA shape mismatch，先打印 `tp_size`、`ep_size`、`dp_size`、`moe_tp_size`，再确认 adapter 权重是否按 MoE expert 维度组织。

## 12. embedding 和 lm_head LoRA 为什么看起来规则不同

embedding LoRA B 不按 TP 切片，lm_head LoRA B 会按 vocab 维度对 TP rank 切片。非最后 PP stage 可以合法跳过本地不存在的 lm_head 权重。

embedding LoRA B 分支：

```python
# 来源：python/sglang/srt/lora/mem_pool.py L1235-L1247
                    lora_b_weights = weights
                    # TP is supported by keeping embedding LoRA B unsharded;
                    # no slicing needed.

                    buffer_view = self.embedding_B_buffer[target_module][
                        buffer_id, :, :lora_rank
                    ]
                    lora_b_weights = self._get_maybe_cached_weight_for_transfer(
                        pinned_embedding_layers,
                        name,
                        lora_b_weights,
                    )
                    load_lora_weight_tensor(buffer_view, lora_b_weights)
```

lm_head LoRA B 分支：

```python
# 来源：python/sglang/srt/lora/mem_pool.py L1267-L1282
                elif (
                    target_module == "lm_head"
                    and lora_lm_head_module is not None
                    and "lm_head" in name
                    and ("lora_embedding_B" in name or "lora_B" in name)
                ):
                    assert lora_lm_head_module is not None
                    lora_b_weights = weights
                    # Slice B along vocab dimension for this TP rank
                    if self.tp_size > 1:
                        lora_b_weights = lora_lm_head_module.slice_lora_b_weights(
                            lora_b_weights, self.tp_rank
                        )
                        cache_key = append_cache_key_suffix(name, f"tp{self.tp_rank}")
                    else:
                        cache_key = name
```

验证方法：排查 embedding 或 lm_head adapter 时，不要只看普通 Linear LoRA 的 A/B 切片规则。要分别看 embedding、lm_head 和 PP rank 所在分支。

## 13. load 返回失败，为什么 worker 内仍可能残留状态

动态加载至少有两层非事务边界。worker 内部先把 config 写入 `self.configs`，再加载权重；异常只返回失败，没有 rollback。控制面则先让 backend 成功，再 register；若 register/LRU 收口失败，也不会撤销 backend 已完成的加载。

验证方法：失败后同时检查 `LoRARegistry.get_all_adapters()`、`lora_ref_cache`、各 worker 返回的 `loaded_adapters`，以及 `LoRAManager.configs/loras/lora_refs`。不要仅凭 HTTP status 推断所有 rank 都回到旧状态。

## 14. unload 成功或失败后，GPU slot 为什么还看得到旧 ID

`LoRAManager.unload_lora_adapter` 只删除 CPU 字典和 pinned 计数，没有删除 `memory_pool.uid_to_buffer_id`。旧 GPU slot 会留作冷 cache，直到后续 batch 需要 slot 时被 eviction。显式 unload 也不删除 TokenizerManager 的 `lora_ref_cache`，所以未来同名请求可以隐式 reload，并生成新的动态 ID。

验证方法：区分“名称当前不可 acquire”“worker CPU adapter 已删除”“GPU slot 已擦除”三个命题。前两者成功也不能推出第三个；reload 后还应确认请求使用新 ID，而不是旧 pool uid。

## 15. 为什么 pool 映射缺失时可能不是报错，而是读到 slot 0

`LoRAManager.prepare_lora_batch` 把 `weight_indices` 初始化为全 0；若 uid 不在 `uid_to_buffer_id`，循环直接 continue。正常调度应先 fetch/等待 overlap event，但一旦控制面与 pool 状态失配，这里没有 fail fast。slot 0 可能属于 base，也可能属于其他 adapter。

验证方法：出现疑似串 adapter 时同时打印 `forward_batch.lora_ids`、`uid_to_buffer_id`、`buffer_id_to_uid` 和最终 `weight_indices`。发现 uid 缺失时应在进入 kernel 前停止，而不是把 0 当作合法 base 默认值。

## 16. lm_head 只在 input logprobs 分块时崩，查哪里

Triton backend 的 per-pass 构造创建局部空列表，却向 `self.lm_head_pass_batch_infos` append；首轮该成员为 `None`。触发条件是 lm_head 被 LoRA target、extend 请求需要 input logprobs、启用 logits processor chunking，且 pruned token 数超过 chunk size。普通 decode、无 lm_head LoRA 或短 prefill 都可能绕过，所以基础 smoke test 不足以发现。

验证方法：构造长 prompt + `return_logprob`/input logprobs，令 pruned token 数跨过 `SGLANG_LOGITS_PROCESSER_CHUNK_SIZE`；预期当前基线可在 `_prepare_lm_head_batch_info` 观察到 `NoneType.append`。修复后应验证每个 pass 的 `expected_tokens` 与实际 hidden-state chunk 一致。

## 排障入口矩阵

| 你看到的现象 | 第一断点 | 预期变量 |
|--------------|----------|----------|
| 请求直接 4xx | `TokenizerManager._validate_and_resolve_lora` | `enable_lora`、`obj.lora_path` |
| 动态 load 4xx/5xx | `TokenizerManager.load_lora_adapter` | `dp_size`、异常类型、`result.success` |
| adapter 加载后请求仍找不到 | `LoRARegistry.acquire` | `_registry` 是否含请求 name |
| batch 进不去 | `Scheduler._can_schedule_lora_req` | `running_loras`、`new_lora_set` |
| slot 满 | `LoRAMemoryPool.prepare_lora_batch` | `cur_uids`、`candidates` |
| 输出串 adapter | `Req.__init__` | `extra_key` 是否含 `lora_id` |
| kernel 读错 slot | `LoRAManager.prepare_lora_batch` | `uid_to_buffer_id`、`weight_indices` |
| MoE shape mismatch | `LoRAMemoryPool.__init__` | `moe_tp_size`、`moe_ep_size` |
| load/unload 后状态分叉 | control mixin + `LoRAManager` | registry/cache/CPU dict/pool uid 四方状态 |
| lm_head logprobs 分块崩溃 | Triton `_prepare_lm_head_batch_info` | `pass_segments`、局部/成员 list、`expected_tokens` |

---

## 运行验证

维护本文时，先用下面的命令确认 LoRA 排障入口仍在：

```powershell
rg -n "_validate_and_resolve_lora|LoRARegistry|load_lora_adapter|_can_schedule_lora_req|prepare_lora_batch|LoRAMemoryPool|moe_tp_size|strict_loading|lora_path" sglang/python/sglang/srt/managers/tokenizer_manager.py sglang/python/sglang/srt/managers/tokenizer_control_mixin.py sglang/python/sglang/srt/managers/scheduler.py sglang/python/sglang/srt/lora sglang/python/sglang/srt/managers/schedule_batch.py
```

预期信号：

- TokenizerManager 仍负责请求侧 LoRA 校验、路径解析和 registry acquire。
- Scheduler 仍负责 batch 准入时的 LoRA 约束。
- `lora_manager.py` 与 `mem_pool.py` 仍负责 adapter 加载、slot 分配、strict loading、MoE TP 规则。
- `schedule_batch.py` 仍能体现 `lora_id` 参与请求缓存隔离。
- 当前基线的 non-strict unknown weight、缺失 uid 默认 slot 0、lm_head pass list 三处边界应保留为显式检查项，直到 upstream 修复并有回归测试。

如果排障入口被挪到新组件，应先更新排障入口矩阵，再检查核心概念和数据流页是否仍准确。
