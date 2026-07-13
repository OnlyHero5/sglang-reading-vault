---
title: "ModelLoader · 排障指南"
type: troubleshooting
framework: sglang
topic: "ModelLoader"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# ModelLoader · 排障指南

本篇不是重复源码走读，而是给排障和改代码时的判断入口。先看症状属于哪本账：文件账、transport 账、参数账，还是加载后处理。

## 症状速查

| 现象 | 优先看 | 判断方式 |
|------|--------|----------|
| `Cannot find any model weights` | `_prepare_weights` | 文件后缀、allow pattern、safetensors index 过滤 |
| `Unexpected extra config keys` | `DefaultModelLoader.__init__` | 当前 loader 是否接受这个 extra config |
| 某个 TP rank OOM 或 shape mismatch | 当前 loader 的 rank-local 协议 | 是全量、presharded、iterator 内切片、rank state 还是 direct transfer |
| `Parameter not found in params_dict` | 模型类 `load_weights` | checkpoint name 是否被 remap、跳过或落空 |
| `remote_instance` 没走远端 | `ServerArgs._handle_load_format` | 配置不完整会回退到 `auto` |
| 量化模型加载后输出异常 | `load_weights_and_postprocess` | `quant_method.process_weights_after_loading` 是否执行 |
| 热更新失败后状态不清楚 | `update_weights_from_disk` | 所有原地更新都按可能部分写入处理；现有“rollback”不证明恢复旧 checkpoint |

## Q1：找不到权重时，先查文件模式，不要查模型类

`_prepare_weights` 会按 allow pattern 找文件。找到 safetensors 后还会用 index 过滤重复文件；找不到任何候选文件才抛 `Cannot find any model weights`。

```python
# 来源：python/sglang/srt/model_loader/loader.py L492-L520
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )
```

验证方法：在本地模型目录列出 `*.safetensors`、`model.safetensors.index.json`、`*.bin`、`*.pt`。如果 `load_format=safetensors` 但只有 `.bin`，错误会在文件账阶段出现，模型 `load_weights` 根本还没开始。

## Q2：`model_loader_extra_config` 不是全局扩展口

默认 loader 只接受两个 key：`enable_multithread_load` 和 `num_threads`。给 `DefaultModelLoader` 塞 remote、quant 或自定义字段会直接报错。

```python
# 来源：python/sglang/srt/model_loader/loader.py L392-L403
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )
```

判断方法：先确认 factory 实际返回哪个 loader，再确认该 loader 的 `__init__` 接受哪些 key。不要把 `model_loader_extra_config` 当作所有 loader 共享的自由字典。

## Q3：TP/rank-local 化到底在哪里发生？

没有一个对所有 loader 都成立的位置。普通 HF 全量 tensor + `RowParallelLinear` 的基线是在 parameter loader 中以 `start_idx = tp_rank * shard_size` narrow：

```python
# 来源：python/sglang/srt/layers/linear.py L1448-L1487
            and not use_bitsandbytes_4bit
            and not self.use_presharded_weights
        ):
            shard_size = param_data.shape[input_dim]
            start_idx = self.tp_rank * shard_size

            if _is_cpu:
                from sglang.srt.model_loader.weight_utils import (
                    narrow_padded_param_and_loaded_weight,
                )

                param_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                    param_data,
                    loaded_weight,
                    0,  # param_data_start
                    start_idx,
                    input_dim,
                    shard_size,
                )
            else:
                # Padding for special case like qwen2_5_VL's mlp which is not 8-aligned
                end_idx = start_idx + shard_size
                if end_idx > loaded_weight.shape[input_dim]:
                    loaded_weight = pad_or_narrow_weight(
                        loaded_weight, input_dim, start_idx, shard_size
                    )
                else:
                    loaded_weight = loaded_weight.narrow(
                        input_dim, start_idx, shard_size
                    )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert (
            param_data.shape == loaded_weight.shape
        ), f"{param_data.shape=} {loaded_weight.shape=}"
        param_data.copy_(loaded_weight)
```

源码条件同时给出了反例：BitsAndBytes 4bit 和 `use_presharded_weights` 会跳过二次 narrow。ShardedState 直接读 rank 文件；Remote KV 可调用 `weight_iterator(rank)`；BitsAndBytes 非预量化会在 iterator 内切片；RemoteInstance 直接广播/写入已初始化参数。

验证方法：先打印 loader 类型和 tensor 在进入模型前的 shape，再看 parameter 上的 `use_presharded_weights`/quant 属性。目标不是找到某个固定切片函数，而是证明该 tensor 恰好完成一次 rank-local 化。普通 linear 的 `self.tp_rank` 默认来自运行时 `get_parallel()`，不要误认为由 `LoadConfig.tp_rank` 注入。

## Q4：名字不匹配时，先读模型类 `load_weights`

Llama 类模型会改写 scale 名、跳过 rope cache、跳过部分 vision tower tensor、处理 tied embedding，并把 QKV/O gate 等 stacked 参数映射到融合参数。

```python
# 来源：python/sglang/srt/models/llama.py L641-L700
        for name, loaded_weight in weights:
            if name.endswith(".activation_scale"):
                name = name.replace(".activation_scale", ".input_scale")
            if name.endswith(".weight_scale_inv"):
                name = name.replace(".weight_scale_inv", ".weight_scale")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            # Handle FP8 kv-scale remapping
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip loading kv_scale from ckpts towards new design.
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")
```

判断方法：如果日志里有 `Parameter <name> not found in params_dict`，先把原始 checkpoint name 按这段规则手算一遍。若手算后仍不在 `params_dict`，再看模型结构是否和 checkpoint 架构一致。

## Q5：为什么用了 generator，CPU 内存仍然很高？

`_get_weights_iterator` 决定 safetensors、fastsafetensors、PT、NPCACHE 和多线程路径。无论走哪个 iterator，输出都仍是 `(name, tensor)`。

```python
# 来源：python/sglang/srt/model_loader/loader.py L541-L620
    def _get_weights_iterator(
        self, source: Source
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        use_multithread = extra_config.get("enable_multithread_load", True)
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt
        )

        if use_safetensors and source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files,
                hf_folder,
                "model.safetensors.index.json",
                source.model_config.hf_config,
            )

        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif use_safetensors:
            server_args = get_global_server_args()
            weight_loader_disable_mmap = server_args.weight_loader_disable_mmap
            weight_loader_prefetch = server_args.weight_loader_prefetch_checkpoints
            prefetch_num_threads = server_args.weight_loader_prefetch_num_threads
            weight_loader_drop_cache_after_load = (
                server_args.weight_loader_drop_cache_after_load
            )

            if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                )
            elif use_multithread:
                weights_iterator = buffered_multi_thread_safetensors_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                    disable_mmap=weight_loader_disable_mmap,
                    prefetch=weight_loader_prefetch,
                    prefetch_num_threads=prefetch_num_threads,
                    drop_cache_after_load=weight_loader_drop_cache_after_load,
                )
            else:
                weights_iterator = safetensors_weights_iterator(
                    hf_weights_files,
                    disable_mmap=weight_loader_disable_mmap,
                    prefetch=weight_loader_prefetch,
                    prefetch_num_threads=prefetch_num_threads,
                    drop_cache_after_load=weight_loader_drop_cache_after_load,
                )

        else:
            if use_multithread:
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(hf_weights_files)

        if self.load_config.draft_model_idx is not None:
            return self._filter_mtp_weights(
                weights_iterator, source.prefix, self.load_config.draft_model_idx
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
```

generator 只定义迭代接口，不保证常数内存。mmap safetensors 可按 key 获取；`disable_mmap` 整文件读取；buffered 多线程维持 `max_workers + 1` shard 窗口；PT 多线程会为全部文件提交 future；page-cache prefetch 又由本机 ranks 分摊后台读取。

验证方法：同时记录 shard size、`num_threads`、mmap、prefetch、drop-cache、RSS 与 page cache。多线程通常不改变名字语义，但会显著改变在途 shard 数、完成顺序和 CPU/page-cache 峰值。

## Q6：参数已 copy，为什么模型仍不可执行？

权重 copy 完不等于量化模型可执行。loader 会遍历所有 module，调用量化方法的 `process_weights_after_loading`。

```python
# 来源：python/sglang/srt/model_loader/loader.py L812-L821
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)
```

这张卡证明默认路线的 quant process。还要区分模型级 `post_load_weights`：绕过 `model.load_weights` 的 Dummy、ShardedState、RemoteInstance、Remote KV 等路线需要显式调用。前者常做 repack/quantize/layout，后者补模型派生状态；KV scale 又是第三条完成轨。

验证方法：量化模型异常时，分别确认 module quant process、model post-load fixup、KV scale 是否由当前 loader 路线负责。CPU offload 还要观察 `device_loading_context` 在 parameter 被替换或改变 storage 后如何恢复。

## Q7：热更新失败的风险取决于入口

从磁盘热更新失败时，日志声称回滚原权重，但代码在更新前已经把 `self.model_config.model_path` 改成目标路径；异常后又用同一个目标配置重建 iterator。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L1854-L1858
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message
```

因此这段只能证明“再次尝试装载目标 checkpoint”，不能证明恢复旧权重。第一次 `model.load_weights` 还可能已经部分改写参数。分布式/tensor 更新同样没有事务提交。

生产处置：失败实例应先隔离；若要求原子切换，应保留旧模型副本/备用实例或显式版本化恢复。不要根据返回字符串继续把该实例视为旧权重一致状态。

## 小结

排查 ModelLoader 时按顺序问五个问题：

1. `ServerArgs` 最终把 `load_format` 改成了什么。
2. `_prepare_weights` 找到的是哪些文件。
3. iterator 产出的 `(name, tensor)` 是否符合模型类 `load_weights` 的名字规则。
4. 当前路线在哪里完成 rank-local 化，是否漏切或重切。
5. quant process、model fixup、KV scale 和热更新失败一致性是否闭合。
