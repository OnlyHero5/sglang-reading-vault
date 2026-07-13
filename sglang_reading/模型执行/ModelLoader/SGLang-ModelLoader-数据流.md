---
title: "ModelLoader · 数据流"
type: dataflow
framework: sglang
topic: "ModelLoader"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/dataflow
  - source-reading
updated: 2026-07-11
---
# ModelLoader · 数据流

## 你为什么要读

本篇解决一个边界问题：冷启动加载、运行时换权重、远端实例加载看起来都在“灌权重”，但它们穿过的对象不同。读完后应能判断一次异常发生在配置改写、权重 transport、iterator、模型名字映射，还是参数写入阶段。

## 数据流总图

```mermaid
flowchart LR
  A[ServerArgs] --> B[load_format 规范化]
  B --> C[LoadConfig]
  C --> D[get_model_loader]
  D --> E{loader 路线}
  E --> F[DefaultModelLoader<br/>本地或 HF checkpoint]
  E --> G[RemoteModelLoader<br/>远程存储 connector]
  E --> H[RemoteInstanceModelLoader<br/>远端实例参数]
  F --> I[(name, tensor) iterator]
  G --> I
  I --> K{写入协议}
  K --> L[model.load_weights<br/>param.weight_loader]
  K --> N[rank-local state dict<br/>direct copy]
  H --> O[parameter broadcast / address transfer]
  L --> J[rank-local Parameter]
  N --> J
  O --> J
  J --> M[quant process / post-load fixup]
```

这张图的关键是区分两类流：

| 流 | 入口 | 经过 loader factory 吗 | 最终写入 |
|----|------|------------------------|----------|
| 冷启动 | `ModelRunner.load_model` | 是 | `model.load_weights` 或特殊 loader 的等价路径 |
| 从磁盘热更新 | `ModelRunner.update_weights_from_disk` | 是，但目前只接受 `DefaultModelLoader` | `load_weights_and_postprocess` |
| 分布式热更新 | `update_weights_from_distributed` | 否 | `model.load_weights` |
| tensor 热更新 | `update_weights_from_tensor` | 否 | `model.load_weights`、direct loader 或自定义 loader |
| 远端实例冷启动 | `RemoteInstanceModelLoader.load_model` | 是 | 广播到已初始化参数，或由 backend loader 填充 |

## 1. ServerArgs 会先改写加载路线

读加载问题不能只看用户传入的 `--load-format`。`ServerArgs` 会根据模型路径和远端配置把 `auto` 改成 `gguf`、`mistral`、`remote` 或 `runai_streamer`，remote instance 配置不完整时还会回退到 `auto`。

```python
# 来源：python/sglang/srt/server_args.py L5903-L5918
    def _handle_load_format(self):
        if (
            self.load_format == "auto" or self.load_format == "gguf"
        ) and check_gguf_file(self.model_path):
            self.quantization = self.load_format = "gguf"

        if self.load_format == "auto" and self._is_mistral_native_format():
            self.load_format = "mistral"
            logger.info(
                "Detected Mistral native format checkpoint, setting load_format='mistral'"
            )

        if is_runai_obj_uri(self.model_path):
            self.load_format = "runai_streamer"
        elif is_remote_url(self.model_path):
            self.load_format = "remote"
```

```python
# 来源：python/sglang/srt/server_args.py L5923-L5947
        if self.load_format == "remote_instance":
            if self.remote_instance_weight_loader_backend != "modelexpress" and (
                self.remote_instance_weight_loader_seed_instance_ip is None
                or self.remote_instance_weight_loader_seed_instance_service_port is None
            ):
                logger.warning(
                    "Fallback load_format to 'auto' due to incomplete remote instance weight loader settings."
                )
                self.load_format = "auto"
            elif (
                self.remote_instance_weight_loader_send_weights_group_ports is None
                and self.remote_instance_weight_loader_backend == "nccl"
            ):
                logger.warning(
                    "Fallback load_format to 'auto' due to incomplete remote instance weight loader NCCL group ports settings."
                )
                self.load_format = "auto"
            elif (
                self.remote_instance_weight_loader_backend == "transfer_engine"
                and not self.validate_transfer_engine()
            ):
                logger.warning(
                    "Fallback load_format to 'auto' due to 'transfer_engine' backend is not supported."
                )
                self.load_format = "auto"
```

排障时的第一步应是确认最终 `LoadConfig.load_format`，而不是只确认命令行参数。很多“为什么没走 remote instance”的问题在这里已经变成 `auto`。

## 2. 冷启动的主对象是 LoadConfig 和 loader

冷启动时，`ModelRunner` 只把加载事实表交给 loader。loader 返回的是已经装好权重的模型对象，随后 `ModelRunner` 继续做执行态初始化。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L1421-L1437
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
```

冷启动数据流里，`tp_rank` 从这里进入 `LoadConfig`，主要供 remote-instance 等 loader 使用。普通 linear parameter loader 的 `self.tp_rank` 默认来自运行时并行上下文。`_prepare_weights` 还可能按 TP rank 错峰调整 shard 文件读取顺序，但那只是 I/O 顺序，不是 tensor 切片。

## 3. 从磁盘热更新复用默认 loader，但不重建模型

运行时从磁盘换权重时，代码会创建新的 `LoadConfig` 和 loader，但显式要求 loader 是 `DefaultModelLoader`。它复用 `_get_weights_iterator`，然后对现有 `self.model` 调 `load_weights_and_postprocess`。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L1804-L1840
    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model
```

这里的设计压力是一致性：热更新不重新走 `_initialize_model`，所以新 checkpoint 的结构必须能被现有模型参数接住。更危险的是它原地逐参数写入，没有事务边界；失败时模型可能已经部分变更。

当前异常分支日志称“Rolling back to original weights”，但代码已经把 `self.model_config.model_path` 改成目标路径，随后用同一个目标配置重新创建 iterator。这只能再次尝试装载目标 checkpoint，不能证明恢复了旧权重。生产系统若需要原子切换，应在外层保留旧 checkpoint/备用实例或另行实现版本化恢复，不能依赖这条日志承诺。

## 4. 分布式和 tensor 热更新绕过文件层

分布式热更新直接广播 tensor，然后调用 `model.load_weights`。当 `load_format == "flattened_bucket"` 时，它先把多个 tensor 打平成一个连续 bucket 广播，再重建名字和 shape。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L2057-L2082
        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."
```

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L2093-L2114
    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
```

tensor 热更新也是同一个参数写入协议，只是入口来自序列化 payload，而不是 checkpoint 文件。

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L2124-L2153
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"
```

因此，热更新的核心不是 `DefaultModelLoader`，而是“transport 产出的名字、shape、dtype 是否与现有参数协议一致，以及失败时如何处理已发生的部分写入”。

## 5. FlattenedTensorBucket 是零拷贝 view 重建协议，不是模型语义

`FlattenedTensorBucket` 保存的是名字、shape、dtype 和 byte 范围。它提高传输效率，但重建后仍然回到 `(name, tensor)`。

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L53-L72
            for i, (name, tensor) in enumerate(named_tensors):
                flattened = tensor.flatten().view(torch.uint8)
                flattened_tensors[i] = flattened

                # Store metadata

                numel = flattened.numel()
                metadata_obj = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                self.metadata[i] = metadata_obj
                current_idx += numel

            # Concatenate all flattened tensors
            self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
```

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L90-L107
    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # preallocate the result list
        reconstructed = [None] * len(self.metadata)

        for i, meta in enumerate(self.metadata):
            tensor = (
                self.flattened_tensor[meta.start_idx : meta.end_idx]
                .view(meta.dtype)
                .reshape(meta.shape)
            )

            reconstructed[i] = (meta.name, tensor)

        return reconstructed
```

重建 tensor 是对同一个 uint8 flattened storage 的 slice/view/reshape，不是逐 tensor clone。排障时既要检查 metadata 的 byte offset/dtype/shape，也要保证 flattened storage 在消费完成前存活。它仍不是新模型语义；重建后回到 model/direct loader 协议。

## 6. 远端存储和远端实例不是一回事

`RemoteModelLoader` 从远程存储 connector 读 checkpoint，再进入模型写入流程。它的关键分支是 connector 类型：KV 或 FS。

```python
# 来源：python/sglang/srt/model_loader/loader.py L2517-L2548
        start = time.perf_counter()
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        model_weights = model_config.model_path
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)

            with create_remote_connector(
                model_weights, device=device_config.device
            ) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.KV:
                    self._load_model_from_remote_kv(model, model_config, client)
                elif connector_type == ConnectorType.FS:
                    self._load_model_from_remote_fs(
                        model, client, model_config, device_config
                    )

        end = time.perf_counter()
        logger.info("Loaded weights from remote storage in %.2f seconds.", end - start)
        return model.eval()
```

`RemoteInstanceModelLoader` 从另一个 SGLang 实例取已经服务态的参数。NCCL backend 会为每个 rank 构造 `instance://ip:port`，然后按参数广播。

```python
# 来源：python/sglang/srt/model_loader/loader.py L2228-L2243
        if (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            model_weights = f"instance://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_send_weights_group_ports[load_config.tp_rank]}"
            with create_remote_connector(model_weights, device_config.device) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.INSTANCE:
                    self.load_model_from_remote_instance_by_nccl(
                        model, client, model_config, device_config
                    )
                else:
                    raise ValueError(
                        f"Unsupported connector type {connector_type} for "
                        f"remote tensor model loading."
                    )
```

这两条路线的共同点是最终都要交付 `model.eval()`；中间协议却不同。Remote FS 走 `model.load_weights`；Remote KV 按 rank iterator 直接 copy state dict；RemoteInstance NCCL 按 `model.named_parameters()` 顺序广播，TransferEngine 按参数名校验 numel/element size 后写入注册地址。后二者绕过模型名字 remap，因此源/目标模型参数集合、顺序或地址 metadata 必须严格一致。

## 7. Layered loader 改变峰值显存边界

`LayeredModelLoader` 不是新的 checkpoint 语义。它把模型先建在 `meta` device，再逐模块 materialize 和灌权重，用时间换峰值内存。

```python
# 来源：python/sglang/srt/model_loader/loader.py L846-L864
        with set_default_torch_dtype(model_config.dtype):
            # Create model on meta device
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            # Check model's layered load support
            if not hasattr(model, "load_weights_to_module"):
                raise ValueError(
                    "LayeredModelLoader requires the model to have a "
                    "`load_weights_to_module` method. "
                    f"{model_config.model_path} does not support it."
                )

            # Get all weights from disk
            weights = self._get_all_weights(model_config, model)
```

如果 layered load 失败，先看模型类是否提供 `load_weights_to_module`，再看具体模块是否能从 `meta` materialize 到目标设备。注意它把同一个 generator 递归传给各模块，模型方法必须以单向消费方式正确分派权重；这不是简单地把默认 loader 自动切成逐层模式。

## 运行验证

| 现象 | 验证入口 | 预期判断 |
|------|----------|----------|
| `remote_instance` 没生效 | 搜日志里的 fallback warning，或断点 `ServerArgs._handle_load_format` | 配置不完整会在进入 `LoadConfig` 前回退到 `auto` |
| 从磁盘热更新失败 | 断点 `update_weights_from_disk` 的 `get_weight_iter` 和 `model_load_weights` | iterator 构造失败与模型写入失败会返回不同消息 |
| 分布式热更新 shape mismatch | 记录 `names/dtypes/shapes` 与 `params_dict` | transport 已完成，错误在现有模型写入协议；失败后按“可能部分写入”处理 |
| bucket 更新后输出异常 | 对比 bucket `metadata` 与重建后的 tensor shape/dtype | bucket 只负责还原 `(name, tensor)`，不会修正模型语义 |
| layered load 报不支持 | 检查模型类是否有 `load_weights_to_module` | loader 需要模型类提供逐模块灌权重入口 |

## 复盘

ModelLoader 的交互边界可以压成三句话：

- 冷启动先看 `ServerArgs` 到 `LoadConfig` 的事实表，再看 loader factory。
- `(name, tensor)` 是常见接口，不是所有路线必经点；rank-local state dict 和 parameter-address transfer 可以绕过模型 remap。
- 热更新、远端加载、layered load 改变 transport、rank-local 化和内存策略，但不能绕过参数集合、shape、dtype、完成动作与失败一致性这些不变量。
