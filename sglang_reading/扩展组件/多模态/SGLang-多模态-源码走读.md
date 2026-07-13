---
title: "多模态 · 源码走读"
type: walkthrough
framework: sglang
topic: "多模态"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# 多模态 · 源码走读

## 长文读法

不要按文件树顺序读。本篇沿一项媒体的生命周期走：

```text
模型架构
→ Processor 选择
→ raw / preprocessed / precomputed 三路输入
→ prompt placeholder 展开
→ input_ids + MultimodalDataItem
→ TokenizerManager 发送
→ Scheduler IPC 重建与 hash/pad
→ ViT / Audio tower 或预计算 embedding
→ prefix cache
→ 可选 encoder disaggregation
```

每一节只回答一个设计判断，并给出它的适用边界。

## 1. Processor 是架构适配器，不是通用图片函数

启动时 `import_processors()` 触发各模型 processor 注册。TokenizerManager 还可导入 `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`，并以 `overwrite=True` 覆盖同名模型映射。

选择时先用 `hf_config.architectures` 匹配模型类名，再检查当前是否走 Transformers backend。后者只有在 processor 声明 `supports_transformers_backend` 时才允许返回。

```python
# 来源：python/sglang/srt/managers/multimodal_processor.py L44-L67
def get_mm_processor(
    hf_config,
    server_args: ServerArgs,
    processor,
    transport_mode,
    model_config=None,
    **kwargs,
) -> BaseMultimodalProcessor:
    model_impl = str(getattr(server_args, "model_impl", "auto")).lower()
    uses_transformers_backend = model_impl == "transformers"
    if model_impl == "auto" and model_config is not None:
        from sglang.srt.model_loader.utils import get_resolved_model_impl

        uses_transformers_backend = (
            get_resolved_model_impl(model_config) == ModelImpl.TRANSFORMERS
        )

    for model_cls, processor_cls in PROCESSOR_MAPPING.items():
        if model_cls.__name__ not in hf_config.architectures:
            continue
        if not uses_transformers_backend or getattr(
            processor_cls, "supports_transformers_backend", False
        ):
            return processor_cls(
```

### 为什么按类名匹配

HF config 提供的 architecture 常是字符串；注册表持有模型类。当前实现用 `model_cls.__name__` 对字符串列表匹配，使 processor 与模型实现保持松耦合。

### 失效边界

同名类来自不同包时，仅类名不能表达全部语义；外部覆盖必须由维护者确认行为兼容。注册成功也不代表 encoder disaggregation 支持，后者还有独立架构白名单。

## 2. TokenizerManager 即使跳过 tokenizer，也不跳过媒体

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L327-L350
    def init_tokenizer_and_processor(self):
        server_args = self.server_args

        # Initialize tokenizer and processor
        if self.model_config.is_multimodal:
            import_processors("sglang.srt.multimodal.processors")
            if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
                import_processors(mm_process_pkg, overwrite=True)
            _processor = _get_processor_wrapper(server_args)
            transport_mode = _determine_tensor_transport_mode(self.server_args)

            # We want to parallelize the image pre-processing so we create an executor for it
            # We create mm_processor for any skip_tokenizer_init to make sure we still encode
            # images even with skip_tokenizer_init=False.
            self.mm_processor = get_mm_processor(
                self.model_config.hf_config,
                server_args,
                _processor,
                transport_mode,
                model_config=self.model_config,
            )

            if server_args.skip_tokenizer_init:
                self.tokenizer = self.processor = None
```

`skip_tokenizer_init=True` 只要求调用方直接提供文本 `input_ids`。媒体仍要经过模型专用 processor，因为 grid、offset、pixel/audio feature 等契约不能由普通 tokenizer 替代。

## 3. 请求先形成文本骨架，再接入媒体

`_tokenize_one_request()` 按以下优先级形成文本侧输入：

1. `input_embeds`：要求关闭 radix cache；
2. `input_ids`：直接采用；
3. 文本：通过 tokenizer；
4. audio-only 多模态：允许先用空 list，等待 processor 生成。

随后它规范化 image/video/audio 为 list，并执行每请求数量限制。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/tokenizer_manager.py L793-L949
input_embeds / input_ids / text
→ contains_mm_input
→ normalize media to lists
→ _validate_mm_limits
→ local processor or remote encoder receiver
→ replace input_ids/token_type_ids from mm output
→ optional caller mm_hashes
→ optional SGLANG_MM_PRECOMPUTE_HASH
→ _validate_one_request
→ _create_tokenized_object
```

这个顺序揭示一个重要事实：长度校验发生在 processor 已可能展开媒体 placeholder 之后，所以它看到的是更接近实际 prefill 的 token 数；但自动截断只裁 input ids，后文会看到风险。

## 4. 三路输入如何收敛

### 4.1 Raw media

Processor 负责下载/解码、resize/抽帧、模型专用预处理，得到 `pixel_values`、audio features、grid 等。

### 4.2 已预处理 feature

调用方可复用 processor output，避免再次下载和预处理，但仍需视觉塔或音频 encoder 计算 embedding。

### 4.3 预计算 embedding

item 标记为 `PRECOMPUTED_EMBEDDING`，模型执行侧跳过 encoder，直接把 embedding 对齐到 placeholder span。它常见于 encoder disaggregation。

三路最终都必须给出同一组语义：媒体 modality、在 prompt 中的位置、内容身份，以及模型侧所需 metadata。

## 5. Processor 输出通常不是 visual embedding

`MultimodalDataItem` 同时容纳 `feature` 和 `precomputed_embeddings`，注释明确区分二者：

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L259-L268
    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Union[torch.Tensor, np.ndarray] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Model-specific data stored in a dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)
```

因此普通本地路径是：Processor 产生 encoder 输入，模型执行侧再跑 ViT/Audio tower。把 processor 写成“输出最终视觉 embedding”会混淆预处理和模型 forward 的边界。

## 6. 为什么分箱后仍能保持 prompt 顺序

通用结果容器按 IMAGE→VIDEO→AUDIO 整理：

```python
# 来源：python/sglang/srt/multimodal/processors/base_processor.py L72-L81
    def organize_results(self) -> List[Tuple[Modality, Any]]:
        """

        :return: a list of results, with their corresponding modalities
        """
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )
```

`build_input_ids()` 则先扫描 prompt 中特殊 token，记录全局 modality 序列，再维护 `img_idx`、`video_idx`、`audio_idx` 三个独立游标。于是两种顺序不矛盾：内部按类别存储，展开按 prompt 播放。

### 源码阅读抓手

遇到多图/多视频错位，不要只比较总数。应同时打印：

- `vision_start_indices`；
- `modality_list`；
- 三个 modality 游标；
- 每项 `grid_thw` 或 audio length；
- 生成的 offsets。

## 7. Qwen-VL 为什么必须单独读

`qwen_vl.py` 不是对基类的轻微包装，而是模型契约的集中地。它处理的典型问题包括：

- 图片像素预算和 fast/base image processor 选择；
- 视频采样、帧数、fps 与时间 metadata；
- `grid_thw` 与 spatial merge 后的 token 数；
- Qwen 系列 MRoPE position；
- image/video/audio 特殊 token 的模型差异；
- 部分 CPU/AMX 或设备专用预处理路径。

通用 processor 只保证框架级形状；Qwen-VL processor 决定“这份形状是否符合 Qwen-VL”。接入新模型时，应把模型专用约束放在对应 processor，而不是塞进基类。

## 8. `validate()` 仍是空壳

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L335-L338
    def validate(self):
        ...
        # TODO
```

这段极短代码比长篇注释更值得记住：当前基线没有一个统一入口证明 item 的 feature、offset、hash、grid 都自洽。严谨文档必须把验证责任描述为“分散式”，并在排障与实验里逐层验证。

## 9. hash、pad 与 RadixAttention

每项媒体先对 feature 或 precomputed embedding 做 hash，再派生词表外 pad value。若设置 `SGLANG_MM_SKIP_COMPUTE_HASH`，则使用随机 UUID，等价于放弃跨请求的稳定内容身份。

TokenizerManager 允许外部路由器传 `mm_hashes`：

- item 数不匹配：整组忽略并内部重算；
- 单项十六进制解析失败：该项回退内部重算；
- 合法：写入 item.hash，使后续 pad 与外部路由身份一致。

`SGLANG_MM_PRECOMPUTE_HASH` 只决定 pad 是在 tokenizer 阶段算，还是留给 Scheduler。它不改变最终身份语义。

## 10. Scheduler 不是被动收件人

`MultimodalInputs.from_processor_output()` 会重建 IPC、计算 hash/pad，并复制模型专用可选字段。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L505-L544
    def from_processor_output(obj: MultimodalProcessorOutput):
        mm_items = obj.mm_items
        assert isinstance(mm_items, list)
        mm_items = [item for item in mm_items if item.is_valid()]

        # try reconstructing from cuda-ipc
        reconstruct_device = None
        for mm_item in mm_items:
            if mm_item.has_cuda_ipc_proxy():
                if reconstruct_device is None:
                    reconstruct_device = torch.cuda.current_device()
                mm_item.reconstruct(reconstruct_device)

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)
```

“临时上 GPU 算 hash，随后回 CPU”说明 feature 的 device 位置是实现策略，不是语义身份。监控和断点应记录每一站的 device，而不是假定一次驻留后永不变化。

## 11. CUDA IPC 为什么不是零复制

producer 侧 proxy 保存 CUDA storage handle、shape、dtype、stride、offset 与共享同步区 metadata。consumer 打开 storage slice后，仍会创建新的目标 tensor 并 copy：

```python
# 来源：python/sglang/srt/utils/cuda_ipc_transport_utils.py L407-L430
    def _copy_slice_tensor_to_target(
        self,
        slice_tensor: torch.Tensor,
        rebuild_device: torch.device,
        recons_shape,
        recons_dtype,
    ):
        with torch.cuda.device(rebuild_device):
            reconstructed_tensor = torch.empty(
                recons_shape, dtype=recons_dtype, device=rebuild_device
            ).contiguous()
            reconstructed_tensor.view(torch.int8).view(-1).copy_(slice_tensor)

            open(SHM_LOCK_FILE, "a").close()
            # write the shm_sync_buffer with a file lock
            with open(SHM_LOCK_FILE, "w+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                sync_flag = self.get_sync_flag
                sync_flag += 1
                fcntl.flock(f, fcntl.LOCK_UN)

            self.close_shm()

        return reconstructed_tensor
```

共享计数的含义是“这个 consumer 已完成复制”。当预期 TP consumers 都完成后，producer pool 才能回收对应 chunk。

### pool 大小陷阱

每 worker 至少 128 MiB：

```python
# 来源：python/sglang/srt/multimodal/processors/base_processor.py L265-L269
            worker_num = self.server_args.tokenizer_worker_num
            per_worker_pool_size = max(
                MM_FEATURE_CACHE_SIZE // worker_num,
                128 * 1024 * 1024,
            )
```

所以配置的全局预算更像“参与计算的目标值”，不是硬总上限。

## 12. feature 驻留策略

`keep_mm_feature_on_device=False` 时，普通 processor 路径倾向把 feature 移回 CPU；CUDA IPC 路径会跳过那次 CPU move。开启该参数后，即使 pool 包装失败，也可能继续保留 CUDA tensor。

准确的判断表：

| IPC | keep on device | 可能结果 |
|---|---|---|
| 可用且入池 | 任意 | proxy 跨进程，consumer 重建并 device copy |
| pool 满/包装失败 | false | fallback 到 CPU feature |
| pool 满/包装失败 | true | fallback 保留 CUDA tensor |

实际分支仍应以当前函数为准；表格表达的是设计语义，不是替代源码。

## 13. ViT CUDA Graph：总长度不是完整布局

```python
# 来源：python/sglang/srt/multimodal/vit_cuda_graph_runner.py L115-L118
    def _get_graph_key(self, x_3d: torch.Tensor) -> int:
        # x_3d: [S, B, H], B=1, S as graph_key
        return x_3d.shape[0]
```

graph 首次 capture 时还接收 `cu_seqlens`、`cu_window_seqlens`、position embeddings 等。key 却只有 `S`。因此：

- 同 `S`、同分段布局：合理复用候选；
- 同 `S`、不同图片数量或窗口分段：可能复用旧 metadata；
- 当前代码没有以布局差异触发 eager fallback 的判断。

这是“形状稳定”与“语义布局稳定”不同的典型例子。

## 14. 自动截断为何跨层危险

Processor 已可能生成展开后的 input ids 和 offsets。之后 `_validate_one_request()` 若发现超长，会直接 `del input_ids[_max_req_len:]`。它没有同步更新 `mm_inputs`。

因此安全边界是：

- 纯文本尾部截断通常只影响文本；
- 截断点进入媒体 span，就可能使 offsets、feature 数和 token span 不一致；
- 当前空 `validate()` 不会统一捕获这种破坏。

生产建议是提前做多模态 token 预算，而不是依赖此开关兜底。

## 15. Encoder disaggregation 改变了什么

### 15.1 角色

- `encoder_only`：只加载 encoder/processor 相关权重与服务；
- `language_only`：语言侧可从远端 encoder 接收 embedding；
- 两者互斥；encoder-only 也不与 PD prefill/decode 混用。

### 15.2 传输 backend

| backend | embedding 到哪里 | 关键生命周期 |
|---|---|---|
| `zmq_to_tokenizer` | TokenizerManager | tokenizer 等待 embedding 后再发 Scheduler |
| `zmq_to_scheduler` | Scheduler endpoint | encoder 可等待 scheduler 注册 receive URL |
| `mooncake` | 远端 buffer/RDMA | `/encode` 可先回 metadata，`/send` 等 ready 后传输 |

ZMQ 路径把 embedding 转 CPU contiguous `TensorWrapper` 后 multipart 发送；它不是 GPU RDMA。Mooncake 则把元数据响应和真正数据 transfer 分成两个阶段。

### 15.3 Encoder batching 与缓存

image/audio 可跨请求 fusion；video 因每请求预处理 metadata 不同，不进入同样的 batch fusion。prefix MM cache 与 global cache 可跳过重复 ViT，但 TP rank 任一 cache hit 失败会广播 fallback mask，让所有 rank 一致重跑，避免分布式分歧。

### 15.4 DP dispatcher

DP encoder 要求 `dp_size > 1` 且 `tp_size == 1`。dispatcher 按最少 pending 加 round-robin 选 rank，并保存 `req_id → rank`，确保后续 `/send` 回到执行 `/encode` 的同一 worker。watchdog、result listener 超时和 stale mapping 清理共同防止死 worker 永久占住请求。

### 15.5 动态发现

`language_only` 可不提供静态 `encoder_urls`，由 bootstrap server 接受 encoder 注册。encoder 启动后在后台重试注册，不阻塞自身启动；退出时尝试注销。

## 16. 参数校验顺序为什么值得读

`ServerArgs.__post_init__()` 先做 `mm_process_config` 浅层结构校验，再加载模型并处理大量 backend/default，之后才执行 `_handle_encoder_disaggregation()`、tokenizer batching 和其他校验。

这意味着：

- `mm_process_config` 通过只说明形状像 `{image:{}, video:{}, audio:{}}`，不说明模型支持全部键；
- VLM 会根据 vision config 启发式降低 `mem_fraction_static`，给视觉处理留空间；
- `limit_mm_data_per_request` 的 modality key 校验更晚；
- tokenizer batch encode 与 dynamic tokenizer 互斥，并且 generation 多模态会额外拒绝 batch encode。

读参数时要问“用户输入值是什么、哪一步被自动改写、最终运行值是什么”，而不是只看 dataclass 默认值。

## 17. 运行验证

### 17.1 语法与入口

```powershell
@(
  'sglang/python/sglang/srt/managers/multimodal_processor.py',
  'sglang/python/sglang/srt/multimodal/processors/base_processor.py',
  'sglang/python/sglang/srt/multimodal/processors/qwen_vl.py',
  'sglang/python/sglang/srt/managers/schedule_batch.py',
  'sglang/python/sglang/srt/managers/tokenizer_manager.py',
  'sglang/python/sglang/srt/multimodal/customized_mm_processor_utils.py',
  'sglang/python/sglang/srt/multimodal/evs/evs_core.py',
  'sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py',
  'sglang/python/sglang/srt/utils/cuda_ipc_transport_utils.py',
  'sglang/python/sglang/srt/disaggregation/encode_server.py',
  'sglang/python/sglang/srt/server_args.py'
) | ForEach-Object { python -m py_compile $_ }
```

预期：11 个文件都能编译。该检查只证明语法，不证明运行语义。

### 17.2 静态语义实验

```powershell
rg -n "def validate\(|\.\.\.|TODO" sglang/python/sglang/srt/managers/schedule_batch.py
rg -n "del input_ids\[_max_req_len:\]" sglang/python/sglang/srt/managers/tokenizer_manager.py
rg -n "return x_3d.shape\[0\]" sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py
rg -n "torch.empty\(|copy_\(slice_tensor\)" sglang/python/sglang/srt/utils/cuda_ipc_transport_utils.py
```

预期：分别证实空校验、仅裁 token、graph key 仅 `S`、consumer 新分配并复制。

### 17.3 有 GPU 时的最小对照

1. 同一图片跑 CPU feature transport 与 CUDA IPC，比较 logits/embedding。
2. 构造同 `S` 不同分段布局，比较 ViT eager 与 graph。
3. 构造接近 context limit 的媒体请求，确认禁用 auto truncate 时得到明确错误。
4. 重复同一媒体，观察 hash/pad 与 prefix cache hit 是否稳定。

预期：transport 只改变性能不改变结果；若 graph 布局实验不一致，应立即禁用或修正 key；超长请求不应静默破坏 offset。

## 复盘

源码主线不是“图片经过哪些函数”，而是“谁在每一步拥有文本位置与媒体内容之间的对应关系”。Processor 建立它，TokenizerManager 搬运它，Scheduler重建并内容寻址，模型执行兑现它；IPC、graph、cache 和 encoder disaggregation 都只能优化这条契约，不能改写它。
