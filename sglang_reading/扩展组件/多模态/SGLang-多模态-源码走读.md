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
updated: 2026-07-10
---
# 多模态 · 源码走读

> 走读主线：multimodal manager 先把模型架构映射到 processor；processor 基类负责媒体加载、特殊 token 识别、占位符展开和 `mm_items` 组装；模型专用 processor 再补齐图像/视频预处理、MRoPE 等模型契约；ViT CUDA Graph 与 EVS 分别处理视觉编码加速和长视频 token 裁剪。

## 长文读法

这篇不要按文件清单硬读，而要按一次多模态请求的形态变化读：`hf_config.architectures → processor → raw media / processor output / precomputed embedding → input_ids + mm_items → scheduler → 视觉优化`。读源码时先判断当前问题卡在哪个边界，再进入对应小节。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 判断模型为什么找不到 processor | 1.1、1.2 | `PROCESSOR_MAPPING` 只按模型类名匹配，Transformers backend 还有兼容性闸门 |
| 排查图片、视频、音频占位符对不上 | 2.2、3.1、3.2 | 文本特殊 token、媒体列表顺序、展开后的 token span 必须同时对齐 |
| 理解 `mm_items` 为什么能跨 tokenizer/scheduler 边界 | 2.1、3.3 到 3.6 | raw media、processor output 和预计算 embedding 最终都收敛成 `MultimodalDataItem` |
| 定位 Qwen-VL token 数、帧数或 MRoPE 异常 | 4.1 到 4.4 | 像素预算、抽帧、grid 和位置编码是模型专用契约，不属于通用 processor |
| 排查视觉性能或长视频 token 爆炸 | 5.1、5.2 | ViT CUDA Graph 只复用稳定 shape 的视觉 block，EVS 只负责长视频 token 裁剪 |
| 改代码前做回归确认 | 6 | 先用静态检索确认六个边界仍存在，再更新本文判断 |

读完整篇后应该能回答三个问题：一个新 VLM 怎么接入 processor；一个带媒体请求如何从 prompt 占位符变成 scheduler 可消费的 `MultimodalProcessorOutput`；视觉侧性能优化为什么不能混进通用数据契约里。

---

## 1. Processor 注册与选择

### 1.1 `import_processors` 以包扫描维护架构到 processor 的映射

问题与约束：
- 多模态模型数量多，新增模型应该只增加 processor 类，不应该在统一入口里手写大量 if/else。
- 某个 processor 模块导入失败不能拖垮整个 serving 启动，否则可选依赖会变成全局依赖。

设计选择：
- 启动时扫描目标包下的模块，收集定义在模块内且继承 `BaseMultimodalProcessor` 的类，再把类声明的 `models` 注册到全局 `PROCESSOR_MAPPING`。

**读法：**
`import_processors` 把“发现 processor”和“选择 processor”拆开。它只做包扫描、类筛选和映射注册；模块导入异常被记录为 warning 后跳过，`overwrite=True` 时允许按模型类名替换已有注册项。

来源：python/sglang/srt/managers/multimodal_processor.py L16-L41

**源码锚点：**
```python
def import_processors(package_name: str, overwrite: bool = False):
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)
            except Exception as e:
                logger.warning(f"Ignore import error when loading {name}: {e}")
                continue
            all_members = inspect.getmembers(module, inspect.isclass)
            classes = [
                member
                for name, member in all_members
                if member.__module__ == module.__name__
            ]
            for cls in (
                cls for cls in classes if issubclass(cls, BaseMultimodalProcessor)
            ):
                assert hasattr(cls, "models")
                for arch in getattr(cls, "models"):
                    if overwrite:
                        for model_cls, processor_cls in PROCESSOR_MAPPING.items():
                            if model_cls.__name__ == arch.__name__:
                                del PROCESSOR_MAPPING[model_cls]
                                break
                    PROCESSOR_MAPPING[arch] = cls
```

代码逻辑：
- 导入 processor 包，并枚举包下非 package 模块。
- 每个模块导入失败时只 warning 并继续。
- 只保留当前模块自己定义的 class，避免把 import 进来的基类重复注册。
- 对每个 `BaseMultimodalProcessor` 子类，读取 `models` 并写入 `PROCESSOR_MAPPING`。

为什么这样写：
- processor 的扩展点是 class-level `models`，新模型只需要在子类声明支持的架构。
- import 失败不阻断启动，可以让缺少可选视觉、视频或音频依赖的环境仍服务其他模型。

不变量与失败模式：
- processor 子类必须有 `models` 属性，否则 `assert hasattr(cls, "models")` 会失败。
- `PROCESSOR_MAPPING` 的 key 是模型类对象；若两个模块注册同名架构，只有 `overwrite=True` 才会先删除旧项。

**要点：**
这里的核心不是自动导入本身，而是把多模态模型支持做成“声明式插件表”。

### 1.2 `get_mm_processor` 根据 backend 兼容性选择原生或 Transformers 路径

问题与约束：
- 同一模型可能既能走 SGLang 原生实现，也可能在 `model_impl=transformers` 下运行。
- 如果 processor 不支持 Transformers backend，错误地复用原生 processor 会破坏模型调用契约。

设计选择：
- 先解析 `server_args.model_impl` 和 `model_config`，再用 `hf_config.architectures` 匹配注册表；只有原生路径或显式声明 `supports_transformers_backend` 的 processor 才能返回。

**读法：**
`get_mm_processor` 的选择逻辑分三层：先判断是否使用 Transformers backend；再按架构名遍历注册表；若没有可用专用 processor 且 backend 是 Transformers，则退到 `TransformersAutoMultimodalProcessor`；否则抛出包含已注册架构名的错误。

来源：python/sglang/srt/managers/multimodal_processor.py L44-L83

**源码锚点：**
```python
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
            hf_config, server_args, processor, transport_mode, **kwargs
        )

if uses_transformers_backend:
    from sglang.srt.multimodal.processors.transformers_auto import (
        TransformersAutoMultimodalProcessor,
    )

    return TransformersAutoMultimodalProcessor(
        hf_config, server_args, processor, transport_mode, **kwargs
    )
```

代码逻辑：
- `model_impl=auto` 时用 model loader 的解析结果判断实际 backend。
- 注册表匹配依赖 `hf_config.architectures` 中的模型类名。
- Transformers backend 下，专用 processor 必须显式声明兼容。
- 无专用 processor 时，Transformers backend 进入自动 processor 回退。

为什么这样写：
- 多模态预处理和模型 forward 的输入字段强绑定；backend 不同，字段语义和调用路径可能不同。
- 自动回退只对 Transformers backend 开放，避免原生路径悄悄使用性能和字段契约都不同的 HF 默认逻辑。

不变量与失败模式：
- `hf_config.architectures` 必须包含模型类名，否则无法命中注册表。
- 非 Transformers backend 且无注册 processor 时会抛 `ValueError`，部署新 VLM 时通常先从这里定位。

**要点：**
这个函数是多模态入口的“兼容性闸门”：先保证 processor 和 backend 匹配，再谈后续媒体处理。

## 2. 基类数据契约

### 2.1 `BaseMultiModalProcessorOutput` 保存加载后的媒体和可选原始 token

问题与约束：
- 请求可能带图片、视频、音频，也可能已经离线预处理成 tensor 或 input ids。
- 后续组合逻辑需要保持媒体出现顺序，同时又要按 modality 分派给不同处理器路径。

设计选择：
- 用轻量 dataclass 保存 `input_text`、可选 `input_ids` 和三类媒体列表，并提供 `organize_results()` 统一展开为 `(Modality, data)` 序列。

**读法：**
这个结构处在“加载媒体”和“调用 HF processor/构造 mm_items”之间。它不承诺媒体已经变成特征，只保证文本、原始 token 和各 modality 的已加载对象可以被下一阶段统一消费。

来源：python/sglang/srt/multimodal/processors/base_processor.py L48-L81

**源码锚点：**
```python
@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    input_text: str
    input_ids: Optional[Union[List[int], torch.Tensor]] = None
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )
    videos: Optional[list[Union[torch.Tensor, dict]]] = dataclasses.field(
        default_factory=list
    )
    audios: Optional[list[Union[np.ndarray, dict]]] = dataclasses.field(
        default_factory=list
    )

    def organize_results(self) -> List[Tuple[Modality, Any]]:
        return (
            [(Modality.IMAGE, data) for data in self.images]
            + [(Modality.VIDEO, data) for data in self.videos]
            + [(Modality.AUDIO, data) for data in self.audios]
        )
```

代码逻辑：
- `input_text` 保存展开或重建后的文本。
- `input_ids` 允许上游传入预 tokenized prompt。
- 三个列表分别保存 image、video、audio 的加载结果或预处理 dict。
- `organize_results()` 把三类列表合并为带 modality 标签的列表。

为什么这样写：
- 下一阶段只关心“这个对象属于哪种 modality”，不应该重复判断它来自哪个字段。
- 预处理输入可以携带 `input_ids`，避免强制 decode 再 tokenize。

不变量与失败模式：
- 列表顺序必须与 prompt 中对应特殊 token 的顺序能匹配，否则后续 offset 和特征会错位。
- `dict` 项需要后续通过 `format` 或字段名识别；字段缺失会在收集 `mm_items` 时暴露。

**要点：**
它是多模态流水线中最早的统一容器，把“原始媒体形态很多”收敛成“带 modality 的对象流”。

### 2.2 `MultimodalSpecialTokens` 统一 token、token id 和正则匹配

问题与约束：
- 不同 VLM 的图片、视频、音频占位符可能是单 token，也可能是展开后的 token 序列。
- 加载媒体时既要按文本切分特殊 token，也要在 input ids 中根据 modality 找 token id。

设计选择：
- 用 `MultimodalSpecialTokens` 同时保存 token 字符串、token id 和 modality 正则，并在 `build()` 中补齐字符串、正则和组合正则。

**读法：**
这个类把多模态占位符抽象成一个可查询对象：文本阶段用 `combined_regex` split prompt，收集阶段用 `get_modality_of_token` 判断片段属于哪种媒体，offset 阶段用 `get_token_id_by_modality` 找对应 token id。

来源：python/sglang/srt/multimodal/processors/base_processor.py L84-L177

**源码锚点：**
```python
@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[str, List[str]]] = None
    video_token: Optional[Union[str, List[str]]] = None
    audio_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None
    combined_regex: Optional[re.Pattern] = None

    def build(self, processor):
        self.convert_to_strs(processor)
        self.parse_regex()
        self.get_combined_regex()
        return self

    def get_token_id_by_modality(self, modality: Modality) -> Optional[int]:
        return {
            Modality.IMAGE: self.image_token_id,
            Modality.VIDEO: self.video_token_id,
            Modality.AUDIO: self.audio_token_id,
        }.get(modality)
```

代码逻辑：
- `build()` 将 token id 转回 token 字符串，并补齐默认正则。
- `get_modality_of_token()` 先查精确 token，再查 modality regex。
- `get_combined_regex()` 把所有存在的 modality regex 合并成一个 split pattern。
- `get_token_id_by_modality()` 为 offset 计算提供反向映射。

为什么这样写：
- 文本切分、token id 替换、modality 判断使用同一份配置，避免每个 processor 重复维护占位符规则。
- regex 支持展开后的图片 token 串，例如 Qwen-VL 的 vision start/pad/end 组合。

不变量与失败模式：
- 至少要有一个 modality token 或 regex；否则组合正则会没有有效模式。
- token id 和 token 字符串必须对应同一 tokenizer，否则文本 split 和 ids offset 会出现不一致。

**要点：**
这是多模态 prompt 解析的中心结构，后面的 fast path、legacy path 和 offset 计算都依赖它。

### 2.3 `BaseMultimodalProcessor.__init__` 固定配置、执行器和字段归属表

问题与约束：
- 媒体加载包含 I/O、CPU 预处理和可选 CUDA IPC，不能阻塞 tokenizer event loop。
- HF processor 输出字段很多，后续要把字段归属到 image/video/audio 的 `MultimodalDataItem`。

设计选择：
- 基类构造时保存 server 配置、拆分 image/video/audio 配置，创建 I/O 与 CPU executor，并维护字段名到 modality 的映射表。

**读法：**
构造函数把 processor 子类共享的运行环境一次性准备好：解析 tokenizer，设置媒体处理配置，创建线程/进程池，维护 `ATTR_NAME_TO_MODALITY` 与 `FEATURE_NAMES`，并在启用 CUDA IPC 时给 tokenizer worker 分摊特征缓存池。

来源：python/sglang/srt/multimodal/processors/base_processor.py L180-L282

**源码锚点：**
```python
class BaseMultimodalProcessor(ABC):
    models = []
    gpu_image_decode = True

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode

        mm_process_config = self.server_args.mm_process_config
        self.image_config = mm_process_config.get("image", {})
        self.video_config = mm_process_config.get("video", {})
        self.audio_config = mm_process_config.get("audio", {})

        if hasattr(self._processor, "tokenizer"):
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = self._processor

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )
```

代码逻辑：
- 保存 config、processor、server args 和 transport mode。
- 从 `mm_process_config` 拆出三类 modality 的配置。
- 兼容直接传 tokenizer 或带 tokenizer 的 processor。
- 建立 I/O thread pool 和 CPU process pool。
- 后续代码继续定义字段归属表、feature 字段名以及可选 CUDA IPC pool。

为什么这样写：
- I/O 与 CPU 预处理被隔离到 executor，避免媒体解码拖慢请求主路径。
- 字段归属表集中在基类，模型子类返回 HF processor output 时不必重复拆字段。

不变量与失败模式：
- `server_args.mm_process_config` 应该是按 modality 分组的 dict。
- Windows 或非 fork 环境下进程池上下文可能需要特别注意；源码这里明确使用 `mp.get_context("fork")`。
- CUDA IPC pool 的每 worker 分配依赖 `tokenizer_worker_num`，配置异常会影响特征缓存容量。

**要点：**
基类构造函数把“模型无关的多模态运行时”固定下来，子类主要补模型特有 token 和处理逻辑。

## 3. Prompt、媒体和 `mm_items` 的组合

### 3.1 `load_mm_data` 在 fast path 与 legacy path 之间做一致性分流

问题与约束：
- Prompt 中的多模态特殊 token 数量必须和实际传入媒体数量一致。
- 离线预处理数据可能已经带 `input_ids` 和特征，不能再强行 decode 或重新加载。

设计选择：
- 先验证媒体数据，再对“预 tokenized 且媒体都已预处理”的请求直接返回；否则用组合正则统计 prompt 中 modality token 数，数量不一致或需要动态帧展开时转 legacy path。

**读法：**
`load_mm_data` 是媒体加载的路由层。它优先保护预处理输入的零拷贝/少处理路径；对普通文本 prompt，则通过 `MultimodalSpecialTokens` split 文本并统计 image/video/audio token，只有 token 数和媒体数完全对齐时才进入 fast load。

来源：python/sglang/srt/multimodal/processors/base_processor.py L785-L864

**源码锚点：**
```python
BaseMultimodalProcessor.validate_mm_data(image_data, video_data, audio_data)

input_ids = prompt if isinstance(prompt, list) else None
if input_ids is not None and self._all_mm_data_is_preprocessed(
    image_data, video_data, audio_data
):
    return BaseMultiModalProcessorOutput(
        input_text="",
        input_ids=input_ids,
        images=list(image_data or []),
        videos=list(video_data or []),
        audios=list(audio_data or []),
    )

multimodal_tokens_pattern = multimodal_tokens.get_combined_regex()
if isinstance(prompt, list) and return_text:
    assert len(prompt) and isinstance(prompt[0], int)
    prompt = self._tokenizer.decode(prompt)

text_parts = re.split(multimodal_tokens_pattern, prompt)
cnt = {Modality.IMAGE: 0, Modality.VIDEO: 0, Modality.AUDIO: 0}
for text_part in text_parts:
    modality = multimodal_tokens.get_modality_of_token(text_part)
    if modality is not None:
        cnt[modality] += 1
```

代码逻辑：
- 校验媒体输入格式。
- 如果 prompt 已是 ids 且所有媒体都是预处理数据，直接构造 `BaseMultiModalProcessorOutput`。
- 将 ids prompt decode 成文本，或直接使用字符串 prompt。
- 用组合正则拆分文本，统计每种 modality 的占位符数量。
- 数量不匹配、跳过 tokenizer 初始化或动态帧展开时转 legacy path，否则进 fast path。

为什么这样写：
- 预处理输入通常来自离线 engine API，重复 tokenize 或解码会改变用户给定 token。
- fast path 只在 token 数和媒体数完全对齐时成立，避免图片/视频错配。

不变量与失败模式：
- `prompt` 最终必须是字符串；若 list prompt 为空或元素不是 int，会触发断言。
- prompt 中 token 数和媒体列表长度不一致时不会走 fast path。
- `support_dynamic_frame_expansion` 打开后，普通数量对齐也必须走 legacy path。

**要点：**
这里的关键判断是“是否能确定 prompt 占位符与媒体列表一一对应”。

### 3.2 `build_input_ids` 把一个占位符扩展成真实视觉/音频 token span

问题与约束：
- 模型 forward 需要的不是一个图片 token，而是按 grid 或音频长度展开后的连续 token span。
- scheduler 需要知道每个媒体对象在 `input_ids` 中的 offset，以便替换 pad value 和绑定特征。

设计选择：
- 基类按 prompt 中 image/video/audio token 的出现顺序扫描，依据 `grid_thw` 或 `audio_seq_lens` 计算 token 数，将占位符替换为重复的 modality token id，并记录 `(start, end)` offset。

**读法：**
`build_input_ids` 是从 prompt-level 占位符到 model-level token span 的转换点。它不处理特征张量，只根据模型输出的 grid/长度信息重写 `input_ids`，并返回保持出现顺序的 `modality_list`。

来源：python/sglang/srt/multimodal/processors/base_processor.py L295-L360

**源码锚点：**
```python
if not isinstance(prompt, list):
    prompt = self._tokenizer.encode(prompt)

img_token_id = getattr(self, "IM_TOKEN_ID", None)
video_token_id = getattr(self, "VIDEO_TOKEN_ID", None)
audio_token_id = getattr(self, "audio_token_id", None)
spatial_merge_size = getattr(self, "spatial_merge_size", 1)

vision_start_indices = []
for i in range(len(prompt) - 1):
    if img_token_id is not None and prompt[i + 1] == img_token_id:
        vision_start_indices.append((i, Modality.IMAGE))
    elif video_token_id is not None and prompt[i + 1] == video_token_id:
        vision_start_indices.append((i, Modality.VIDEO))
    elif audio_token_id is not None and prompt[i + 1] == audio_token_id:
        vision_start_indices.append((i, Modality.AUDIO))

for mm_start_idx, modality in vision_start_indices:
    if modality == Modality.IMAGE:
        mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)
        mm_token_id = img_token_id
    elif modality == Modality.VIDEO:
        mm_token_num = video_grid_thw[video_idx].prod() // (
            spatial_merge_size**2
        )
        mm_token_id = video_token_id
    elif modality == Modality.AUDIO:
        mm_token_num = int(audio_seq_lens[audio_idx].item())
        mm_token_id = audio_token_id

    input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
    mm_offset_start = len(input_ids)
    input_ids.extend([mm_token_id] * mm_token_num)
    offsets.append((mm_offset_start, len(input_ids) - 1))
```

代码逻辑：
- 字符串 prompt 先由 tokenizer 编码成 ids。
- 扫描 prompt 中紧跟在 start token 后的 modality token。
- 图片和视频 token 数来自 `grid_thw.prod() / spatial_merge_size^2`。
- 音频 token 数来自 `audio_seq_lens`。
- 将一个占位符扩展成连续 token span，并记录 offset。

为什么这样写：
- 多模态特征长度由视觉网格或音频特征长度决定，不能只保留单个占位符。
- offset 和 modality 顺序在这里同时生成，后续 `mm_items` 可以直接按同一顺序绑定特征切片。

不变量与失败模式：
- `img_grid_thw`、`video_grid_thw`、`audio_seq_lens` 的长度必须覆盖 prompt 中出现的对应媒体数量。
- `spatial_merge_size` 必须与视觉模型实际 merge 规则一致，否则 token span 长度会错。
- `cur_idx <= mm_start_idx` 断言保护 prompt 扫描顺序。

**要点：**
这一步把用户 prompt 的“语义占位符”改写成模型实际看到的“特征 token 区间”。

### 3.3 `get_mm_data` 将预计算 embedding 切成带 offset 的 `MultimodalDataItem`

问题与约束：
- 离线或上游可能直接提供 image/video/audio embedding，而不是原始图片或视频。
- embedding 是按 modality 拼接的，但 `input_ids` offset 是按 prompt 出现顺序排列的。

设计选择：
- 先复用 `build_input_ids` 生成 offsets 和 modality 顺序，再用每个 modality 的消费游标从 embedding dict 中切片，构造 `precomputed_embeddings` 类型的 `MultimodalDataItem`。

**读法：**
`get_mm_data` 把预计算特征接入同一个 scheduler 契约：`input_ids` 描述 token 序列，`mm_items` 保存每段特征及其 offset。不同 modality 的 embedding 被独立消费，避免图片和视频交错出现时切错特征。

来源：python/sglang/srt/multimodal/processors/base_processor.py L362-L400

**源码锚点：**
```python
input_ids, offsets, modality_list = self.build_input_ids(
    prompt,
    img_grid_thw=img_grid_thw,
    video_grid_thw=video_grid_thw,
    audio_seq_lens=audio_feature_lens,
)
assert all(isinstance(modality, Modality) for modality in modality_list)

mm_items = []
consumed_per_modality = {}

for modality, offset in zip(modality_list, offsets):
    num_tokens = offset[1] - offset[0] + 1
    embedding_start = consumed_per_modality.get(modality, 0)
    embedding_slice = embeddings[modality][
        embedding_start : embedding_start + num_tokens
    ]
    consumed_per_modality[modality] = embedding_start + num_tokens
    mm_items.append(
        MultimodalDataItem(
            modality=modality,
            offsets=[offset],
            precomputed_embeddings=embedding_slice,
        )
    )
```

代码逻辑：
- 调用 `build_input_ids` 得到扩展后的 ids、offsets 和 modality 顺序。
- 为每种 modality 维护已消费 token 数。
- 按当前 offset 的 token 数从对应 modality embedding 中切片。
- 将切片和 offset 包装成 `MultimodalDataItem`。
- 返回 `MultimodalProcessorOutput`，同时填入 image/video token id。

为什么这样写：
- 同一 prompt 中可能出现“图、文、图、视频”的交错顺序；按 modality 独立消费才能保持特征和 offset 对齐。
- 预计算 embedding 仍然进入标准 `MultimodalProcessorOutput`，scheduler 不需要关心特征来自哪里。

不变量与失败模式：
- `embeddings` 必须以 `Modality` 为 key，并且每种 modality 的长度不少于所有对应 offset token 数之和。
- `modality_list` 和 `offsets` 必须一一对应。
- 切片长度如果和 token span 不一致，会在后续模型输入对齐时暴露。

**要点：**
这段说明 SGLang 的多模态契约并不强制从原始媒体开始，也可以从预计算特征进入。

### 3.4 `collect_mm_items_from_processor_output` 将 HF processor 字段归并到 `MultimodalDataItem`

问题与约束：
- HF processor 返回值可能是 dict，也可能是带属性对象；字段名随模型变化。
- 一次 processor 输出可能同时包含 image、video、audio 字段，还可能包含 metadata、hash、pad value。

设计选择：
- 使用统一 getter 读取返回对象，先解析显式 modality，再按基类字段归属表把字段归到对应 `MultimodalDataItem`；metadata 字段跳过通用 set，单 item 场景再单独处理。

**读法：**
这个函数把模型专用 processor output 转成 scheduler 可理解的 `MultimodalDataItem`。特征字段名如 `pixel_values`、`pixel_values_videos`、`audio_features` 会被规范成 item 的 `feature`，非特征字段保存在 item 的模型专用数据里。

来源：python/sglang/srt/multimodal/processors/base_processor.py L1098-L1186

**源码锚点：**
```python
get_data_value = (
    data_dict.get
    if hasattr(data_dict, "get")
    else lambda name, default=None: getattr(data_dict, name, default)
)

explicit_modality = modality
modality_value = get_data_value("modality")
if explicit_modality is None and modality_value is not None:
    explicit_modality = (
        modality_value
        if isinstance(modality_value, Modality)
        else Modality.from_str(str(modality_value))
    )

items: dict[Modality, MultimodalDataItem] = {}
for attr_name, value in data_dict.items():
    if attr_name in (
        "input_ids",
        "format",
        "modality",
        "hash",
        "pad_value",
        "offsets",
    ):
        continue

    current_modality = explicit_modality or self.ATTR_NAME_TO_MODALITY.get(
        attr_name
    )

    if attr_name == "precomputed_embeddings":
        current_modality = current_modality or Modality.IMAGE
```

代码逻辑：
- 构造兼容 dict 和对象属性的 getter。
- 解析显式 modality 参数或返回对象里的 `modality` 字段。
- 跳过 input ids、format、hash、pad value、offsets 等 metadata。
- 用显式 modality 或字段归属表判断当前字段属于哪类媒体。
- `precomputed_embeddings` 没有显式 modality 时默认归入 image。
- 单 item 时，后续代码把 offsets、hash、pad value 从 tensor 转成 Python 类型。

为什么这样写：
- HF processor 输出不稳定，统一 getter 和字段映射能吸收模型差异。
- metadata 不能和普通 tensor 字段一样塞进 item，否则 pad/hash/offset 的语义会混乱。

不变量与失败模式：
- 未出现在 `ATTR_NAME_TO_MODALITY` 且没有显式 modality 的字段不会生成 item。
- 多 item 场景下 metadata 归属不明确，源码只在 `len(items) == 1` 时回填 offsets/hash/pad value。
- 默认把裸 `precomputed_embeddings` 归为 image 是兼容策略，音频或视频预计算输入应显式提供 modality。

**要点：**
这是模型输出到 scheduler 输入的适配层，字段名差异在这里被压平。

### 3.5 `process_and_combine_mm_data` 统一处理 raw、processor output 和预计算输入

问题与约束：
- 同一个请求可能包含原始媒体、已处理 processor output、预计算 embedding，甚至文本-only。
- 重新 tokenize 可能导致 prompt token drift，但不处理 raw media 又无法得到真实特征长度。

设计选择：
- 先按 `BaseMultiModalProcessorOutput.organize_results()` 分类 raw 和 dict item；raw 走 processor，dict item 按 format 直通或包装；随后补齐 input ids、offsets，最后拆分 bundled item 并设置 pad value。

**读法：**
这个函数是多模态基类的汇合点。它把媒体形态差异收敛成三件事：一组 `MultimodalDataItem`、一个 tensor 化的 `input_ids`、以及 processor 返回的额外数据。对 raw image 的 retokenize drift，还提供 `SGLANG_MM_AVOID_RETOKENIZE` 保护路径。

来源：python/sglang/srt/multimodal/processors/base_processor.py L1293-L1451

**源码锚点：**
```python
all_loaded_data = base_output.organize_results()
if not all_loaded_data:
    input_ids = self._tokenizer(
        base_output.input_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids.flatten()
    return [], input_ids, {}

dict_items, raw_images, raw_audios, raw_videos = [], [], [], []
for modality, item in all_loaded_data:
    if isinstance(item, dict):
        dict_items.append((modality, item))
    elif modality == Modality.IMAGE:
        raw_images.append(item)
    elif modality == Modality.AUDIO:
        raw_audios.append(item)
    elif modality == Modality.VIDEO:
        raw_videos.append(item)

if raw_images or raw_audios or raw_videos:
    collected_items, input_ids, ret = self._process_and_collect_mm_items(
        input_text=base_output.input_text,
        images=raw_images,
        audios=raw_audios,
        videos=raw_videos,
        **kwargs,
    )
    all_collected_items = collected_items
```

代码逻辑：
- 无媒体时直接 tokenize 文本并返回空 `mm_items`。
- 按 item 类型和 modality 将数据分为 dict、raw image、raw audio、raw video。
- raw 媒体调用 `_process_and_collect_mm_items` 获取 processor output、input ids 和 items。
- dict item 根据 format 进入 `PROCESSOR_OUTPUT` 或 `PRECOMPUTED_EMBEDDING` 分支。
- 若还没有 input ids，则依次尝试 `base_output.input_ids`、dict item 的 `input_ids`、重新 tokenize 文本。
- 对缺 offset 的 item，根据 modality token id 在 input ids 中查找 offset。
- 调用 `get_new_expanded_mm_items` 拆分 bundled item，并给已处理/预计算 item 设置 pad value。

为什么这样写：
- 一个统一汇合点能让 raw media、离线 processor output 和预计算 embedding 共享后续 scheduler 逻辑。
- offset 补齐放在最后，能覆盖 raw 和 dict 两类输入。
- bundled item 拆分提高缓存粒度，避免多个媒体共享一个粗粒度缓存键。

不变量与失败模式：
- dict item 的 `format` 必须能被 `_get_preprocessed_input_format` 识别。
- 缺 offset 的 item 必须能通过 modality 找到 token id，否则抛 `ValueError`。
- `SGLANG_MM_AVOID_RETOKENIZE` 只覆盖“raw image 且无 audio/video”的场景，其他场景仍可能 fallback 到 decode+retokenize。

**要点：**
理解这段后，可以把多模态处理看成统一产物：`input_ids + mm_items + optional model-specific metadata`。

### 3.6 `MultimodalProcessorOutput` 是 scheduler 前的稳定交付格式

问题与约束：
- processor 输出进入 scheduler 前，需要携带 input ids、已构造的媒体 item、padding/hash 信息和模型专用位置字段。
- scheduler 需要把媒体 token span 替换为 pad value，但只有 item 具备 hash/pad/offset 时才能安全替换。

设计选择：
- 用 `MultimodalProcessorOutput` dataclass 固定字段集合，并提供 `build_padded_input_ids` 在 item 完整时构造 padded ids。

**读法：**
这个输出结构是 processor 与 scheduler 的边界。它把通用的 `mm_items`、input ids、image/video/audio token id、Qwen MRoPE、Moss-VL 字段和 Transformers 兼容字段放在同一个 typed object 里，减少 dict 字段拼写错误。

来源：python/sglang/srt/managers/schedule_batch.py L380-L461

**源码锚点：**
```python
@dataclasses.dataclass
class MultimodalProcessorOutput:
    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None
    padded_input_ids: Optional[List[int]] = None

    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    video_token_id: Optional[int] = None
    audio_token_id: Optional[int] = None
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def build_padded_input_ids(input_ids, mm_items: List[MultimodalDataItem]):
        if input_ids is None or not mm_items:
            return None

        for item in mm_items:
            if item.pad_value is None or item.offsets is None:
                return None

        padded_input_ids = list(input_ids)
        for item in mm_items:
            for start, end in item.offsets:
                padded_input_ids[start : end + 1] = [item.pad_value] * (end - start + 1)
        return padded_input_ids
```

代码逻辑：
- dataclass 声明 processor 到 scheduler 的所有可选字段。
- `from_dict` 兼容旧 dict 输出。
- `build_padded_input_ids` 在 input ids、items、pad value、offsets 都存在时才替换。
- 每个 item 的 offset span 被 item 的 `pad_value` 覆盖。

为什么这样写：
- scheduler 可以先处理文本 token，再通过 pad value/hash 将多模态特征关联到缓存和模型输入。
- 字段可选让不同模型只填自己需要的 position 或 modality 信息。

不变量与失败模式：
- `mm_items` 是必填；没有 item 时 padded ids 返回 None。
- 任一 item 缺少 `pad_value` 或 `offsets`，就不能构造 padded ids。
- 如果 input ids 是 tensor，需要先 flatten/list 化；源码包含 tensor 分支。

**要点：**
这是多模态 processor 最终交给调度层的“稳定合同”。

## 4. Qwen-VL 代表实现

### 4.1 Qwen-VL 常量把图像、视频预算对齐到 patch/grid 约束

问题与约束：
- Qwen-VL 系列视觉 token 数由图像尺寸、patch/grid 和 spatial merge 决定。
- 用户可能传入超大图片或长视频，需要在预处理阶段限制像素和帧数，避免显存和计算爆炸。

设计选择：
- 在 processor 模块层定义 image/video 的 factor、min/max pixels、frame factor、fps 上下限，并允许部分预算通过环境变量调整。

**读法：**
Qwen-VL 的多模态 processor 先导入模型类、`MRotaryEmbedding` 和基类契约，再定义 `IMAGE_FACTOR=28`、图片像素上下限、视频总像素预算、帧率/帧数限制。这些常量会被 `smart_resize`、`smart_nframes` 和视频预处理复用。

来源：python/sglang/srt/multimodal/processors/qwen_vl.py L1-L60

**源码锚点：**
```python
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = envs.SGLANG_IMAGE_MAX_PIXELS.get()
MAX_RATIO = 200
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
```

代码逻辑：
- 导入 MRoPE、scheduler 数据结构和 processor 基类。
- 定义图像尺寸对齐 factor 和像素上下限。
- 从环境变量读取图片/视频上限。
- 定义视频帧数必须对齐的 factor 和默认 FPS 策略。

为什么这样写：
- factor 和像素预算必须与模型视觉 patch/grid 规则一致，否则 token 数和视觉特征会不匹配。
- 环境变量让部署侧可以按显存和吞吐调整上限，而不改代码。

不变量与失败模式：
- 输入图像长宽比超过 `MAX_RATIO` 会在 `smart_resize` 中报错。
- 视频总像素预算和帧数上限过大时会增加视觉编码成本；过小则可能损失视觉信息。

**要点：**
Qwen-VL processor 的大部分“模型特异性”首先体现在这些预算和对齐常量上。

### 4.2 `smart_resize` 在长宽比、像素预算和 factor 对齐之间折中

问题与约束：
- 图像尺寸必须能被 factor 整除，且总像素要落在模型可接受范围内。
- 直接裁剪会改变图像语义，直接缩放到固定尺寸又可能破坏宽高比。

设计选择：
- 先按 factor 近似取整，再根据最大/最小像素预算按面积比例缩放，并用 floor/ceil 到 factor 的倍数。

**读法：**
`smart_resize` 先拒绝极端长宽比，再得到 factor 对齐后的 `h_bar/w_bar`。若面积超过 `max_pixels`，用 `sqrt(height*width/max_pixels)` 计算缩小比例；若低于 `min_pixels`，用相反比例放大；最终返回仍满足 factor 对齐的尺寸。

来源：python/sglang/srt/multimodal/processors/qwen_vl.py L80-L110

**源码锚点：**
```python
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar
```

代码逻辑：
- 检查长宽比是否超过 `MAX_RATIO`。
- 将高宽 round 到 factor 的倍数，且至少为 factor。
- 面积超过上限时按面积比例缩小，并向下对齐 factor。
- 面积低于下限时按面积比例放大，并向上对齐 factor。
- 返回新的高宽。

为什么这样写：
- 视觉 token 数近似与像素面积成正比，面积约束就是 token 预算约束。
- factor 对齐保证后续 patch/grid 计算不会产生非整数 token。
- 保持宽高同比例缩放，尽量不改变视觉内容比例。

不变量与失败模式：
- `height` 和 `width` 必须为正数。
- 极端长宽比直接抛错，而不是试图压缩到模型不可解释的形态。
- `max_pixels` 和 `min_pixels` 配置不合理时，可能导致过度缩放或低质量输入。

**要点：**
这是 Qwen-VL 图片预处理里最直接控制 token 成本的函数。

### 4.3 `smart_nframes` 和 `preprocess_video` 把长视频转换成受控帧序列

问题与约束：
- 视频输入同时受总帧数、原始 FPS、目标 FPS、最小/最大帧数和总像素预算约束。
- 视频帧数还需要对齐 `FRAME_FACTOR`，否则模型侧时间维 token 可能不匹配。

设计选择：
- `smart_nframes` 从显式 `nframes` 或目标 `fps` 二选一计算帧数；`preprocess_video` 再等距抽帧、按每帧像素预算 resize，并返回视频 metadata。

**读法：**
Qwen-VL 视频路径先计算应抽取的帧数，再用 `np.linspace` 在原视频范围内等距取帧。每帧最大像素受总像素预算按帧数摊分限制，随后复用 `smart_resize` 得到尺寸，并通过 torchvision resize 得到 `TCHW` tensor。

来源：python/sglang/srt/multimodal/processors/qwen_vl.py L130-L256

**源码锚点：**
```python
assert not (
    "fps" in ele and "nframes" in ele
), "Only accept either `fps` or `nframes`"
if "nframes" in ele:
    nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
else:
    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(
        ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
    )
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)

idx = np.linspace(0, total_frames - 1, num=nframes, dtype=np.int64)
idx = np.unique(idx)
video = vr.get_frames_as_tensor(idx.tolist())
video = video.permute(0, 3, 1, 2)
```

代码逻辑：
- 禁止同时指定 `fps` 和 `nframes`。
- 显式帧数路径按 `FRAME_FACTOR` 四舍五入。
- FPS 路径根据原始时长估算帧数，再夹到最小/最大帧数和总帧数范围。
- 等距抽帧后去重，读取帧 tensor。
- 将 `NHWC` 转成 `TCHW`，再按像素预算 resize。
- 返回视频 tensor 和包含 FPS、duration、帧索引、backend 的 metadata。

为什么这样写：
- FPS 路径保留时间覆盖范围，`nframes` 路径满足用户显式预算。
- 总像素预算按帧数摊分，避免“帧多且每帧很大”同时发生。
- metadata 保留原始时长和帧索引，便于后续时间相关模型逻辑。

不变量与失败模式：
- `fps` 和 `nframes` 只能二选一。
- `nframes` 必须在 `[FRAME_FACTOR, total_frames]` 内，否则抛 `ValueError`。
- `np.unique` 可能在极短视频或重复索引场景减少实际帧数，后续逻辑必须使用实际 tensor 形状。

**要点：**
视频路径的目标不是简单解码，而是在时间覆盖、帧数和每帧 token 预算之间做可控折中。

### 4.4 `QwenVLImageProcessor` 声明支持架构、特殊 token 和 MRoPE 计算

问题与约束：
- Qwen2-VL、Qwen2.5-VL、Qwen3-VL、Omni 等模型共享相似视觉占位符，但配置字段和位置编码细节不同。
- 多模态 rotary 需要 image/video grid 信息，不能只依赖普通 token position。

设计选择：
- Qwen processor 在 class-level 声明支持架构和 Transformers backend 兼容；初始化时从 hf config 读取 token id，构造 vision start/pad/end 的特殊 token regex；`compute_mrope_positions` 调用 `MRotaryEmbedding.get_rope_index`。

**读法：**
这个类把 Qwen 系列模型和基类契约连起来：`models` 让注册表能命中，`mm_tokens` 让 prompt split 和 offset 计算认识 Qwen 的 vision token 串，`compute_mrope_positions` 则把 `mm_items` 中的 grid 合并后交给 MRoPE。

来源：python/sglang/srt/multimodal/processors/qwen_vl.py L260-L416

**源码锚点：**
```python
class QwenVLImageProcessor(SGLangBaseProcessor):
    supports_transformers_backend = True
    models = [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
        Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5ForCausalLMMTP,
        InternS2PreviewForConditionalGeneration,
        Qwen3OmniMoeForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type
        if hf_config.model_type == "qwen3_omni_moe":
            hf_config = hf_config.thinker_config

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.IM_TOKEN_ID = hf_config.image_token_id
        self.VIDEO_TOKEN_ID = hf_config.video_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            image_token_regex=re.compile(
                r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>"
            ),
            video_token_id=self.VIDEO_TOKEN_ID,
            audio_token_id=self.audio_token_id,
        ).build(_processor)
```

代码逻辑：
- class-level `models` 列出注册表可匹配的模型架构。
- `supports_transformers_backend=True` 允许 Transformers backend 下仍用该 processor。
- Omni 模型使用 `thinker_config` 进入通用 Qwen 视觉配置。
- 初始化 image/video/audio token id，并构造 Qwen vision token regex。
- `compute_mrope_positions` 合并 image/video grid 后调用 `MRotaryEmbedding.get_rope_index`。

为什么这样写：
- Qwen 系列模型多，但视觉 token 和 MRoPE 契约高度相似，可以共享一个 processor。
- regex 匹配展开后的 `<|image_pad|>` 序列，适配已经包含多个 image pad 的 prompt。
- MRoPE 由 processor 计算后传给模型，避免 scheduler 推测模型专用位置编码。

不变量与失败模式：
- `hf_config` 必须提供 vision/image/video token id；缺字段会在初始化阶段失败。
- `mm_items` 必须包含正确的 `image_grid_thw`、`video_grid_thw`，否则 MRoPE position 会错。
- Omni 的配置分支依赖 `thinker_config` 存在。

**要点：**
Qwen-VL 是典型的模型专用 processor：它不重写所有基类流程，而是提供模型需要的 token、grid 和位置编码细节。

## 5. 视觉运行时优化

### 5.1 ViT CUDA Graph runner 用 shape key 捕获并复用视觉 block 执行图

问题与约束：
- ViT block 在小 batch 或固定形状视觉输入上有明显 kernel launch overhead。
- CUDA Graph 要求输入 buffer、position embedding workspace 和输出 buffer 形状稳定。

设计选择：
- `run` 先把 `[seq_len, hidden]` 变成 `[S, B=1, H]`，用形状生成 graph key；没有图时创建，有图时更新 workspace/input 并 replay。

**读法：**
runner 的 `run` 负责 graph lifecycle，`replay` 负责把新输入和位置编码拷入已捕获 graph 的 workspace。可选的 `output_indices` 用于 Qwen2.5-VL window permutation 的逆重排。

来源：python/sglang/srt/multimodal/vit_cuda_graph_runner.py L330-L388

**源码锚点：**
```python
if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
    head_dim = rotary_pos_emb_cos.shape[1]
    self._ensure_sin_cos_ws(graph_key, head_dim)
    used_cos_ws = self.sin_cos_ws[0][:graph_key, :]
    used_sin_ws = self.sin_cos_ws[1][:graph_key, :]
    used_cos_ws.copy_(rotary_pos_emb_cos)
    used_sin_ws.copy_(rotary_pos_emb_sin)

self.block_input[graph_key].copy_(x_3d)
self.block_graphs[graph_key].replay()
out = self.block_output[graph_key]

if output_indices is not None:
    out = out.index_select(0, output_indices)

def run(
    self,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_window_seqlens: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    rotary_pos_emb_cos: Optional[torch.Tensor] = None,
    rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    output_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_3d = x.unsqueeze(1)
    graph_key = self._get_graph_key(x_3d)
```

代码逻辑：
- 根据 rotary 输入更新 sin/cos workspace。
- 将当前 `x_3d` 拷入 graph 对应输入 buffer。
- replay 已捕获 graph，并读取对应输出 buffer。
- 如需重排，按 `output_indices` 做 `index_select`。
- `run` 中先扩 batch 维，再按 shape key 创建或复用 graph。

为什么这样写：
- CUDA Graph replay 只能复用已捕获的静态 buffer，更新输入必须用 copy 到 graph buffer 的方式完成。
- shape key 避免不同长度的视觉 token 误用同一张 graph。
- 输出重排放在 replay 后，保留 graph 内部固定执行形态。

不变量与失败模式：
- 新输入形状必须与 graph key 对应 buffer 兼容。
- position embedding 或 rotary workspace 维度必须和捕获时一致。
- 未命中的 shape 会创建新 graph，形状过多会增加 graph 缓存占用。

**要点：**
这段优化只改变视觉编码执行方式，不改变 processor 和 scheduler 之间的 `mm_items` 契约。

### 5.2 EVS 用跨帧差异度裁剪长视频 token

问题与约束：
- 长视频的视觉 token 数随帧数线性增长，直接保留全部 token 会拉高显存和 attention 成本。
- 首帧通常提供全局上下文，不能因为裁剪率高而全部丢失。

设计选择：
- 先计算最少保留 token 数，保证至少保留一帧；再用相邻帧 cosine dissimilarity 排序，保留差异最大的 token，并人为提升首帧权重。

**读法：**
EVS 的 retention mask 将 video embeddings reshape 成 `[T, H', W', hidden]`，比较相邻帧同空间位置的相似度，以 `1 - similarity` 作为变化量。首帧 dissimilarity 被设为很高，确保首帧 token 总能进入 top-k。

来源：python/sglang/srt/multimodal/evs/evs_core.py L21-L97

**源码锚点：**
```python
def compute_retained_tokens_count(
    tokens_per_frame: int, num_frames: int, q: float
) -> int:
    total_tokens = tokens_per_frame * num_frames
    evs_num_tokens = int(total_tokens * (1 - q))
    min_num_tokens = tokens_per_frame
    return max(min_num_tokens, evs_num_tokens)

T, H, W = map(int, video_size_thw)
video_embeds = video_embeds.reshape(
    T,
    H // spatial_merge_size,
    W // spatial_merge_size,
    video_embeds.size(-1),
)
tokens_per_frame = (H // spatial_merge_size) * (W // spatial_merge_size)
similarity = torch.nn.functional.cosine_similarity(
    video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1
)
dissimilarity = 1 - similarity
dissimilarity = torch.cat(
    [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0
)
order = torch.argsort(dissimilarity.view(-1), dim=-1, descending=True, stable=True)
```

代码逻辑：
- `compute_retained_tokens_count` 根据裁剪率计算保留数量，并至少保留一帧 token。
- 将扁平 video embedding reshape 回时间和空间网格。
- 计算相邻帧同位置 embedding 的 cosine similarity。
- 用 `1 - similarity` 表示变化量。
- 首帧变化量填成高值，保证进入保留集合。
- 对扁平变化量稳定降序排序，取前 `retain_num_tokens` 构造 boolean mask。

为什么这样写：
- 变化大的 token 更可能携带新增视觉信息，适合作为长视频裁剪依据。
- 首帧保留策略给模型稳定的场景起点，避免视频只剩中后段局部变化。
- stable sort 让相同分数下的选择更可复现。

不变量与失败模式：
- `video_size_thw` 与 `video_embeds` 长度必须满足 `T * H/spatial_merge_size * W/spatial_merge_size`。
- `q` 应在 `[0, 1)`；过高会退化为只保留首帧级别 token。
- 相邻帧差异度只衡量局部变化，静态但语义重要的后续帧可能被裁掉。

**要点：**
EVS 是多模态长视频路径的 token 预算阀门，和前面的 Qwen 视频抽帧共同控制视觉输入规模。

## 6. 运行验证

这条多模态主线可以先用静态检索验证，不必启动 VLM 服务。检查目标是确认四个边界仍然存在：Processor 选择、prompt/media 组合、模型专用 Qwen-VL 处理、视觉运行时优化。

```powershell
rg -n "PROCESSOR_MAPPING|import_processors|get_mm_processor|MultimodalSpecialTokens|load_mm_data|build_input_ids|collect_mm_items_from_processor_output|process_and_combine_mm_data|MultimodalProcessorOutput|smart_resize|smart_nframes|QwenVLImageProcessor|_get_graph_key|replay|compute_retained_tokens_count|cosine_similarity|argsort" `
  sglang/python/sglang/srt/managers/multimodal_processor.py `
  sglang/python/sglang/srt/multimodal/processors/base_processor.py `
  sglang/python/sglang/srt/multimodal/processors/qwen_vl.py `
  sglang/python/sglang/srt/multimodal/vit_cuda_graph_runner.py `
  sglang/python/sglang/srt/multimodal/evs/evs_core.py `
  sglang/python/sglang/srt/managers/schedule_batch.py
```

预期现象：

- `multimodal_processor.py` 命中 `PROCESSOR_MAPPING`、`import_processors` 和 `get_mm_processor`，证明模型架构到 processor 的选择仍在 manager 层完成。
- `base_processor.py` 命中特殊 token、媒体加载、`build_input_ids`、`collect_mm_items_from_processor_output` 和 `process_and_combine_mm_data`，证明 prompt 占位符到 `mm_items` 的转换仍由基类主导。
- `schedule_batch.py` 命中 `MultimodalProcessorOutput`，证明交付给 scheduler 的稳定对象没有换名。
- `qwen_vl.py` 命中 `smart_resize`、`smart_nframes` 和 `QwenVLImageProcessor`，证明 Qwen-VL 的像素/帧预算和模型专用 processor 仍在同一实现内。
- `vit_cuda_graph_runner.py` 命中 shape key 与 replay，`evs_core.py` 命中保留 token 数、相邻帧相似度和排序，证明运行时优化没有进入通用 processor 契约。

如果某条命中消失，先不要直接改本文结论；应回到对应文件看职责是否迁移。例如 `build_input_ids` 消失通常意味着占位符展开路径重构，`MultimodalProcessorOutput` 消失则表示 scheduler 交付契约发生了更大变化。
