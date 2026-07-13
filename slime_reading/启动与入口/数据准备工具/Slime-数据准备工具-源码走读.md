---
title: "数据准备工具 · 源码走读"
type: walkthrough
framework: slime
topic: "数据准备工具"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# 数据准备工具 · 源码走读

这篇追踪一条真实路径：下载一个 HF 模型，先转换成 Megatron `torch_dist` 给训练用；训练后再把 Megatron checkpoint 导回 HF safetensors。

读完后，你应该能定位这些问题：为什么 conversion 命令必须带 `MODEL_ARGS`，为什么 `--hf-checkpoint` 和 `--ref-load` 不是同一个目录，为什么导出 HF 时 embedding 行数不对，为什么 output 目录默认拒绝覆盖。

## 长文读法

这篇按“训练用 Megatron，rollout / tokenizer 仍看 HF”读：HF checkpoint 先借助 Megatron 参数转换成 `torch_dist` 给训练用；训练结束或周期保存时，再把 Megatron checkpoint 转回 HF safetensors。`--hf-checkpoint`、`--ref-load`、`--load`、`--save`、`--save-hf` 分别服务不同边界，不能当成同一个路径。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立权重格式全景 | 贯穿场景、步骤一到二 | `MODEL_ARGS` 是 Megatron 构图输入，HF 目录只是权重、tokenizer 和 config 来源 |
| 排查 HF→torch_dist 转换失败 | 步骤二到四 | 转换脚本复用 Megatron parser，初始化分布式后用 AutoBridge 把 HF 权重灌进 Megatron 模型 |
| 理解多卡转换和 PP 推导 | 步骤三 | 只在 PP 默认值为 1 且 world size>1 时尝试推导；候选只沿“world size 不断除以 2”搜索 |
| 排查训练脚本路径混淆 | 步骤六到七 | `--ref-load` 指向 bf16 torch_dist，`--hf-checkpoint` 可用于 tokenizer / rollout 初始化，FP8 HF 不替代训练权重 |
| 排查 torch_dist→HF 导出 | 步骤八到十 | 导出先读 dist checkpoint metadata，再展开层和 expert 参数，最后按模型 converter 保存 safetensors |
| 排查 embedding 行数或缺权重 | 步骤九到十 | padding 要显式处理；缺失权重和 tokenizer/config 等资产可从 origin HF 目录补齐 |
| 区分离线导出和训练中导出 | 步骤十一到十二 | CLI 默认拒绝覆盖输出目录；训练中的 `--save-hf` 走 actor 保存路径，不等同于离线转换脚本 |

读的时候先画清路径：HF 原始目录、Megatron `torch_dist` 目录、训练 checkpoint 目录、最终 HF 导出目录。多数问题不是转换算法本身，而是这些目录语义混了。

## 贯穿场景

以 Qwen3-4B 为例，训练脚本里有三段关键配置：

```bash
# 来源：scripts/run-qwen3-4B.sh L37-L47
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /root/Qwen3-4B_torch_dist
   --load /root/Qwen3-4B_slime/
   --save /root/Qwen3-4B_slime/
   --save-interval 20
)
```

`source models/qwen3-4B.sh` 给 Megatron 构图；`--ref-load` 指向转换后的 `torch_dist`；`--hf-checkpoint` 仍指 HF 目录，供 tokenizer、SGLang 和 AutoConfig 使用。

## 步骤一：quick start 先 source model args，再转换

系统压力：HF checkpoint 不足以让 Megatron 构出模型。转换脚本复用 Megatron parser，所以需要用户先把结构参数注入 CLI。

设计选择：官方 quick start 要求先 source model script，再运行 `convert_hf_to_torch_dist.py`。

```bash
# 来源：docs/en/get_started/quick_start.md L76-L77
cd /root/slime
source scripts/models/glm4-9B.sh
```

```bash
# 来源：docs/en/get_started/quick_start.md L85-L88
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/GLM-Z1-9B-0414 \
    --save /root/GLM-Z1-9B-0414_torch_dist
```

读者抓手：`MODEL_ARGS` 是 Megatron 构图输入；`--hf-checkpoint` 是权重来源；`--save` 是转换产物目录。

## 步骤二：HF→torch_dist 的 parser 复用 Megatron

系统压力：转换脚本要和训练使用同一套 Megatron 参数默认值，否则转换出来的 checkpoint 和训练时模型结构可能不一致。

设计选择：脚本调用 Megatron `parse_args`，再执行 Slime 的 `set_default_megatron_args`。

```python
# 来源：tools/convert_hf_to_torch_dist.py L21-L41
def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--custom-model-provider-path",
        type=str,
        default=None,
        help="Path to a custom model provider function.",
    )
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    parser.add_argument("--allgather-cp", action="store_true", default=False)
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser
```

```python
# 来源：tools/convert_hf_to_torch_dist.py L44-L57
def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )
```

这段还说明了多卡转换的边界：world size 不能超过 layer 数。

## 步骤三：多卡转换会自动推导 PP

系统压力：大模型单卡转换可能 OOM。多卡转换时，只有当解析后的 PP 仍为 1，脚本才把 world size 用作候选；若最后一个 stage 没层，只在候选为偶数时除以 2。它不是遍历所有约数，更不保证任意 world size 都能找到方案。

```python
# 来源：tools/convert_hf_to_torch_dist.py L59-L81
def ceildiv(a, b):
    return -(a // -b)

if args.pipeline_model_parallel_size == 1 and world_size > 1:
    pp_size = world_size
    while True:
        args.pipeline_model_parallel_size = pp_size
        args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
            args.num_layers, args.pipeline_model_parallel_size
        ) * (args.pipeline_model_parallel_size - 1)

        if args.decoder_last_pipeline_num_layers > 0:
            break

        if pp_size % 2 == 0:
            pp_size //= 2
        else:
            raise ValueError(
                f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
            )
print(
    f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
)
```

不变量：多卡不是“随便多少卡都能跑”。world size 先受 `world_size <= num_layers` 限制；自动搜索只覆盖 `world_size, world_size/2, ...` 这条链；显式 PP 则不进入该自动分支，交给 Megatron 后续校验。

## 步骤四：main 初始化分布式，再用 AutoBridge 灌权重

系统压力：Megatron checkpoint 保存依赖分布式环境和并行状态；不能只在普通 Python 进程里把 safetensors 改名。

设计选择：`main()` 初始化 NCCL process group、调用 `init(args)`，构建 Megatron model，再用 mbridge `AutoBridge` 从 HF 加载权重。

```python
# 来源：tools/convert_hf_to_torch_dist.py L87-L126
def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    args = get_args()
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
```

ROCm 分支也在这里：HIP 环境必须 `--use-cpu-initialization`。但这段只 patch 了 checkpoint writer，process group 的 backend 仍显式写成 `nccl`；不能仅凭 HIP 分支就推断它会改用 Gloo 或另一套通信后端。

## 步骤五：保存 step 1，再改成 Megatron release checkpoint

系统压力：训练脚本希望从 Megatron checkpoint 根目录加载；初始转换产物应该像 release checkpoint，而不是普通训练 step。

设计选择：先 `save_checkpoint` 保存 step 1，rank 0 写 tracker 为 `release`，再把 step 目录 move 到 release 目录。

```python
# 来源：tools/convert_hf_to_torch_dist.py L129-L148
if args.use_cpu_initialization:
    model[0] = model[0].cpu()

print_memory("after loading model")
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()

save_checkpoint(1, model, None, None, 0)

if dist.get_rank() == 0:
    # change to release ckpt
    tracker_filename = get_checkpoint_tracker_filename(args.save)
    with open(tracker_filename, "w") as f:
        f.write("release")
    source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
    target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
    shutil.move(source_dir, target_dir)
dist.barrier()
dist.destroy_process_group()
```

运行验证：转换后检查 `--save` 目录下的 tracker 文件内容是 `release`，并存在 release checkpoint 目录。

重跑边界：代码在写 tracker 后直接 `shutil.move(source_dir, target_dir)`，没有先删除旧 `release/`。因此 `--save` 应指向全新目录；在已有 release checkpoint 上原地重跑可能形成嵌套/混合旧分片，不能视为安全覆盖流程。

## 步骤六：训练 CKPT_ARGS 把两个生态接起来

系统压力：训练启动时，Megatron 侧要加载 reference/actor checkpoint；SGLang 侧要加载 HF config/tokenizer，然后等 actor 首次推权。

设计选择：脚本同时传 `--hf-checkpoint` 和 Megatron `--ref-load/--load/--save`。

```bash
# 来源：docs/en/get_started/quick_start.md L137-L149
CKPT_ARGS=(
   # To load tokenizer and other information, won't actually use model weight parameters from hf path
   --hf-checkpoint /root/GLM-Z1-9B-0414
   # Reference Model's Megatron format checkpoint
   --ref-load /root/GLM-Z1-9B-0414_torch_dist
   # Actor model loading path. Should typically match --save for checkpoint resumption
   # If empty or doesn't contain a valid checkpoint, loads from --ref-load instead
   --load /root/GLM-Z1-9B-0414_slime/
   # Model save path during training
   --save /root/GLM-Z1-9B-0414_slime/
   # Model save interval (steps)
   --save-interval 20
)
```

读者抓手：`--load` 是 actor 续训目录；没有有效 checkpoint 时会从 `--ref-load` 初始化。

## 步骤七：FP8 HF 只影响 rollout 初始化，不替代 bf16 torch_dist

系统压力：有些部署希望 SGLang 用 FP8 HF 权重做推理初始化，但 Megatron 训练仍需要 bf16 转出的 `torch_dist`。

quick start 明确写了这个边界：

```bash
# 来源：docs/en/get_started/quick_start.md L401-L405
   # Used to load tokenizer and other information, actually won't use model weight parameters from hf path
   --hf-checkpoint /root/Qwen3-4B-FP8

   # The megatron checkpoint still needs to be the dist weights converted from bf16 huggingface at the beginning, not modified because of FP8 rollout.
   --ref-load /root/Qwen3-4B_torch_dist
```

结论：FP8 HF 不是 Megatron 训练 checkpoint 的替代品。

## 步骤八：torch_dist→HF 先读 metadata，再只加载模型权重

系统压力：Megatron checkpoint 里不只有模型权重，还有 optimizer 和状态。导出 HF 只需要模型参数，并且 metadata 里可能引用 Megatron 类。

设计选择：自定义 unpickler 吞掉 Megatron/GLM 类，planner 跳过 optimizer 和 `_state`。

```python
# 来源：tools/convert_torch_dist_to_hf.py L19-L31
class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper
```

```python
# 来源：tools/convert_torch_dist_to_hf.py L48-L63
class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)
```

## 步骤九：参数名先展开层和 expert，再路由到 HF converter

系统压力：Megatron state dict 里可能有合并的 layer 或 expert 维度，HF 期望逐层逐 expert 的参数名。

设计选择：`get_named_params` 给每个参数加 `module.module.` 前缀，再通过 layer/expert 展开；`convert_to_hf` 按 model name 分发到架构 converter。

```python
# 来源：tools/convert_torch_dist_to_hf.py L66-L103
def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)
```

converter 分发：

```python
# 来源：slime/backends/megatron_utils/megatron_to_hf/__init__.py L25-L46
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    param = remove_padding(name, param, args.vocab_size)

    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    return quantize_params(args, name, converted_named_tensors, quantization_config)


# TODO optimize
_cached_tensors = {}


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    if "minimaxm2" in model_name or "minimax_m2" in model_name:
        converted_named_tensors = convert_minimax_m2_to_hf(args, name, param)
    elif "glm4moelite" in model_name or "deepseekv3" in model_name or "glmmoedsa" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)
    elif "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
```

## 步骤十：保存 HF 时要处理 padding、missing weights 和 assets

系统压力：HF 输出需要 safetensors 分片、index、config/tokenizer assets。某些权重可能没有从 Megatron state dict 转出来，需要从原 HF 补。

设计选择：`save_tensors` 负责分片和 index，`origin_hf_dir` 可复制 missing weights 和非权重文件。

```python
# 来源：tools/convert_torch_dist_to_hf.py L106-L144
def save_tensors(args, model_name, state_dict, output_dir, chunk_size, vocab_size=None, origin_hf_dir=None):
    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # 2GB
    current_size = 0
    total_size = 0
    modeltensors = [{}]
    converted_names = set()
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            converted_names.add(converted_name)
            tensor_size = converted_param.numel() * converted_param.element_size()
            if tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size
            total_size += tensor_size

    if origin_hf_dir is not None:
        safetensors_files = [f for f in os.listdir(origin_hf_dir) if f.endswith(".safetensors")]
        for filename in safetensors_files:
            with safetensors.safe_open(os.path.join(origin_hf_dir, filename), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k not in converted_names:
                        converted_name = k
                        print(f"add {k} from origin hf checkpoint")
                        converted_param = f.get_tensor(k)
                        converted_names.add(k)
                        tensor_size = converted_param.numel() * converted_param.element_size()
                        if tensor_size + current_size > chunk_size:
                            modeltensors.append({})
                            current_size = 0
                        modeltensors[-1][converted_name] = converted_param
                        current_size += tensor_size
                    total_size += tensor_size
```

这里有四个严格边界：

- `--add-missing-from-origin-hf` 只扫描 `origin_hf_dir` 顶层、后缀为 `.safetensors` 的文件；不解析 `.bin`，也不递归子目录。
- `copy_assets()` 同样只复制顶层普通文件，跳过目录、原 index 和所有 `.safetensors`。
- CLI `--vocab-size` 的预裁剪之后，`convert_to_hf()` 还会按 checkpoint `args.vocab_size` 再裁一次；两者不一致时要检查最终 shape，不能假设 CLI 完全覆盖 checkpoint 值。
- 当前 `total_size += tensor_size` 位于“是否缺失”的条件块之外；启用补权重后，原 HF 中已存在于 `converted_names` 的 key 也会重复累计上一次的 `tensor_size`。这会让 index 的 `metadata.total_size` 失真，因此完整性验收应以 `weight_map`、实际 shard 与 tensor key/shape 为主，不能只信该数字。

assets 复制：

```python
# 来源：tools/convert_torch_dist_to_hf.py L165-L175
def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)
```

## 步骤十一：CLI 防止覆盖，并从 origin HF 推导 model name

系统压力：导出 HF 可能覆盖已有模型目录；如果不知道 model name，就无法路由 converter。

设计选择：默认拒绝写入已存在目录；`model_name` 缺失时要求 `origin_hf_dir`。但 `--force` 只跳过拒绝检查，既不删除目录，也不清理旧 safetensors/assets。

```python
# 来源：tools/convert_torch_dist_to_hf.py L178-L219
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "-a", "--add-missing-from-origin-hf", action="store_true", help="Add missing weights from origin hf checkpoint"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        args.model_name = type(hf_config).__name__.lower()
```

另一个源码漂移是 `--chunk-size`：代码默认值是 `5 * 1024**3`（5 GiB），help 仍写 “default is 2GB”。运行语义以代码默认值为准；需要可复现实验时显式传值。

## 步骤十二：训练中的 `--save-hf` 是另一条导出路径

系统压力：有时不想离线跑转换脚本，而是在 actor 保存模型时顺便导出 HF。

设计选择：`Actor.save_model` 在保存 Megatron checkpoint 后，如果 `save_hf` 不为空且 role 是 actor，就调用 `save_hf_model_to_path`。

```python
# 来源：slime/backends/megatron_utils/actor.py L558-L577
def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
    if self.args.debug_rollout_only:
        return

    # torch dist may trigger nccl communication during saving.
    if self.args.offload_train:
        self.wake_up()

    if self.args.async_save:
        from megatron.training.async_utils import maybe_finalize_async_save

        maybe_finalize_async_save(blocking=True)

    save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

    if force_sync and self.args.async_save:
        maybe_finalize_async_save(blocking=True)

    if self.args.save_hf is not None and self.role == "actor":
        save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)
```

导出实现按 `megatron_to_hf_mode` 分 raw / bridge。它与离线脚本不是同一实现：raw 路径会先清理目标目录中的已识别 HF 权重文件、复制顶层非权重 assets，再由分布式 writer 写分片；bridge 路径委托 Megatron Bridge。

```python
# 来源：slime/backends/megatron_utils/hf_checkpoint_saver.py L22-L42
def save_hf_model_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
) -> None:
    """Save a Megatron model as an HF checkpoint at a concrete directory."""
    if args.megatron_to_hf_mode == "bridge":
        save_hf_model_bridge_to_path(args, output_dir, model)
    else:
        save_hf_model_direct_to_path(
            args,
            output_dir,
            model,
            model_name=model_name,
            quantization_config=quantization_config,
            progress_desc=progress_desc,
        )
```

## 运行验证

静态验证：

```powershell
Set-Location slime
python -m py_compile tools/convert_hf_to_torch_dist.py tools/convert_torch_dist_to_hf.py tools/convert_torch_dist_to_hf_bridge.py tools/convert_torch_dist_to_hf_parallel.py slime/backends/megatron_utils/hf_checkpoint_saver.py
```

完整验证需要真实模型和 Megatron：

```bash
cd /root/slime
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-4B \
  --save /root/Qwen3-4B_torch_dist
```

预期现象：

- `Qwen3-4B_torch_dist` 下有 tracker 文件，内容指向 `release`。
- `release/` 下存在 Megatron dist checkpoint 文件。
- 训练脚本的 `--ref-load` 指向这个根目录。

## 复盘迁移

读转换工具要按产物倒推：

1. 先问这个目录给谁吃：HF 生态、Megatron、还是 SGLang。
2. 再问构图信息从哪里来：`MODEL_ARGS`、HF config、还是 checkpoint metadata。
3. 最后看保存格式：Megatron tracker/release，还是 HF safetensors/index/assets。

下一篇 [[Slime-数据准备工具-数据流]] 会把这些目录关系画成数据流。
