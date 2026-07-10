---
title: "数据准备工具 · 核心概念"
type: concept
framework: slime
topic: "数据准备工具"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-10
---
# 数据准备工具 · 核心概念

## 你为什么要读

这篇先建立权重形态模型。Slime 训练同时站在两个生态里：Megatron 负责训练，SGLang 负责 rollout；它们吃的 checkpoint 形态不同。

## 三种目录

| 目录 | 谁消费 | 典型参数 | 内容 |
|------|--------|----------|------|
| HF checkpoint | SGLang、tokenizer、AutoConfig、converter | `--hf-checkpoint`、`--origin-hf-dir` | `config.json`、tokenizer 文件、safetensors |
| Megatron `torch_dist` checkpoint | Megatron actor/ref/critic | `--ref-load`、`--load`、`--save` | tracker、`release/` 或 `iter_xxx/`、`.metadata`、`common.pt` |
| HF export output | HF 生态或人工检查 | `--output-dir`、`--save-hf` | 导出的 safetensors 和从原 HF 复制的 assets |

关键区别：`--hf-checkpoint` 给 SGLang 初始化和 tokenizer；`--ref-load` / `--load` 给 Megatron 训练权重。训练前 actor 会把 Megatron 权重同步到 SGLang，所以 `--hf-checkpoint` 不需要是最新训练权重。

usage 文档也明确区分了两条加载路径：

```text
来源：docs/en/get_started/usage.md L127-L145
When using slime, there are three parameters for loading and saving checkpoints:

  - `--ref-load`: The Megatron checkpoint for the reference model.
  - `--load`: The Megatron checkpoint for the actor. If `--load` is not set, or if the specified directory does not exist or does not contain `latest_checkpointed_iteration.txt`, the actor will be initialized from the `--ref-load` checkpoint.
  - `--save`: The path where the actor's checkpoints are saved.

Note:

  - Regardless of the checkpoint storage method (i.e., however `--ckpt-format` is set), Megatron can load both `torch` and `torch_dist` formats.

### Loading SGLang

Loading SGLang is very simple. You only need:

  - `--hf-checkpoint`: The Hugging Face checkpoint used to initialize SGLang.

Note:

  - Before the first training step, slime will synchronize the parameters from Megatron to SGLang. Therefore, the `--hf-checkpoint` does not need to contain the latest training parameters, and you do not need to change the HF checkpoint when resuming training.
```

## 为什么要 `MODEL_ARGS`

HF checkpoint 有 `config.json`，但 Megatron 的模型构图仍由 CLI 参数驱动。转换脚本复用 Megatron parser，所以必须把结构超参传进去。

Qwen3-4B 的 model args 示例：

```bash
# 来源：scripts/models/qwen3-4B.sh L1-L17
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
```

quick start 也强调需要 `source scripts/models/<model>.sh`：

```text
来源：docs/en/get_started/quick_start.md L118-L131
quick_start 要求 source `scripts/models/glm4-9B.sh` 这类模型脚本。
这些配置是 Megatron 所需的模型超参。
Megatron 不能直接从 checkpoint 读取全部模型配置，必须人工指定。
文档特别提醒检查 `--rotary-base` 等参数是否和当前 HF 版本一致。
```

如果 `rotary_base`、layer 数、hidden size、head 数不对，最坏不是马上报错，而是训练/导出后质量异常。

## HF→torch_dist 是“灌进 Megatron 再保存”

转换不是简单重命名 safetensors。脚本会：

1. 初始化 torch distributed。
2. 用 Megatron parser 和 `MODEL_ARGS` 建模型。
3. 用 `AutoBridge.from_pretrained` 从 HF 加载权重到 Megatron model。
4. 调 Megatron `save_checkpoint`。
5. rank 0 把 step 目录改成 `release`。

源码主线：

```python
# 来源：tools/convert_hf_to_torch_dist.py L121-L148
model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

# Load model
hf_model_path = args.hf_checkpoint
bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
bridge.load_weights(model, hf_model_path, memory_efficient=True)
print(f"Model loaded: {hf_model_path}")

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

因此转换产物不是 HF 目录，而是 Megatron release checkpoint。

## torch_dist→HF 是“读分布式 state dict 再写 safetensors”

反向转换要读 Megatron dist checkpoint 的 metadata，只加载模型权重，跳过 optimizer 和内部状态，再把参数名转成 HF 命名。

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

保存 HF safetensors 时，脚本会按 `model_name` 路由 converter，并生成 index：

```python
# 来源：tools/convert_torch_dist_to_hf.py L146-L164
metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

num_files = len(modeltensors)
for i, tensors in enumerate(modeltensors):
    filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
    for key in tensors.keys():
        metadata["weight_map"][key] = filename
index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
json.dump(metadata, open(index_filepath, "w"), indent=2)
print(f"{index_filepath} saved.")

for i, tensors in enumerate(modeltensors):
    filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
    t = time.time()
    filepath = os.path.join(output_dir, filename)
    safetensors.torch.save_file(tensors, filepath)
    print(f"{filename} saved in {time.time() - t:.2f} sec.")
```

## padding 是导出 HF 时最常见的坑

Megatron 可能为了并行或性能把 vocab padding 到更大的 `padded_vocab_size`。HF embedding/output layer 不应该保留 padding 行。

去 padding 逻辑非常窄：

```python
# 来源：slime/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py L6-L12
def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.
    """
    if strip_param_name_prefix(name) in {"embedding.word_embeddings.weight", "output_layer.weight"}:
        return param[:vocab_size]
    return param
```

这意味着 `--vocab-size` 不只是“可选美化”，它决定导出的 embedding/output layer 是否裁回 HF vocab 行数。

## `origin_hf_dir` 是 assets 和 model_name 的来源

导出 HF 时，如果不显式传 `--model-name`，脚本会从 `origin_hf_dir` 的 AutoConfig 推导 model name；同时把 tokenizer、config 等非权重文件复制到输出目录。

```python
# 来源：tools/convert_torch_dist_to_hf.py L209-L244
if os.path.exists(args.output_dir) and not args.force:
    raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

if args.model_name is None and args.origin_hf_dir is None:
    raise ValueError(
        "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
    )

if args.model_name is None:
    hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
    args.model_name = type(hf_config).__name__.lower()

state_dict = {}
print(f"loading model from {args.input_dir}")
t = time.time()
megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
dist_cp.state_dict_loader._load_state_dict(
    state_dict,
    storage_reader=WrappedStorageReader(args.input_dir),
    planner=EmptyStateDictLoadPlanner(),
    no_dist=True,
)
print(f"model loaded in {time.time()-t:.2f} sec.")

save_tensors(
    megatron_args,
    args.model_name,
    state_dict,
    args.output_dir,
    args.chunk_size,
    args.vocab_size,
    args.origin_hf_dir if args.add_missing_from_origin_hf else None,
)

if args.origin_hf_dir:
    copy_assets(args.origin_hf_dir, args.output_dir)
```

## 复盘

读 Tools-DataPrep 时，记住三条线：

1. HF 线：`--hf-checkpoint` / `--origin-hf-dir`，提供 config、tokenizer、SGLang 初始化和导出 assets。
2. Megatron 线：`--ref-load` / `--load` / `--save`，提供训练侧 checkpoint。
3. 转换线：`MODEL_ARGS + AutoBridge + save_checkpoint` 或 `dist_cp + convert_to_hf + safetensors`。

下一篇 [[Slime-数据准备工具-源码走读]] 沿这三条线走源码。
