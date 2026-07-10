---
title: "数据准备工具 · 排障指南"
type: troubleshooting
framework: slime
topic: "数据准备工具"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 数据准备工具 · 排障指南

这篇不是零散问答，而是排障入口。先按症状定位路径，再回到源码确认是哪条不变量被破坏。

## 症状总表

| 症状 | 先查什么 | 源码入口 | 常见修复 |
|------|----------|----------|----------|
| Megatron 启动找不到 checkpoint | `--ref-load` / `--load` 是否指向 Megatron checkpoint 根目录 | `docs/en/get_started/usage.md` L127-L145 | 用 HF→torch_dist 的输出根目录，不要指 HF 目录 |
| 转换时参数 shape 不匹配 | `MODEL_ARGS` 是否与 HF config 对齐 | `docs/en/get_started/quick_start.md` L118-L131 | 重新 source 正确 model script，核对 RoPE/vocab/head/layer |
| 多卡转换失败 | world size 与 layer 数、PP size 是否合法 | `tools/convert_hf_to_torch_dist.py` L44-L84 | 减少卡数或显式设置 PP |
| ROCm 转换断言失败 | 是否传 `--use-cpu-initialization` | `tools/convert_hf_to_torch_dist.py` L87-L120 | HIP 环境加 `--use-cpu-initialization` |
| 导出 HF 后 embedding 行数不对 | 是否传 `--vocab-size` | `padding_remover.py` L6-L12 | 用 HF config 或 `MODEL_ARGS` 的真实 vocab size |
| HF 导出目录已存在 | 是否需要覆盖 | `tools/convert_torch_dist_to_hf.py` L209-L210 | 加 `-f` 或换输出目录 |
| 导出脚本不知道模型类型 | 是否传 `--model-name` 或 `--origin-hf-dir` | `tools/convert_torch_dist_to_hf.py` L212-L219 | 传原始 HF 目录或显式 model name |
| FP8 rollout 后训练权重不对 | 是否误把 FP8 HF 当 `--ref-load` | `docs/en/get_started/quick_start.md` L392-L405 | `--hf-checkpoint` 可换 FP8，`--ref-load` 仍用 bf16 torch_dist |

## Q1：为什么 `--hf-checkpoint` 不能直接作为 `--ref-load`？

因为它们服务两个加载器。`--hf-checkpoint` 是 HF 目录，给 SGLang、tokenizer、AutoConfig；`--ref-load` 是 Megatron checkpoint，给 reference model 和 actor fallback。

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

判断方法：如果目录里主要是 `config.json`、tokenizer、safetensors，它是 HF 目录；如果目录由 Megatron tracker、`release/` 或 `iter_xxx/`、`common.pt`、dist metadata 组成，它才是训练 checkpoint。

## Q2：为什么必须 source `scripts/models/*.sh`？

因为 Megatron 不能只靠 checkpoint 目录恢复模型结构。它需要 CLI 超参来构图，再把 HF 权重灌进去。

```text
来源：docs/en/get_started/quick_start.md L118-L131
quick_start 要求通过 `source` 加载 `scripts/models/*.sh` 中的模型配置。
这些配置是 Megatron 构图所需的超参。
Megatron 不能直接从 checkpoint 读取全部模型配置。
文档要求检查 `--rotary-base` 等值是否与当前 HF 模型版本完全匹配。
```

Qwen3-4B 的关键结构参数包括层数、hidden、attention heads、GQA groups、RoPE base、vocab size：

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

排障动作：先把当前 HF `config.json` 与 `MODEL_ARGS` 对照，不要只看模型名称相同。

## Q3：多卡转换为什么会自动改 PP？

因为转换脚本要让每个 pipeline stage 分到合法层数。用户没有显式设置 PP 时，脚本从 world size 开始找可行值。

```python
# 来源：tools/convert_hf_to_torch_dist.py L44-L84
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

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)
```

失败模式：

- world size 大于层数：直接 assert。
- 自动推不出合法 PP：报 `Cannot find a valid pipeline model parallel size`。
- 手动设置 PP：脚本不会替你覆盖，错误会留给 Megatron validate 或后续 shape 检查。

## Q4：ROCm 为什么必须 `--use-cpu-initialization`？

HIP 环境下脚本先 patch checkpoint writer，然后断言 CPU 初始化。

```python
# 来源：tools/convert_hf_to_torch_dist.py L87-L120
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
```

```python
# 来源：tools/convert_hf_to_torch_dist.py L117-L120
# if using AMD gpus, we have to do the conversion in cpu
if hasattr(torch.version, "hip") and torch.version.hip is not None:
    assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"
```

排障动作：HIP 环境下转换命令显式加 `--use-cpu-initialization`，并确认日志出现 ROCm patch 信息。

## Q5：导出 HF 后 embedding 或 output layer 多了 padding 行怎么办？

传 `--vocab-size`。去 padding 只作用在两个张量：embedding 和 output layer。

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

导出命令示例：

```bash
python tools/convert_torch_dist_to_hf.py \
  --input-dir /root/Qwen3-4B_slime/release \
  --output-dir /root/Qwen3-4B_export_hf \
  --origin-hf-dir /root/Qwen3-4B \
  --vocab-size 151936
```

不要把 `padded_vocab_size` 当成 HF vocab size。这里要填 HF config 或 `MODEL_ARGS --vocab-size` 的真实词表大小。

## Q6：导出目录已存在为什么报错？

为了避免覆盖已有 HF 输出，脚本默认拒绝写入已存在目录。

```python
# 来源：tools/convert_torch_dist_to_hf.py L209-L210
if os.path.exists(args.output_dir) and not args.force:
    raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")
```

修复方法：

- 想保留旧结果：换 `--output-dir`。
- 确认要覆盖：加 `-f` 或 `--force`。

## Q7：没有 `--origin-hf-dir` 行不行？

可以，但你必须显式提供 `--model-name`，否则脚本不知道该用哪个 converter。

```python
# 来源：tools/convert_torch_dist_to_hf.py L212-L219
if args.model_name is None and args.origin_hf_dir is None:
    raise ValueError(
        "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
    )

if args.model_name is None:
    hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
    args.model_name = type(hf_config).__name__.lower()
```

实用建议：多数发布场景都传 `--origin-hf-dir`，因为它还能复制 tokenizer、config、generation config 等非权重 assets。

## Q8：FP8 HF rollout 要不要重新转 torch_dist？

通常不要。quick start 写得很明确：FP8 HF 只替换 `--hf-checkpoint`，Megatron `--ref-load` 仍然使用 bf16 HF 转出的 torch_dist。

```text
来源：docs/en/get_started/quick_start.md L392-L405
Slime 支持 bf16 training 与 fp8 inference 组合。
FP8 rollout 只需要把 `--hf-checkpoint` 换成 FP8 HF 目录。
Megatron `--ref-load` 仍然使用从 bf16 HF 转出的 torch_dist。
```

判断标准：`--hf-checkpoint` 可以服务 rollout 初始化；`--ref-load` 必须服务 Megatron 训练。

## Q9：什么时候用离线导出、bridge 导出、parallel 导出、`--save-hf`？

| 方式 | 入口 | 适合场景 | 关键依赖 |
|------|------|----------|----------|
| 离线基础导出 | `tools/convert_torch_dist_to_hf.py` | 单机读取一个 Megatron checkpoint，生成 HF 目录 | `common.pt`、dist metadata、可用 converter |
| bridge 导出 | `tools/convert_torch_dist_to_hf_bridge.py` | 想复用 mbridge/provider 逻辑 | custom provider、origin HF |
| parallel 导出 | `tools/convert_torch_dist_to_hf_parallel.py` | 大 checkpoint 导出需要并行处理和合并 index | 多进程/分片规划 |
| 训练中导出 | `--save-hf` | actor 保存模型时同步产出 HF | `Actor.save_model`、`hf_checkpoint_saver` |

训练中导出入口：

```python
# 来源：slime/backends/megatron_utils/actor.py L571-L578
save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

if force_sync and self.args.async_save:
    maybe_finalize_async_save(blocking=True)

if self.args.save_hf is not None and self.role == "actor":
    save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)
```

选择原则：已有 checkpoint 就离线导出；希望训练保存时就带 HF 输出，用 `--save-hf`；新架构或 provider 特殊时看 bridge 路径。

## Q10：这个专题和 [[Slime-Megatron到HF转换]] 怎么分工？

Tools-DataPrep 讲训练前后的 CLI 和目录形态：HF→torch_dist、CKPT_ARGS、torch_dist→HF。[[Slime-Megatron到HF转换]] 讲运行时 Megatron 到 HF naming/converter 逻辑，以及它如何服务权重同步或保存。

判断方法：如果你在排查“路径该填什么、目录里缺什么、convert 命令为什么失败”，看本专题；如果你在排查“某个 Megatron 参数名怎么变成 HF 参数名、某个架构 converter 怎么写”，看 [[Slime-Megatron到HF转换]]。
