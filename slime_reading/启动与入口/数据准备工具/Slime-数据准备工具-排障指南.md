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
updated: 2026-07-12
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
| 导出 HF 后 embedding 行数不对 | CLI vocab 与 `common.pt` 中 `args.vocab_size` 是否一致 | `save_tensors` L106-L117；`convert_to_hf` L25-L30 | 同时核对两级裁剪与 HF config |
| HF 导出目录已存在 | 是否误以为 `--force` 会清空目录 | `tools/convert_torch_dist_to_hf.py` L209-L210 | 首选新目录；复用前人工确认并清理旧产物 |
| 反向导出找不到 `common.pt` | `--input-dir` 是否误指 checkpoint 根目录 | `tools/convert_torch_dist_to_hf.py` L221-L230 | 指到具体 `release/iter_xxx` 目录 |
| 开启 missing 补权重后 index 的 `total_size` 异常 | 是否把 metadata 数字当作完整性唯一依据 | `tools/convert_torch_dist_to_hf.py` L128-L145 | 对照 `weight_map`、shard、tensor key/shape；当前累计位置可能重复计数 |
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

```bash
# 来源：docs/en/get_started/quick_start.md L121-L122
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/glm4-9B.sh"
```

紧随该代码的官方说明指出：这些是 Megatron 所需超参，并提醒逐项核对 `--rotary-base` 等版本敏感配置。

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
# 来源：tools/convert_hf_to_torch_dist.py L44-L68
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
# 来源：tools/convert_hf_to_torch_dist.py L87-L108
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
# 来源：tools/convert_hf_to_torch_dist.py L117-L119
# if using AMD gpus, we have to do the conversion in cpu
if hasattr(torch.version, "hip") and torch.version.hip is not None:
    assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"
```

排障动作：HIP 环境下转换命令显式加 `--use-cpu-initialization`，并确认日志出现 ROCm patch 信息。源码中的 process group backend 仍是 `nccl`；writer patch 与通信后端是两件事，不要把该日志解释成整条分布式栈已经切换后端。

## Q5：导出 HF 后 embedding 或 output layer 多了 padding 行怎么办？

先核对三处 vocab：HF config、CLI `--vocab-size`、checkpoint `common.pt` 中的 `args.vocab_size`。去 padding 只作用在 embedding 和 output layer，但基础脚本可能先按 CLI 值裁、再在 `convert_to_hf()` 中按 checkpoint 值裁；第二次切片不会把第一次裁掉的行恢复。

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

不要把 `padded_vocab_size` 当成 HF vocab size。这里要填真实词表大小，并检查 checkpoint 保存的 `args.vocab_size` 是否一致；若它已被错误地保存为更小值，单靠 CLI 不能扩回张量。

## Q6：导出目录已存在为什么报错？

脚本默认拒绝写入已存在目录，但其 `--force` 名字比真实行为更强：它只跳过下面的存在性检查。

```python
# 来源：tools/convert_torch_dist_to_hf.py L209-L210
if os.path.exists(args.output_dir) and not args.force:
    raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")
```

修复方法：

- 最稳妥：换一个全新的 `--output-dir`。
- 必须复用：先自行确认并清理旧权重分片和过期 assets，再加 `-f`；脚本本身不会删除它们。

这也是为什么“命令成功”仍不等于目录干净：新 index 可能只引用新分片，但旧文件仍滞留在目录里。

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

实用建议：多数发布场景都传 `--origin-hf-dir`，因为它还能复制 tokenizer、config、generation config 等顶层非权重 assets。子目录不会递归复制。

## Q8：FP8 HF rollout 要不要重新转 torch_dist？

通常不要。quick start 写得很明确：FP8 HF 只替换 `--hf-checkpoint`，Megatron `--ref-load` 仍然使用 bf16 HF 转出的 torch_dist。

```bash
# 来源：docs/en/get_started/quick_start.md L401-L405
   # Used to load tokenizer and other information, actually won't use model weight parameters from hf path
   --hf-checkpoint /root/Qwen3-4B-FP8

   # The megatron checkpoint still needs to be the dist weights converted from bf16 huggingface at the beginning, not modified because of FP8 rollout.
   --ref-load /root/Qwen3-4B_torch_dist
```

判断标准：`--hf-checkpoint` 可以服务 rollout 初始化；`--ref-load` 必须服务 Megatron 训练。

## Q9：什么时候用离线导出、bridge 导出、parallel 导出、`--save-hf`？

| 方式 | 入口 | 适合场景 | 关键依赖 |
|------|------|----------|----------|
| 离线基础导出 | `tools/convert_torch_dist_to_hf.py` | 单机读取一个 Megatron checkpoint，生成 HF 目录 | `common.pt`、dist metadata、可用 converter |
| bridge 导出 | `tools/convert_torch_dist_to_hf_bridge.py` | 让 Megatron Bridge 从 origin HF provider 恢复正确模型类并 export | `--origin-hf-dir` 必填；`--force` 同样不清目录 |
| parallel 导出 | `tools/convert_torch_dist_to_hf_parallel.py` | 用多进程分 key、线程并发加载/转换/保存 | 转换异常会被打印后返回空结果，必须额外核对 key 数与 index 完整性 |
| 训练中导出 | `--save-hf` | actor 保存模型后同步产出 HF | raw/bridge 两分支；实现与离线脚本不同 |

训练中导出入口：

```python
# 来源：slime/backends/megatron_utils/actor.py L571-L577
save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

if force_sync and self.args.async_save:
    maybe_finalize_async_save(blocking=True)

if self.args.save_hf is not None and self.role == "actor":
    save_hf_model_to_path(self.args, Path(self.args.save_hf.format(rollout_id=rollout_id)), self.model)
```

选择原则：基础离线脚本适合可由内置 naming converter 覆盖的具体 checkpoint 目录；Bridge 路径适合依赖 provider/model class 的架构；parallel 路径只在你愿意承担更强完整性校验时使用；训练中 `--save-hf` 是保存时直接从活模型导出，并非调用基础离线脚本。

## Q10：这个专题和 [[Slime-Megatron到HF转换]] 怎么分工？

Tools-DataPrep 讲训练前后的 CLI 和目录形态：HF→torch_dist、CKPT_ARGS、torch_dist→HF。[[Slime-Megatron到HF转换]] 讲运行时 Megatron 到 HF naming/converter 逻辑，以及它如何服务权重同步或保存。

判断方法：如果你在排查“路径该填什么、目录里缺什么、convert 命令为什么失败”，看本专题；如果你在排查“某个 Megatron 参数名怎么变成 HF 参数名、某个架构 converter 怎么写”，看 [[Slime-Megatron到HF转换]]。
