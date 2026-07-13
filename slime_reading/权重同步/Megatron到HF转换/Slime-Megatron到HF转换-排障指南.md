---
title: "Megatron到HF转换 · 排障指南"
type: troubleshooting
framework: slime
topic: "Megatron到HF转换"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# Megatron到HF转换 · 排障指南

本篇按症状排障。先判断问题发生在加载入口、模式选择、converter、参数重建，还是 safetensors 文件收敛。

## 症状速查

| 症状 | 可能原因 | 源码入口 | 验证方式 |
|------|----------|----------|----------|
| HF 目录加载时报 raw 不支持 | `--megatron-to-hf-mode` 不是 `bridge` | `_load_checkpoint_hf` | 检查启动参数和断言信息 |
| `--save-hf` 覆盖了模板目录 | 输出目录等于 `--hf-checkpoint` | `save_hf_model_direct_to_path` | 单测 `test_save_hf_model_direct_to_path_rejects_origin_checkpoint` |
| raw 导出目录缺 config 或 tokenizer | raw 只能复制本地顶层普通文件，嵌套资产不会递归 | `_copy_hf_assets` | 检查模板与输出目录的资产文件 |
| safetensors index 指向缺失文件 | writer state 或 finalize 异常 | `_finalize_shard_files` | 检查 `model.safetensors.index.json` 与 shard 文件集合 |
| duplicate HF tensor | converter 把多个 Megatron 参数映射到同名 HF tensor | `_SafetensorShardWriter.write` | 搜索 converter 输出名 |
| unsupported model | `model_name` 没命中路由 | `_convert_to_hf_core` | 打印广播后的 `model_name` |
| QKV shape 不对 | GQA group、head dim、bias 拆分参数不匹配 | `convert_qwen2_to_hf` | 对照 `num_attention_heads`、`num_query_groups`、`kv_channels` |
| 大模型加载很慢 | ShardedTensor metadata 校验或 checkpoint 文件系统压力 | checkpoint import patch | 看日志阶段和 torch distributed shard 版本 |
| 目录非空但 HF 无法加载 | finalize 中途失败，index 缺失或指向缺 shard | `_finalize_shard_files` | 逐项校验 index 的 `weight_map` |
| 多节点只有某些机器写盘 | rank 没按 node 连续排列，或 node/GPU 配置与实际拓扑不符 | `_get_node_save_layout` | 记录 global rank、hostname 和 writer ranks |
| HF 导出失败后 GPU 仍保持 wake | `save_model` 的 `sleep()` 不在 `finally` | `MegatronTrainRayActor.save_model` | 检查异常后 offload 状态 |
| 调小 buffer 仍 OOM | 单个 full param 已大于软阈值 | `_get_megatron_local_param_info_buckets` | 计算最大 `info.size * tp_size` |

## Q1：为什么 HF 加载只能用 bridge？

HF 目录没有 Megatron optimizer、scheduler、RNG 和 iteration 状态。Slime 的 raw converter 是 Megatron 到 HF 的导出路径，不负责把 HF 权重灌回 Megatron DDP model。加载 HF 时必须通过 Megatron Bridge。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/checkpoint.py` L129-L152；只展示入口）
def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")
```

排障动作：如果你要从 HF 初始化训练，设置 `--megatron-to-hf-mode bridge`；如果你要导出 HF 供 SGLang 使用，可以选择 bridge 或 raw。

安全动作：Bridge 加载和保存都使用 `trust_remote_code=True`。不要把未审计的 HF 目录当作纯数据投入训练进程。另外，HF 分支不传递 `skip_load_to_model_and_opt`，所以不能用该标志期待“只读 metadata 不灌权重”。

## Q2：`--load` 和 `--hf-checkpoint` 到底有什么区别？

路径误判时，先看这个极小分类器，而不是先查权重内容：

```python
# 来源：slime/backends/megatron_utils/checkpoint.py L114-L126
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )
```

| 参数 | 角色 | 参与哪条路径 |
|------|------|--------------|
| `--load` | 模型进入训练的权重来源 | Megatron checkpoint 恢复或 HF bridge 初始化 |
| `--hf-checkpoint` | raw 保存时的 HF 资产模板 | 提供 config/tokenizer，并可推断 model name 与量化配置 |
| `--save-hf` | 训练中额外导出的 HF 输出目录模板 | actor save 后调用 saver |

输出目录不能等于模板目录，否则 raw 保存会拒绝执行。

但源码只拒绝 resolve 后完全相等，没拒绝父子目录。实践中应使用两棵独立目录树；同时确认 config/tokenizer 都在模板顶层，因为 `_copy_hf_assets` 不递归复制子目录。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/hf_checkpoint_saver.py` L45-L64；截至模板目录校验）
def save_hf_model_direct_to_path(
    args,
    output_dir: str | Path,
    model,
    *,
    model_name: str | None = None,
    quantization_config: dict[str, Any] | None = None,
    progress_desc: str = "Save HF checkpoint",
) -> None:
    """Save a Megatron model as an HF safetensors checkpoint without Megatron Bridge."""
    path = Path(output_dir)
    hf_checkpoint = Path(args.hf_checkpoint).resolve()
    save_path = path.resolve()
    if hf_checkpoint == save_path:
        raise ValueError("HF save output path must not point to the same directory as --hf-checkpoint")
```

## Q3：新增模型族要改哪里？

最小改动路径：

1. 在 `slime/backends/megatron_utils/megatron_to_hf/` 增加模型 converter。
2. 在 `_convert_to_hf_core` 按子串匹配顺序注册模型名。
3. 覆盖 fused QKV、MLP gate/up、norm、embedding、output head 的 HF 目标命名。
4. 跑 raw saver 测试，再用一个小模型目录实际 `from_pretrained` 加载产物。

源码路由会在未知模型上直接失败。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/megatron_to_hf/__init__.py` L38-L66；只展示路由前半段）
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

## Q4：QKV 转换为什么是高风险点？

Megatron fused QKV 的排布包含 GQA 信息。Qwen2 converter 先计算 `head_dim` 和每个 query group 内的 Q 数量，再 reshape 和 split。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/megatron_to_hf/qwen2.py` L25-L47；只展示 QKV 权重分支）
        elif rest == "self_attention.linear_qkv.weight":

            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
```

排障动作：不要直接按三等分拆 QKV；先确认 `num_attention_heads % num_query_groups == 0`，再检查 reshape 后的维度是否与 HF config 一致。

## Q5：多节点保存为什么只让部分 rank 写文件？

raw saver 通过 `_get_node_save_layout` 每个 node 选 writer rank，chunk 按 node 轮转。这样避免所有 rank 竞争同一输出目录，也分摊磁盘写入。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/hf_checkpoint_saver.py` L361-L377；省略返回行）
def _get_node_save_layout(args) -> tuple[int, int, bool, list[int]]:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return 1, 0, True, [0]

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gpus_per_node = int(getattr(args, "actor_num_gpus_per_node", None) or getattr(args, "num_gpus_per_node", 1) or 1)
    gpus_per_node = max(1, gpus_per_node)
    inferred_nodes = max(1, math.ceil(world_size / gpus_per_node))
    configured_nodes = int(getattr(args, "actor_num_nodes", None) or inferred_nodes)
    num_nodes = max(1, min(configured_nodes, inferred_nodes))
```

排障动作：如果 shard 数量或 writer 分布异常，先核对 `actor_num_gpus_per_node`、`num_gpus_per_node`、实际 world size。

再做一次拓扑反证：在每个 rank 打印 `(global_rank, hostname)`，确认同一节点的 rank 是连续区间。该函数只根据 world size 和 GPUs-per-node 做算术推断，不识别真实 node identity。

## Q6：为什么会出现 duplicate HF tensor？

这是 converter 级别的问题。writer 只负责拦截：同名 tensor 在历史 shard 或当前 shard 中出现过，就立即报错。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/hf_checkpoint_saver.py` L175-L216；只展示重复检查）
class _SafetensorShardWriter:
    def __init__(self, path: Path, *, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.total_size = 0
        self.weight_map: dict[str, str] = {}
        self.shard_files: list[str] = []

    def write(self, named_tensors, shard_idx: int) -> None:
        if not self.enabled:
            return
        assert shard_idx is not None, "shard_idx must be set when writing HF shards"

        from safetensors.torch import save_file

        state_dict = {}
        total_size = 0
        for name, tensor in named_tensors:
            if name in self.weight_map or name in state_dict:
                raise ValueError(f"Duplicate HF tensor while saving: {name}")
```

排障动作：在对应模型 converter 中搜索重复返回的 HF 名称，尤其是 fused 参数拆分、shared expert、MoE router、output head 这类一对多或多对一位置。

## Q7：大模型加载为什么 patch ShardedTensor？

`checkpoint.py` import 时 patch PyTorch sharded tensor 的部分 metadata 校验，目的是减少大模型多 shard 加载成本。这个 patch 依赖调用方保证 shard metadata 正确。

```python
# 定位骨架（基于 `slime/backends/megatron_utils/checkpoint.py` L13-L88；只展示 import-time patch 入口）
try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
```

排障动作：如果是 checkpoint metadata 本身损坏，patch 可能让错误推迟到更下游；需要回到原始 checkpoint 生成过程排查。

这不是局部 context manager：`checkpoint.py` 被 import 时会直接替换 PyTorch 类方法，影响当前 Python 进程之后的相关 ShardedTensor 初始化。只有 `ImportError` 被吞掉；如果 PyTorch 内部 API 存在但签名或行为漂移，可能以非 `ImportError` 方式失败。升级 PyTorch 后应把 checkpoint 加载回归当作必测项。

## Q8：为什么“Megatron checkpoint 已保存”不等于这次 actor save 完全成功？

`save_model` 先调 Megatron `save`，后调 HF saver。HF 导出失败时，Megatron checkpoint 可能已成功落盘，但 actor 方法仍会向上抛异常，形成“部分成功”。offload 模式下，`wake_up()` 与尾部 `sleep()` 之间没有 `try/finally`；中间任意保存异常都可能跳过 sleep。

排障动作：分别检查 Megatron tracker/iteration 目录和 HF index，不要用其中一方的存在替代另一方验收；异常后显式确认 actor 是否恢复 offload 状态。

## Q9：如何判断 raw 目录是否真正发布完成？

raw saver 直接写最终目录，没有整目录 staging 和原子 rename。最小检查顺序是：

1. `model.safetensors.index.json` 存在且 JSON 可解析。
2. `weight_map` 中每个文件名都存在。
3. 目录中没有遗留 `model-00001.safetensors` 这类未 finalize 的临时名。
4. 用目标 Transformers/SGLang 版本做真实 load，再用已知输入做最小前向。

## Q10：CI 里哪些测试覆盖 raw saver？

`tests/utils/test_hf_checkpoint_saver.py` 覆盖六类关键行为：资产复制跳过权重、清理旧权重、拒绝覆盖模板目录、writer 生成 index、多 writer state finalize、pending chunk flush。

```python
# 定位骨架（基于 `tests/utils/test_hf_checkpoint_saver.py` L21-L133；只展示首个测试的 arrange 段）
def test_copy_hf_assets_keeps_quantized_config_and_skips_weights(tmp_path: Path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    config = {"model_type": "tiny", "quantization_config": {"quant_method": "fp8"}}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (src / "tokenizer.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
```

本地验证：

```powershell
python -m pytest tests/utils/test_hf_checkpoint_saver.py
python -m pytest tests/utils/test_megatron_bridge_utils.py
```

覆盖边界：这些是 helper 级 CPU 测试，不证明真实 TP/PP/EP collective、多机 rank 布局、各模型 converter 的数值正确性，也不证明导出目录已被 `from_pretrained` 或 SGLang 成功加载。
