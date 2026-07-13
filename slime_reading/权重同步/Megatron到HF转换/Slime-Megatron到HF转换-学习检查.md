---
title: "Megatron到HF转换 · 学习检查"
type: exercise
framework: slime
topic: "Megatron到HF转换"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# Megatron到HF转换 · 学习检查

## 读者能做什么

- [ ] 能画出 `--load -> load_checkpoint -> Megatron/HF branch -> model` 的加载路径。
- [ ] 能解释 `bridge` 和 `raw` 在加载与保存上的不对称能力。
- [ ] 能沿 `Actor.save_model -> save_hf_model_to_path -> HfWeightIteratorDirect -> convert_to_hf -> writer -> index` 复述一次导出。
- [ ] 能说明 `--hf-checkpoint` 是 raw 保存的资产模板，不是当前训练权重来源。
- [ ] 能指出 converter 改动会同时影响 `--save-hf`、[[Slime-磁盘权重同步]] 和 [[Slime-分布式权重同步]]。
- [ ] 能说出 3 个失败模式及入口：raw 加载 HF 被拒、duplicate HF tensor、QKV shape 不匹配。
- [ ] 能解释为什么 `update_weight_buffer_size` 是软阈值，并计算单个 full param 超限的情况。
- [ ] 能指出 raw saver 没有整目录原子发布，并给出 index/shard 完整性检查。
- [ ] 能说明 writer layout 对“global rank 按 node 连续排列”的假设。
- [ ] 能审计 `trust_remote_code=True`、import-time ShardedTensor patch 和 offload 异常后未 sleep 三个进程级边界。

## 可执行验证

raw saver 单测：

```powershell
Push-Location slime
python -m pytest tests/utils/test_hf_checkpoint_saver.py
python -m pytest tests/utils/test_megatron_bridge_utils.py
Pop-Location
```

转换入口静态定位：

```powershell
rg -n 'save_hf_model_to_path|HfWeightIteratorDirect|convert_to_hf|finalize|pending' slime/slime/backends/megatron_utils
rg -n 'megatron_to_hf_mode|hf_checkpoint|save_hf' slime/slime/utils/arguments.py slime/slime/backends/megatron_utils
```

发布边界静态检查：

```powershell
rg -n 'os\.replace|model\.safetensors\.index\.json|path\.mkdir|_clear_existing_hf_weights|_copy_hf_assets' slime/slime/backends/megatron_utils/hf_checkpoint_saver.py
rg -n 'rank // gpus_per_node|node \* gpus_per_node|update_weight_buffer_size|partition_stride' slime/slime/backends/megatron_utils
rg -n 'trust_remote_code=True|__post_init__ =|_init_from_local_shards_and_global_metadata =' slime/slime/backends/megatron_utils
```

如果有可用的真实小模型、CUDA 和共享文件系统，再做动态验收：

1. 用 raw 模式导出一个新目录。
2. 解析 index，确认每个 `weight_map` 目标都存在，且无临时 shard 名残留。
3. 用目标 Transformers/SGLang 版本加载，对固定 prompt 运行最小前向。
4. 对 converter 输出做 tensor name、shape、dtype 和数值对照。

无 GPU/模型环境时，上述静态检查与 helper 测试只能证明机制边界，不得写成真实分布式导出或模型数值已通过。

## 预期现象

- raw saver 六项测试与 Bridge config patch 六项测试通过；若当前 baseline 的测试数发生变化，以 pytest 收集结果为准，不为追求固定数字而忽略新测试。
- 静态结果能串出 actor 保存、iterator、模型 converter、writer/finalize 与 index 发布。
- 能解释模板资产与新权重 tensor 为什么必须分开处理。
- 能明确列出 helper 测试没有覆盖的真实 collective、多机拓扑、converter 数值和 `from_pretrained` 加载边界。

## 权重同步收口

完成本专题后，权重同步三段闭环应能连起来：

| 专题 | 读者应掌握 |
|------|------------|
| [[Slime-分布式权重同步]] | 在线 NCCL 同步如何把 HF tensor 流推给 rollout engine |
| [[Slime-磁盘权重同步]] | 磁盘版本目录如何发布、reload 和清理 |
| [[Slime-Megatron到HF转换]] | Megatron 权重如何导出为标准 HF checkpoint，并服务保存与 disk sync |
