---
title: "磁盘权重同步 · 学习检查"
type: exercise
framework: slime
topic: "磁盘权重同步"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 磁盘权重同步 · 学习检查

## 读者能做什么

- [ ] 能画出 `Megatron actor → updater → shared FS / local checkpoint / IPC → SGLang engine` 的主线。
- [ ] 能说明 actor 选型中 `colocate`、`update_weight_mode`、`update_weight_transport` 的优先级。
- [ ] 能复述 full disk 的五步：pause、flush、save HF、reload、continue。
- [ ] 能解释 delta 首轮只 capture baseline 的原因。
- [ ] 能说明 `snapshot == engine base` 为什么要求 baseline 来自 `hf_checkpoint`。
- [ ] 能对比 `xor` 和 `overwrite` 的幂等性。
- [ ] 能定位 `.delta_sync/state.json`、`base_version`、`delta_encoding`、checksum mismatch 的排障入口。
- [ ] 能说明 `all_engine_actors` 和 `rollout_engines` 在 delta reload 中的不同职责。
- [ ] 能解释 colocate tensor 为什么要 `FlattenedTensorBucket`、`gather_object` 和 `torch.cuda.ipc_collect()`。

## 可执行验证

```bash
cd slime
pytest tests/test_full_disk_weight_update.py -q
```

预期现象：

- 测试启动 full disk weight update。
- `update_weight_disk_dir` 下出现 `weight_v*` 版本目录。
- 至少一个版本目录包含 `model.safetensors.index.json`。
- 至少一个版本目录包含 safetensors 权重分片。

## 排障演练

| 现象 | 应检查 |
|------|--------|
| delta 首轮 engine 没变 | 是否只是 `_capture_baseline` 首轮 |
| checksum mismatch | `.delta_sync/state.json` 与 index `base_version` |
| delta 文件 rank 编号缺口 | `_write_delta_files` 的 `all_gather_object` 计数 |
| full disk 很慢 | `save_hf_model_to_path` 耗时和共享盘带宽 |
| colocate 后显存不降 | `ray.get(refs)` 是否返回以及 `torch.cuda.ipc_collect()` 是否执行 |
| external FS 读旧版本 | `custom_delta_pre_push_path` / `custom_delta_pre_read_path` hook |

## 源码锚点

| 文件 | 关键符号 |
|------|----------|
| `slime/backends/megatron_utils/actor.py` | updater 选型和 delta/colocate assert |
| `slime/backends/megatron_utils/update_weight/update_weight_from_disk.py` | full disk `update_weights` |
| `slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py` | baseline、publish、reload、metrics |
| `slime/utils/disk_delta.py` | local checkpoint、version apply、checksum |
| `slime/backends/sglang_utils/sglang_engine.py` | `sync_local_checkpoint` 与 HTTP disk reload |
| `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | colocate IPC bucket 和混合远端路径 |

## 下一步

- 回到 [[Slime-分布式权重同步]] 对比 NCCL 权重同步。
- 继续读 [[Slime-Megatron到HF转换]]，理解 full disk 如何把 Megatron 权重保存成 HF checkpoint。
