---
title: "磁盘权重同步 · 排障指南"
type: troubleshooting
framework: slime
topic: "磁盘权重同步"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# 磁盘权重同步 · 排障指南

## 读者任务

这篇按症状排障。每个问题都给出源码入口和验证抓手，避免只记住结论。

## Q1：应该选 full disk、delta disk、colocate tensor 还是 NCCL？

| 症状或条件 | 优先路径 | 验证抓手 |
|------------|----------|----------|
| 跨机房或防火墙阻断 NCCL | delta disk 或 full disk | 启动参数里 `--update-weight-transport disk` |
| 先要最稳地跑通 | full disk | `weight_vNNNNNN` 下有完整 HF index 和 safetensors |
| 每轮变更稀疏且共享盘带宽紧 | delta disk | `perf/update_weights_density` 明显低于 1 |
| 训练和 rollout 同机 | colocate tensor | 日志中走 `update_weights_from_tensor` |
| 低延迟网络且通信组可建 | NCCL | 详见 [[Slime-分布式权重同步-排障指南]] |

源码入口仍是 actor 选型：

```python
# 定位骨架（基于 slime/backends/megatron_utils/actor.py L139-L161；省略 updater 构造）
if self.args.colocate:
    assert (
        self.args.update_weight_mode == "full"
    ), "--update-weight-mode=delta is not supported with --colocate"
    update_weight_cls = UpdateWeightFromTensor
elif self.args.update_weight_mode == "delta":
    assert (
        self.args.update_weight_transport == "disk"
    ), "--update-weight-mode=delta requires --update-weight-transport=disk"
    from .update_weight.update_weight_from_disk_delta import UpdateWeightFromDiskDelta

    update_weight_cls = UpdateWeightFromDiskDelta
else:
    assert self.args.update_weight_mode == "full"
    if self.args.update_weight_transport == "disk":
        update_weight_cls = UpdateWeightFromDisk
    else:
        assert (
            self.args.update_weight_mode == "full" and self.args.update_weight_transport == "nccl"
        ), f"unsupported weight sync mode/transport: {self.args.update_weight_mode!r}/{self.args.update_weight_transport!r}"
        update_weight_cls = UpdateWeightFromDistributed
```

## Q2：delta 首轮 update 后 engine 权重没变，是不是失败？

不是。delta 首轮只 capture baseline，不发布新版本。这保证下一轮 diff 的 base 和各 host 本地 HF 副本一致。

```python
# 定位骨架（基于 slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L80-L96；省略注释与尾部阶段）
@torch.no_grad()
def update_weights(self) -> None:
    if not self._baseline_captured:
        self._capture_baseline()
        self._baseline_captured = True
        return

    self.weight_version += 1
    if dist.get_rank() == 0:
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())

    self._publish()
    self._reload_engines()
    self._record_metrics()
```

检查方式：看日志是否出现 captured baseline；下一次 sync 才应看到 `weight_v000001` 或对应 density 日志。

安全前提是 engine 初始 base 与 actor 初始策略一致。若 actor 从不同 Megatron checkpoint 恢复，首轮 baseline capture 不会推送该差异，第一轮 rollout 可能仍使用 `hf_checkpoint` 权重。

## Q3：checksum mismatch 怎么排？

按四个状态检查：

| 检查项 | 文件或字段 |
|--------|------------|
| 本地已 apply 版本 | `local_checkpoint/.delta_sync/state.json` |
| delta base | `weight_vN/model.safetensors.index.json` 的 `base_version` |
| 编码一致性 | index 的 `delta_encoding` 与 trainer 参数 |
| 共享盘可见性 | `custom_delta_pre_read_path` 是否刷新 mount |

源码会在乱序时直接失败，不会继续 reload 坏权重：

```python
# 定位骨架（基于 slime/utils/disk_delta.py L155-L164；省略 compression/encoding 校验）
def _apply_version(local_ckpt_dir: str, version_dir: str) -> None:
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        meta = json.load(f)["metadata"]
    applied = _read_applied_version(local_ckpt_dir)
    if applied == meta["version"]:
        return
    if applied != meta["base_version"]:
        raise RuntimeError(f"out-of-order delta: local at {applied}, delta builds on {meta['base_version']}")
```

但 checksum mismatch 发生在原地 mmap 写之后，没有 rollback。overwrite 同版重放通常可修复；XOR 对已成功区域再次应用会翻回旧值，必须删除并从 base 重新 materialize host-local checkpoint。

## Q4：共享文件系统读到旧文件怎么办？

delta 路径提供两个 hook：

| Hook | 时机 | 用途 |
|------|------|------|
| `custom_delta_pre_push_path` | trainer 清空旧 delta dir 或写完新版本后 | commit 或 flush 共享目录 |
| `custom_delta_pre_read_path` | host apply 前 | 刷新对象存储或远端 mount |

engine actor 在 apply 前会调用 pre-read hook：

```python
# 定位骨架（基于 slime/backends/sglang_utils/sglang_engine.py L396-L413；省略 apply 参数换行）
def sync_local_checkpoint(self, target_version: int):
    from slime.utils.disk_delta import apply_deltas, init_local_checkpoint

    init_local_checkpoint(self.args.update_weight_local_checkpoint_dir, self.args.hf_checkpoint)
    if self.args.custom_delta_pre_read_path:
        from slime.utils.misc import load_function

        load_function(self.args.custom_delta_pre_read_path)(self.args.update_weight_disk_dir, target_version)
    apply_deltas(
        self.args.update_weight_local_checkpoint_dir,
        self.args.update_weight_disk_dir,
        target_version,
    )
```

## Q5：delta 能不能和 colocate 一起开？

不能。`colocate` 路径只支持 full mode，因为它直接传 tensor bucket，不维护 host-local delta 版本链。启动时会 assert。

验证方式：如果你看到 `--update-weight-mode=delta is not supported with --colocate`，不是运行时网络问题，而是配置组合不成立。

## Q6：`density` 和 `wire_bytes` 如何解读？

| 指标 | 含义 | 典型判断 |
|------|------|----------|
| `perf/update_weights_density` | changed bytes / total bytes | 越低越适合 delta |
| `perf/update_weights_wire_bytes` | 压缩后 safetensors 字节数 | 应显著小于 full checkpoint |

```python
# 定位骨架（基于 slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L271-L290；省略日志格式化）
def _record_metrics(self) -> None:
    counts = torch.tensor(
        [self.changed_bytes, self.total_bytes, self.wire_bytes],
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(counts)
    changed, total, wire = counts.tolist()
    m = self.update_weight_metrics
    m["perf/update_weights_density"] = changed / max(total, 1)
    m["perf/update_weights_wire_bytes"] = wire
    if dist.get_rank() == 0:
        logger.info(
            "[disk delta v=%s] density=%.2f%% wire=%.2f GB",
            self.weight_version,
            100.0 * changed / max(total, 1),
            wire / 1e9,
        )
```

## Q7：colocate IPC 后显存不释放怎么查？

看每个 HF chunk 后是否走到 `torch.cuda.ipc_collect()`；如果 Ray call 未返回或 consumer 没关闭 IPC handle，显存会滞留。

```python
# 定位骨架（基于 slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L147-L191；省略量化参数与注释）
for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
    refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
    ray.get(refs)
    del long_lived_tensors, hf_named_tensors
    torch.cuda.ipc_collect()

dist.barrier(group=get_gloo_group())
torch.cuda.ipc_collect()

if rank == 0:
    if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
        post_process_weights(
            restore_weights_before_load=False,
            post_process_quantization=True,
            rollout_engines=self.rollout_engines,
        )
    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
dist.barrier(group=get_gloo_group())
```

该清理只在成功路径执行；Ray/NCCL 异常没有 finally。混合 colocate+远端模式还要注意：pause/flush/continue 与量化 post-process 只覆盖 colocated 子集，远端 engine 只收到 distributed payload。

## Q8：如何验证 full disk 路径真的写盘？

仓库有 E2E smoke test：它启动小模型训练，打开 full disk update，并断言版本目录、HF index 和 safetensors 文件存在。

```python
# 定位骨架（基于 tests/test_full_disk_weight_update.py L91-L133；摘取配置、执行与产物断言）
disk_update_args = (
    "--update-weight-mode full "
    "--update-weight-transport disk "
    f"--update-weight-disk-dir {disk_dir} "
    "--update-weight-disk-keep-files "
)

ci_args = "--ci-test "

U.execute_train(
    train_args=train_args,
    num_gpus_per_node=NUM_GPUS,
    megatron_model_type=MODEL_TYPE,
)

checkpoint_dirs = sorted(Path(disk_dir).glob("weight_v*"))
assert checkpoint_dirs, f"No disk checkpoint directories were written under {disk_dir}"
assert any((path / "model.safetensors.index.json").exists() for path in checkpoint_dirs)
assert any(list(path.glob("*.safetensors")) for path in checkpoint_dirs)
```

建议命令：

```powershell
Set-Location slime
python -m pytest tests/test_full_disk_weight_update.py -q
```

这是 4-GPU、需要下载模型/数据并启动完整训练的 E2E，不是普通 CPU 单测。环境不足时只能做静态入口核验与运行 `tests/test_empty_colocated_weight_bucket.py`，不能宣称 full disk E2E 已通过。

## Q9：不要混淆 Slime delta 和 SGLang delta load_format

本专题的 delta 是 Slime trainer-side 协议：host apply 成完整 HF 目录后，engine 走普通 disk reload。`SGLangEngine.update_weights_from_disk` 虽然有 `load_format` 和 `files` 字段，但 Slime disk delta 主线传的是本地完整目录。

```python
# 定位骨架（基于 slime/backends/sglang_utils/sglang_engine.py L415-L437；省略可选字段分支）
def update_weights_from_disk(
    self,
    model_path: str,
    load_format: str | None = None,
    weight_version: str | None = None,
    files: list[str] | None = None,
):
    payload: dict = {"model_path": model_path}
    if load_format is not None:
        payload["load_format"] = load_format
    if weight_version is not None:
        payload["weight_version"] = weight_version
    if files is not None:
        payload["files"] = files
    return self._make_request("update_weights_from_disk", payload)
```

## Q10：delta 失败后为什么不能直接继续下一版？

trainer 在 `_encode_delta.collect` 中先更新内存 `_snapshot`，之后才写文件、执行 commit hook、让 host apply 和 engine reload。如果后半段失败，trainer 已认为 N 成功，host 可能仍在 N-1；下一次 version 递增后生成的是基于 N 的 diff，版本链已断。

恢复方式应是重建 updater/baseline，并确认所有 host-local checkpoint 回到同一已知版本。不要只修改 `state.json`，因为 mmap 文件可能已部分写入。

## Q11：delta 目录为什么持续增长？

delta updater 不消费 `update_weight_disk_keep_files` 做逐版删除；版本链会一直保留到下一次 baseline capture 清空整个 root。外部清理必须保留所有尚未被最慢 host apply 的版本，否则 `apply_deltas` 无法逐版追平。

## Q12：overwrite 有多大的 tensor 边界？

overwrite positions 被编码成 `<u4`，即 byte offset 为 32 位无符号数。单 tensor 原始 byte 长度达到或超过 2^32 时，源码没有显式拒绝，position cast 可能溢出。超大 tensor 应选 XOR、拆分张量，或先为编码器增加 64 位位置格式与版本标记。
