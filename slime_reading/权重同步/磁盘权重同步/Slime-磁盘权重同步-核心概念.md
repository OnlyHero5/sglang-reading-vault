---
title: "磁盘权重同步 · 核心概念"
type: concept
framework: slime
topic: "磁盘权重同步"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-13
---
# 磁盘权重同步 · 核心概念

## 读者任务

这篇先建立心理模型：权重同步不是单一 API，而是把“训练侧新权重”投递到“rollout engine 可服务权重”的四种运输方式。你读完应能用部署条件选择路径，并知道每条路径最容易坏在哪里。

## 先建立模型：三本账

| 账本 | 谁维护 | 关键字段 | 失败后表现 |
|------|--------|----------|------------|
| 选型账 | `MegatronTrainRayActor` | `colocate`、`update_weight_mode`、`update_weight_transport` | 启动即 assert 或走错 updater |
| 版本账 | updater 与 engine | `weight_version`、`weight_vNNNNNN`、`.delta_sync/state.json` | engine 版本落后、delta 顺序错误 |
| 介质账 | shared FS / local NVMe / IPC | HF safetensors、zstd delta、flattened bucket | 慢、checksum mismatch、IPC 显存占用 |

## 四条路径

| 路径 | 触发条件 | 数据介质 | 适合场景 |
|------|----------|----------|----------|
| full disk | `mode=full` 且 `transport=disk` | 共享目录中的完整 HF checkpoint | NCCL 不可用，先要稳定跑通 |
| delta disk | `mode=delta` 且 `transport=disk` | 共享目录中的压缩 byte diff | 跨机房或共享盘带宽有限，单轮变化稀疏 |
| colocate tensor | `colocate=true` 且 `mode=full` | Gloo gather + Ray IPC flattened bucket | 训练与 rollout 同机，追求低延迟 |
| distributed NCCL | `mode=full` 且 `transport=nccl` | NCCL weight update group | 低延迟网络和通信组可用，详见 [[Slime-分布式权重同步]] |

源码入口把 delta 固定到 disk，并明确禁止 colocate + delta：

```python
# 定位骨架（基于 slime/backends/megatron_utils/actor.py L139-L149；省略 import 与 updater 构造）
if self.args.colocate:
    assert (
        self.args.update_weight_mode == "full"
    ), "--update-weight-mode=delta is not supported with --colocate"
    update_weight_cls = UpdateWeightFromTensor
elif self.args.update_weight_mode == "delta":
    assert (
        self.args.update_weight_transport == "disk"
    ), "--update-weight-mode=delta requires --update-weight-transport=disk"
```

## Full Disk：把同步问题变成 HF 目录版本

full disk 每轮递增 `weight_version`，写入 `update_weight_disk_dir/weight_vNNNNNN`，再让每个 rollout engine 调 `update_weights_from_disk`。它的优势是语义清楚，缺点是每轮都写完整 checkpoint。

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk.py L61-L98
@torch.no_grad()
def update_weights(self) -> None:
    self.weight_version += 1
    version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"

    if dist.get_rank() == 0:
        shutil.rmtree(version_dir, ignore_errors=True)
    dist.barrier(group=get_gloo_group())

    if dist.get_rank() == 0:
        logger.info("Updating rollout weights from disk checkpoint %s", version_dir)
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())

    save_hf_model_to_path(
        self.args,
        version_dir,
        self.model,
        model_name=self.model_name,
        quantization_config=self.quantization_config,
        progress_desc="Save HF  weights for update from disk",
    )
    dist.barrier(group=get_gloo_group())

    if dist.get_rank() == 0:
        refs = [
            engine.update_weights_from_disk.remote(
                model_path=str(version_dir),
                weight_version=str(self.weight_version),
            )
            for engine in self.rollout_engines
        ]
        ray.get(refs)
        if not self.args.update_weight_disk_keep_files:
            shutil.rmtree(version_dir, ignore_errors=True)
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
    dist.barrier(group=get_gloo_group())
```

目录不是以临时目录整体 rename 的方式发布，而是直接在最终 `weight_vN` 下写入；正确性依赖 engine 直到 save barrier 后才收到 reload。共享文件系统还必须提供跨 host 的可见性，full disk 没有 delta 路径那样的 pre-push/pre-read hook。整个 pause→save→reload→cleanup→continue 也没有 `try/finally`。

## Delta Disk：把全量目录拆成 baseline + 版本链

delta disk 的核心不在 SGLang，而在 Slime 侧：trainer 发布 `weight_vN` 的 diff，各 host 把 diff apply 到本地完整 HF 副本，然后 engine 仍然用普通 `update_weights_from_disk(local_dir)` reload。

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

首轮只 capture baseline 是刻意设计。baseline 从 `hf_checkpoint` 读取，因为各 host 的 local checkpoint 也是从同一个 HF 目录 materialize，二者 byte layout 必须一致。这同时构成安全前提：第一次 rollout 使用的 engine base 必须与 actor 要同步的初始语义一致；若 actor 从另一个 Megatron checkpoint 恢复而 engine 仍是旧 HF base，首轮不会把差异推过去。

```python
# 定位骨架（基于 slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L98-L124；省略日志与 barrier 尾部）
def _capture_baseline(self) -> None:
    if dist.get_rank() == 0:
        shutil.rmtree(self.delta_dir, ignore_errors=True)
        os.makedirs(self.delta_dir, exist_ok=True)
        if self._commit_hook is not None:
            self._commit_hook(self.args, self.delta_dir, list(self.rollout_engines))
    dist.barrier(group=get_gloo_group())

    read_hf = make_tensor_reader(self.args.hf_checkpoint)
    for name, tensor in self._iter_hf_tensors():
        try:
            self._snapshot[name] = read_hf(name)
        except KeyError:
            self._snapshot[name] = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().reshape(-1)
            logger.warning("seed: %s absent from hf_checkpoint; seeding from current weights", name)
```

## Delta 编码：xor 与 overwrite 的取舍

delta 是 byte-level，不关心 dtype。`xor` 在压缩前保存完整逐 byte 差分、apply 是逐块异或；`overwrite` 只记录变化位置和新 byte。谁的 wire 更小取决于变化密度与 zstd 可压缩性，不能脱离指标预设结论；语义上 XOR 只能在正确 base 上应用一次，overwrite 更适合同版重放。

```python
# 来源：slime/utils/disk_delta.py L29-L33
def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """The 'overwrite' delta: changed-position count (u4), positions (u4 each), then new values.
    Idempotent to apply, unlike xor (an involution); the trainer picks the encoding per the docs."""
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new[changed_mask]])
```

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_disk_delta.py L227-L239
if self.delta_encoding == "xor":
    diff = new ^ old
    changed = int(np.count_nonzero(diff))
elif self.delta_encoding == "overwrite":
    mask = new != old
    changed = int(np.count_nonzero(mask))
    diff = overwrite_encode(new, mask)
else:
    raise ValueError(f"unknown delta encoding {self.delta_encoding!r}")
if not changed:
    return name, new, None, None, 0
compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
return name, new, compressed, checksum(self.checksum_algorithm, new), changed
```

失败恢复差异比“幂等”更具体：apply 会原地修改 mmap，checksum 只在写后检查且没有回滚。XOR 若进程崩溃或部分 tensor 校验失败，本地文件可能处于半新半旧状态，再次完整 XOR 会把已经成功的区域翻回旧值；应从已知完整 base 重新 materialize。overwrite 可重复覆盖同一位置，但其 position 编码为 `<u4`，单 tensor byte offset 超过 2^32 时没有显式防护。

trainer 侧也不是事务：`collect()` 在 delta 文件写入、commit hook、host apply 与 engine reload 之前就更新 `_snapshot`。后续阶段失败时，内存 snapshot 已前移而 host 可能仍在旧版本，不能在同一 updater 上盲目继续生成下一版。

## Host-Local Checkpoint：每台 rollout host 自己追版本

delta 不要求 engine 理解 diff。每台 host 有一个本地 HF 副本，`.delta_sync/state.json` 记录已 apply 版本，`flock` 保证同 host 多 actor 串行 apply。

```python
# 定位骨架（基于 slime/utils/disk_delta.py L111-L124；省略日志与 page-cache 操作）
def init_local_checkpoint(local_ckpt_dir: str, base_dir: str) -> None:
    """Copy the base HF checkpoint into local_ckpt_dir once if absent (run at engine start). Each
    later delta is applied on top of this copy in place."""
    with _apply_lock(local_ckpt_dir):
        if _read_applied_version(local_ckpt_dir) is not None:
            return
        logger.info("Materializing base checkpoint %s -> %s", base_dir, local_ckpt_dir)
        os.makedirs(local_ckpt_dir, exist_ok=True)
        for entry in os.scandir(base_dir):
            if entry.is_file():
                shutil.copy2(entry.path, os.path.join(local_ckpt_dir, entry.name))
                drop_page_cache(entry.path)
        _write_applied_version(local_ckpt_dir, "000000")
```

`state.json` 只在整版校验通过后原子更新，但权重文件本身不回滚也不 `msync`；状态落后并不保证文件仍是旧版本。该实现依赖 Linux `fcntl.flock`、`mmap.madvise`/POSIX 文件语义，不是 Windows 可执行路径。

delta 版本目录不会读取 `update_weight_disk_keep_files` 做逐版清理；只在下一次 baseline capture 时清空整个 delta root。长任务要自行规划保留策略或由 commit hook 外部归档/清理，同时确保尚未追平的 host 仍能读取完整版本链。

## Colocate Tensor：不是磁盘路径，但必须对照

colocate full 不写 checkpoint。它把 HF tensor chunk 转成 flattened bucket，经 Gloo `gather_object` 收到对应 engine 的 src rank，再用 Ray IPC 发给 SGLang。

```python
# 来源：slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py L245-L287
for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
    flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
    metadata = flattened_tensor_bucket.get_metadata()
    flattened_tensor_data = {
        "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
        "metadata": metadata,
    }
    long_live_tensors.append(flattened_tensor_data)
    serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

serialized_named_tensors = (
    [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == dist.get_rank() else None
)
dist.gather_object(
    serialized_tensors,
    object_gather_list=serialized_named_tensors,
    dst=ipc_gather_src,
    group=ipc_gather_group,
)

refs = []
if dist.get_rank() == ipc_gather_src:
    num_buckets = max(len(tensors) for tensors in serialized_named_tensors)
    empty_serialized_tensor = None
    for i in range(num_buckets):
        serialized_tensors_for_dtype = []
        for tensors in serialized_named_tensors:
            if i < len(tensors):
                serialized_tensors_for_dtype.append(tensors[i])
                continue

            if empty_serialized_tensor is None:
                empty_tensor_data = _empty_flattened_tensor_data()
                long_live_tensors.append(empty_tensor_data)
                empty_serialized_tensor = MultiprocessingSerializer.serialize(empty_tensor_data, output_str=True)
            serialized_tensors_for_dtype.append(empty_serialized_tensor)

        kwargs = {
            "serialized_named_tensors": serialized_tensors_for_dtype,
            "load_format": "flattened_bucket",
            "weight_version": str(weight_version),
        }
        refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))
```

纯 colocate 路径会等待 Ray refs 后释放长寿命 tensor 并调用 `torch.cuda.ipc_collect()`；但异常路径没有 finally。混合远端模式中，`self.rollout_engines` 被切成 colocated 子集，pause/flush/continue 与 compressed-tensors post-process 只覆盖该子集，远端 engine 仅走 distributed send。

## 复盘

- full disk 简单可靠，但每轮写完整 HF checkpoint。
- delta disk 省线宽，但新增 baseline、版本链、checksum 和本地副本状态。
- colocate tensor 低延迟，但需要处理 IPC 生命周期和混合远端 engine。
- 选路径时先问通信拓扑，再问版本语义，最后问性能。
