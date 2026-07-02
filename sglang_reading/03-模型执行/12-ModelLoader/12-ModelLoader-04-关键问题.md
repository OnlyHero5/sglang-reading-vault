---
type: batch-doc
module: 12-ModelLoader
batch: "12"
doc_type: faq
title: "ModelLoader：关键问题"
tags:
 - sglang/batch/12
 - sglang/module/model-loader
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# ModelLoader：关键问题

## Q1：AUTO load_format 如何选文件类型？

**Explain：** DefaultModelLoader 优先 safetensors index，其次 .bin；`fall_back_to_pt` 控制是否允许 PyTorch bin。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L373-L374
# 提交版本：70df09b
        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""
```

---

## Q2：TP rank 如何只加载自己的分片？

**Explain：** 不在 Loader 层切分，而在 `ParallelLinear.weight_loader` / `default_weight_loader` 中按 `tp_rank` slice loaded tensor。

**易错 vs 正确：**

```python
# ❌ 每个 rank 加载完整 checkpoint 再手动 narrow（浪费 IO）
full = load_entire_checkpoint()
param.copy_(full[narrow_slice])

# ✅ iterator + weight_loader 在写入 param 时 narrow
for name, w in safetensors_weights_iterator(files):
 weight_loader(param, w) # ColumnParallelLinear 内部 slice
```

---

## Q3：加载后为什么要 monkey_patch_vllm_parallel_state？

**Explain：** 部分 linear/quant 代码复用 vLLM 并行状态 API，加载前 patch 使 rank 与 SGLang 一致。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L1462-L1463
# 提交版本：70df09b
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()
```

---

## Q4：RemoteInstance 与 RemoteModel 区别？

**Explain：** `RemoteInstanceModelLoader` 从**另一 SGLang 实例**拉；`RemoteModelLoader` 从**远程存储 connector**（S3 等）读。

---

## Q5：量化权重名 remap

**Code：**

```python
# 来源：python/sglang/srt/model_loader/weight_utils.py L216-L221
# 提交版本：70df09b
def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:
    for prefix, new_prefix in prefix_mapping.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_prefix, 1)
    return key

```

---

## Q6：enable_multithread_load 收益与风险

**Explain：** 多线程读 safetensors 加速 NVMe；`num_threads` 默认 8，过高可能 CPU 争抢。

**Code：**

```python
# 来源：python/sglang/srt/model_loader/loader.py L355-L356, L394-L395
# 提交版本：70df09b
    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8
```

---

## 验证建议（零基础可试）

1. **AUTO 优先读 safetensors 还是 .bin** 
 - 操作：打开本地 HF 模型目录，看是否存在 `model.safetensors.index.json` 或 `*.safetensors`；若仅有 `pytorch_model.bin` 且无 safetensors，对照 Q1 理解 `fall_back_to_pt`。 
 - 预期：同时存在时 DefaultModelLoader 走 safetensors iterator；仅 `.bin` 且 `fall_back_to_pt=True` 才回退 PyTorch 格式。 
 - 对应：Q1、[[12-ModelLoader-01-核心概念|01-核心概念 §LoadFormat]]、[[12-ModelLoader-02-源码走读|02-源码走读 §DefaultModelLoader]]

2. **量化权重名 remap（replace_prefix）** 
 - 操作：在 Python 里 `from sglang.srt.model_loader.weight_utils import replace_prefix`，调用 `replace_prefix("model.layers.0.q_proj.weight", {"model.": "language_model."})`。 
 - 预期：返回 `"language_model.layers.0.q_proj.weight"`；无前缀匹配时原样返回。 
 - 对应：Q5、[[12-ModelLoader-02-源码走读|02-源码走读 §weight_utils]]

3. **RemoteInstance 与 RemoteModel 别混** 
 - 操作：读 [[12-ModelLoader-03-数据流与交互|03-数据流与交互 §2]] IO 表，或 grep `RemoteInstanceModelLoader` / `RemoteModelLoader` 的 docstring；口述二者数据源差异。 
 - 预期：Instance = 从**另一台 SGLang 进程**拉权重；Model = 从 **S3/远程存储 connector** 读 checkpoint。 
 - 对应：Q4、[[12-ModelLoader-00-MOC|README §本模块位置]]

4. **多线程读盘配置是否生效** 
 - 操作：启动前加 `--load-format auto` 并在 extra config 里设 `enable_multithread_load=true`（或读 `LoadConfig` 默认值 `DEFAULT_NUM_THREADS=8`）；大模型目录下对比开启/关闭的加载阶段 wall time（无需 forward）。 
 - 预期：NVMe + 多 shard safetensors 时开启通常更快；`num_threads` 过高可能 CPU 争抢反而变慢。 
 - 对应：Q6、[[12-ModelLoader-03-数据流与交互|03-数据流与交互 §1]]
