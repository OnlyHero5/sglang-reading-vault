---
title: "Hopper与CuTe · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Hopper与CuTe · 排障指南

这不是一张“看到报错就换版本”的速查表，而是一套缩小故障域的方法。读完后，你应能把失败定位到包与 import、运行 arch、能力门禁、JIT cache 或 kernel 运行中的最早一层。FA2、FA3、FA4 可以同时存在；同一个 `FlashAttention` 名字背后，可能是不同安装包、Python 入口、GPU 架构分支、编译目标和缓存层。排障时先回答“究竟执行了谁”，再讨论 kernel 对不对、快不快。

## 排障入口：先建立五层故障模型

把一次调用想成经过五道闸门：

```mermaid
flowchart LR
    A[包与 import] --> B[运行 GPU arch]
    B --> C[功能与 dtype 门禁]
    C --> D[compile key 与 JIT cache]
    D --> E[kernel 运行与数值]
```

| 层 | 要回答的问题 | 典型现象 |
|---|---|---|
| 包与 import | 导入的是 FA2、FA3 compiled op，还是 FA4 CuTeDSL？ | `ModuleNotFoundError`、误以为会自动 fallback |
| 运行 arch | `_get_device_arch()` 选择了哪个对象族？ | 走到 SM80/90/100/120 的错误分支 |
| 能力门禁 | 当前 arch、dtype、paged KV、SplitKV、反向是否组成合法组合？ | Python `assert` 或 `NotImplementedError` |
| 编译与缓存 | 这组静态特征是进程内命中、磁盘命中，还是重新编译？ | 首次调用慢、多进程重复编译 |
| kernel 运行 | 编译后的对象是否在真实设备上正确执行？ | launch/illegal instruction、数值偏差、性能异常 |

这五层有严格顺序。import 都没有落到预期实现时，继续调 tile size 没有意义；能力门禁已经拒绝组合时，也不应把它描述成 FlashAttention 算法本身不支持。

## 症状一：不知道实际走的是 FA2、FA3 还是 FA4

### 可能原因

- 环境里安装了多个发行物，上层框架仍导入旧入口。
- 把“仓库里存在 FA4 源码”误当成“当前调用自动升级到 FA4”。
- 在 FA3 CUDA 上期待 import 失败后自动回退；当前同类 fallback 只出现在 HIP 路由。

### 源码入口

FA4 README 给出的独立安装和入口是 `flash-attn-4` 与 `flash_attn.cute`，不是对所有旧入口的透明替换。

````markdown
<!-- 来源：flash_attn/cute/README.md L3-L9 -->
FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

```sh
pip install flash-attn-4
```
````

FA3 的 HIP 路由会在 compiled extension 缺失时切换到 Triton；CUDA 分支则直接 import extension。

```python
# 来源：hopper/flash_attn_interface.py L11-L28
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if not USE_TRITON_ROCM and getattr(torch.version, 'hip', None) is not None:
    try:
        import flash_attn_3._C
    except ImportError:
        warnings.warn("flash_attn_3._C (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
        USE_TRITON_ROCM = True

if USE_TRITON_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3 as flash_attn_3_gpu
else:
    # isort: off
    # We need to import the CUDA kernels after importing torch
    import flash_attn_3._C # Registers operators with PyTorch

    # isort: on

    flash_attn_3_gpu = torch.ops.flash_attn_3
```

### 操作与预期

```powershell
@'
import importlib.util
for name in ["flash_attn", "flash_attn_3", "flash_attn.cute"]:
    try:
        spec = importlib.util.find_spec(name)
        print(name, None if spec is None else spec.origin)
    except Exception as exc:
        print(name, type(exc).__name__, str(exc))
'@ | python -
```

预期：输出每个可见模块的真实位置。`None` 只说明当前 Python 环境找不到该入口，不说明 GPU 或算法不支持。随后在上层框架的 import site 确认它实际引用哪个符号；不要仅凭 `pip list` 推断执行路径。

## 症状二：arch override 后走错分支或运行失败

### 可能原因

- `FLASH_ATTENTION_ARCH` 覆盖了运行时对象选择。
- 只设置了 `FLASH_ATTENTION_ARCH`，却把它误当成同时改变 CuTeDSL 编译目标。
- 为 CPU-only/cross compile 生成了目标对象，随后在不匹配的真实 GPU 上运行。

### 源码入口

运行分支和编译目标是两个旋钮：注释明确要求 CPU-only 编译时同时设置 `FLASH_ATTENTION_ARCH` 与 `CUTE_DSL_ARCH`。

```python
# 来源：flash_attn/cute/interface.py L77-L92
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)
```

### 操作与预期

```powershell
Get-ChildItem Env:FLASH_ATTENTION_ARCH,Env:CUTE_DSL_ARCH -ErrorAction SilentlyContinue
@'
import torch
print("cuda:", torch.version.cuda)
print("device capability:", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
'@ | python -
```

预期：普通在线运行通常不应存在遗留 override，代码会读取真实 device capability；CPU-only/cross compile 则应把“选择哪个对象族”和“编译给哪个目标”成对记录。override 能选择 Python 分支，但不能把一张 GPU 变成另一种架构。

## 症状三：首次调用慢，或多进程反复编译

### 可能原因

- 当前 `compile_key` 在进程内首次出现。
- 误以为默认 cache 会跨进程、跨重启复用。
- 已启用磁盘 cache，但 cache 目录不可共享、不可写，或源码/ABI 指纹变化造成自然失效。
- workload 切换了 dtype、head dim、GQA、mask、tile、arch、SplitKV 等静态特征，产生了新 key；不能笼统归因于“序列长度变化”。

### 源码入口

`compile_key` 由 dtype、head dim、mask/feature 存在性、tile、SplitKV、GQA、arch 等特征组成；这里没有直接把原始 `seqlen_q`、`seqlen_k` 塞进 key。

```python
# 来源：flash_attn/cute/interface.py L718-L750
compile_key = (
    dtype,
    head_dim,
    head_dim_v,
    qhead_per_kvhead,
    causal,
    score_mod_hash,
    mask_mod_hash,
    use_block_sparsity,
    block_sparse_broadcast_pattern,
    aux_tensor_metadata,
    aux_scalar_metadata,
    lse is None,
    cu_seqlens_q is None,
    cu_seqlens_k is None,
    seqused_q is None,
    seqused_k is None,
    page_table is not None,
    window_size_left is not None,
    window_size_right is not None,
    learnable_sink is not None,
    q_descale is not None,
    k_descale is not None,
    v_descale is not None,
    block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
    block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
    tile_m,
    tile_n,
    q_stage,
    num_threads,
    is_split_kv,
    pack_gqa,
    arch,
```

磁盘持久化默认关闭；开启后才按源码指纹建目录。源码或依赖版本变化导致指纹变化，是隔离陈旧对象，不是随机丢缓存。

```python
# 来源：flash_attn/cute/cache_utils.py L264-L281
def get_jit_cache(name: str | None = None) -> JITCache:
    """
    JIT cache factory.
    `name` is an optional identifier to create subdirectories to manage cache.

    When persistent caching is enabled, artifacts are namespaced under a
    source fingerprint directory so that code or dependency changes
    automatically invalidate stale entries.
    """
    if CUTE_DSL_CACHE_ENABLED:
        path = get_cache_path() / _compute_source_fingerprint()
        if name:
            path = path / name
        fa_log(1, f"Creating persistent JIT cache at {path}")
        return JITPersistentCache(path)
    else:
        fa_log(1, "Persistent cache disabled, using in-memory JIT cache")
        return JITCache()
```

### 操作与预期

1. 固定一组调用参数，分别记录第一次和第二次调用延迟。
2. 再启动新进程重复；默认配置下，新进程不能指望复用旧进程的内存 cache。
3. 若确实需要持久化，再显式设置：

```powershell
$env:FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED='1'
$env:FLASH_ATTENTION_CUTE_DSL_CACHE_DIR='D:\flash-attn-cute-cache'
```

预期：同一进程同 key 的第二次调用避开重新编译；磁盘 cache 开启且目录/指纹/key 均匹配时，新进程才可能加载对象文件。shape bucketing 是否有收益，必须根据实际 key 变化和 workload 测量，不能先写成生产硬规则。

## 症状四：FP8 forward/backward 被拒绝

### 可能原因

- 任一相关输入 `requires_grad=True`，触发当前 forward-only 边界。
- GPU 不是 FA4 当前允许 FP8 的 SM100。
- 非 FP8 输入却传了 `q_descale/k_descale/v_descale`。
- descale tensor 的 shape、dtype 或 device 不符合校验。

### 源码入口

FP8 输入只要参与梯度，就会在 kernel 构造前被拒绝；当前 FP8 输出默认是 BF16。

```python
# 来源：flash_attn/cute/interface.py L463-L467
is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
if is_fp8 and requires_grad:
    raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
```

descale 只属于 FP8 路径；传入时必须是 `(batch_size, num_head_kv)` 的 FP32 tensor，并且当前 FP8 只接受 SM100。

```python
# 来源：flash_attn/cute/interface.py L499-L510
if is_fp8:
    for t, name in ((q_descale, "q_descale"), (k_descale, "k_descale"), (v_descale, "v_descale")):
        if t is not None:
            _validate_tensor(t, name, (batch_size, num_head_kv), torch.float32, device)
else:
    assert q_descale is None and k_descale is None and v_descale is None, (
        "q_descale/k_descale/v_descale are only supported for FP8 inputs"
    )

dtype = torch2cute_dtype_map[q_dtype]
if is_fp8:
    assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
```

### 操作与预期

打印 `q/k/v/qv` 的 `dtype`、`requires_grad`、shape、device，再打印三个 descale tensor 的 shape/dtype/device 和真实 compute capability。预期合法的当前路径是 SM100 上的 FP8 forward；输出 dtype 为 BF16。若任务需要反向，应选择受支持的 dtype/实现，而不是绕过门禁后期待 kernel 自动补齐梯度语义。

## 症状五：paged KV、SplitKV 或 block sparsity 被拒绝

### 可能原因

- 把“FA4 总体有此功能”误读成“每个 arch 对象都实现了此功能”。
- 把 Python 构造阶段的能力断言误归因为 CUDA launch 故障。

### 源码入口

当前分支明确给出不对称能力矩阵：SM80 拒绝 paged KV 与 SplitKV，SM90 拒绝 SplitKV，SM120 拒绝 block sparsity、paged KV 与 SplitKV。

```python
# 来源：flash_attn/cute/interface.py L823-L845
if arch // 10 == 8:
    assert page_table is None, "paged KV not supported on SM 8.0"
    assert not is_split_kv, "SplitKV not supported on SM 8.0"
    fa_fwd = FlashAttentionForwardSm80(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=local,
        pack_gqa=pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=1,
        num_threads=num_threads,
        Q_in_regs=False,
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_aux_tensors=aux_tensors is not None,
    )
elif arch // 10 == 9:
    assert not is_split_kv, "SplitKV not supported on SM 9.0"
    fa_fwd = FlashAttentionForwardSm90(
```

SM120 的限制又不同，说明能力属于具体对象族，而不是统一的“FA4 开关”。

```python
# 来源：flash_attn/cute/interface.py L940-L965
elif arch // 10 == 12:
    # SM120 (Blackwell GeForce / DGX Spark): uses SM80 MMA with SM120 SMEM capacity
    assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
    assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
    assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
    fa_fwd = FlashAttentionForwardSm120(
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        is_causal=causal,
        is_local=local,
        pack_gqa=pack_gqa,
        tile_m=tile_m,
        tile_n=tile_n,
        num_stages=1,
        num_threads=num_threads,
        Q_in_regs=False,
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_aux_tensors=aux_tensors is not None,
    )
else:
    raise ValueError(
        f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    )
```

### 操作与预期

先记录 `arch`、`page_table is not None`、`is_split_kv`、`use_block_sparsity`，再定位对应对象构造分支。预期是：被显式断言拒绝的组合属于当前实现覆盖缺口；它既不证明 attention 数学不成立，也不保证换到另一个 arch 就一定支持全部组合。能力判断必须落到当前 baseline 的具体分支。

## 最短排障记录模板

每次至少保存以下信息，避免“同一个报错”其实来自不同实现：

```text
入口符号与模块路径：
flash-attn / flash-attn-3 / flash-attn-4 版本：
torch / CUDA / CuTeDSL 版本：
真实 GPU capability：
FLASH_ATTENTION_ARCH / CUTE_DSL_ARCH：
dtype、head_dim、GQA、mask、paged KV、SplitKV、requires_grad：
进程内首次/再次延迟：
磁盘 cache 开关、目录、是否跨进程：
最早失败层：import / arch / validation / compile / launch / numerical：
```

## 复盘

- FA2、FA3、FA4 是并存实现，不是一条自动替代链。
- `FLASH_ATTENTION_ARCH` 选择运行对象族，`CUTE_DSL_ARCH` 控制编译目标；二者不能替代真实硬件。
- 默认只有进程内 JIT cache；磁盘持久化必须显式开启，并受源码指纹、key、目录与进程条件约束。
- FP8、paged KV、SplitKV 等结论必须带上 arch、dtype、梯度和当前 baseline。
- 只有前四层都正确后，才进入 kernel launch、数值与性能分析。

继续沿调用生命周期阅读：[[FlashAttention-Hopper与CuTe-源码走读]]。要检查自己是否能独立定位，完成 [[FlashAttention-Hopper与CuTe-学习检查]]。
