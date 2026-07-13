---
title: "Hopper与CuTe · 学习检查"
type: exercise
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# Hopper与CuTe · 学习检查

## 必达能力

这份检查不验收术语背诵，而验收你能否从入口一路解释到 kernel 与 cache，并在版本、架构或功能组合变化时重新取证。完成后应满足：

- [ ] 能把 FA2、FA3、FA4 画成并存实现，而不是线性替代链。
- [ ] 能区分 README 的发行定位与当前 baseline 的 live 能力门禁。
- [ ] 能沿 FA3 的 schema→heuristic→static switch→SM8x/SM90 mainloop→scheduler→combine 讲清控制流。
- [ ] 能说明 SM90 中 TMA Q、TMA KV 都是条件路径，并解释何时转向 non-TMA/cp.async。
- [ ] 能沿 FA4 的 API→validation→program object→compile key→compile→callable→cache 讲清对象生命周期。
- [ ] 能区分进程内 JIT cache 与显式开启的磁盘持久化 cache。
- [ ] 能为 FP8、paged KV、SplitKV 写出带 baseline、arch、dtype、feature、grad 的能力判断。
- [ ] 能识别环境限制，不把 import 失败、缺 GPU 或 ABI 问题伪装成 kernel 结论。

## 第一关：不看笔记画两条生命周期

### FA3

至少画出以下对象与先后关系：

```text
Python/torch op
  → C++ schema 与 tensor 校验
  → pagedkv_tma / num_splits / pack_gqa heuristic
  → arch × split × paged-non-TMA × PackGQA × softcap static switch
  → SM8x 或 SM90 kernel type
  → tile scheduler
  → main kernel
  → num_splits > 1 时 partial combine
```

通过标准：能指出 `num_splits=0`、`pack_gqa=None` 会先被入口解析为最终策略；能说明 SplitKV 会物化 partial O/LSE，而不只是“多启动几个 block”。

### FA4

至少画出：

```text
flash_attn_func / flash_attn_varlen_func
  → FlashAttnFunc autograd boundary
  → _flash_attn_fwd validation 与策略归一化
  → compile_key
  → arch-specific program object
  → cache miss: cute.compile
  → compiled callable
  → 本次动态 tensor / scalar / stream 调用
  → autograd state / backward caches
```

通过标准：能明确 program object、compiled callable、输出 tensor 是三种不同对象；能解释 compile key 为什么没有直接包含原始 `seqlen_q/seqlen_k`，却包含 feature 存在性、tile、arch，甚至日志级别。

## 第二关：纠正五个常见错误

逐条判断并改写为严谨表述：

1. “FA3 只支持 H100/H800。”
2. “FA3 的 SM90 kernel 所有输入都使用 TMA。”
3. “FA4 支持 SM8x 到 SM12x，所以每个架构的 feature 完全相同。”
4. “FA4 只要序列长度改变，就一定重新 JIT。”
5. “FA4 默认会把编译对象写到磁盘，供所有进程复用。”

参考判据：

- 第 1 条必须区分 README beta 发布契约与当前 `flash_api.cpp` forward 门禁；后者已接受 Ampere 或更新架构，Ampere/Ada 限 FP16/BF16。
- 第 2 条必须写出 `Use_TMA_Q = !PackGQA`、`Use_TMA_KV = !PagedKVNonTMA`。
- 第 3 条必须引用具体 arch 分支的拒绝条件，不能只写“有些不支持”。
- 第 4 条必须先审计实际 `compile_key`；原始序列长度不是该 key 的直接字段。
- 第 5 条必须指出默认返回进程内 `JITCache`，只有显式环境变量开启后才使用 `JITPersistentCache`。

## 第三关：源码定位，不按文件顺序背诵

| 读者问题 | 应定位的对象 | 通过标准 |
|---|---|---|
| FA3 最初如何发布？ | `README.md` 的 beta release | 能复述发行硬件/dtype 范围，并声明它不是 live source 的全部边界 |
| FA3 当前接受哪些 GPU/dtype？ | `hopper/flash_api.cpp` forward validation | 找到 Ampere-or-newer 与 Ampere/Ada dtype 限制 |
| serving 状态从哪里进入？ | `TORCH_LIBRARY(flash_attn_3, m)` | 找到 page table、append KV、RoPE、descale、scheduler metadata、splits |
| 自动策略在哪里确定？ | `get_pagedkv_tma/get_num_splits/get_pack_gqa` 调用点 | 说清三者顺序与依赖 |
| 动态值如何变成静态实例？ | `run_mha_fwd` static switches | 写出五个分派轴 |
| SM90 为什么不是“全 TMA”？ | `mainloop_fwd_sm90_tma_gmma_ws.hpp` | 找到 `Use_TMA_Q/Use_TMA_KV` 与 cp.async 注释 |
| scheduler 接收什么？ | `flash_fwd_launch_template.h` | 找到 blocks、heads、batch、splits、semaphore、varlen arrays |
| SplitKV 何时 combine？ | `flash_api.cpp` 的 partial 分配与 `run_mha_fwd_combine` | 区分 `num_splits > 1` 与单 split |
| FA4 的静态边界是什么？ | `_flash_attn_fwd` 的 `compile_key` | 归类 dtype/shape-derived/feature/arch/scheduler 字段 |
| FA4 何时编译、何时调用？ | `cute.compile` 与 `compile_cache[compile_key](...)` | 说清 compile args 与 call args 的不同角色 |
| cache 为什么会失效？ | `cache_utils.py` | 找到开关、源码 fingerprint、key hash 与文件锁 |

## 第四关：可执行静态检查

在仓库根目录运行：

```powershell
rg -n 'FlashAttention only supports Ampere|Ampere/Ada cards' flash-attn/flash-attention/hopper/flash_api.cpp
rg -n 'get_pagedkv_tma|get_num_splits|get_pack_gqa' flash-attn/flash-attention/hopper/flash_api.cpp
rg -n 'ARCH_SWITCH|SPLIT_SWITCH|PAGEDKV_SWITCH|PACKGQA_SWITCH' flash-attn/flash-attention/hopper/flash_api.cpp
rg -n 'Use_TMA_Q|Use_TMA_KV|cp.async|GMMA::' flash-attn/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
rg -n 'out_accum|softmax_lse_accum|run_mha_fwd_combine' flash-attn/flash-attention/hopper/flash_api.cpp
rg -n 'compile_key =|get_fa_log_level|cute.compile|compile_cache\[compile_key\]' flash-attn/flash-attention/flash_attn/cute/interface.py
rg -n 'CUTE_DSL_CACHE_ENABLED|_compute_source_fingerprint|JITPersistentCache|FileLock' flash-attn/flash-attention/flash_attn/cute/cache_utils.py
```

预期：七组命中依次覆盖 live 门禁、FA3 heuristic、static dispatch、SM90 搬运/GMMA、SplitKV partial/combine、FA4 key/compile/call 和持久化 cache。若未来 baseline 某组不再命中，应重新阅读附近实现，而不是修改命令让旧答案“看起来通过”。

## 第五关：条件式运行验收

### 只验证 arch override 解析

这个实验不需要实际 GPU，但需要当前 Python 环境能导入 FA4/CuTeDSL 依赖。

```powershell
$env:FLASH_ATTENTION_ARCH='sm_90'
@'
try:
    from flash_attn.cute.interface import _get_device_arch
    print("arch:", _get_device_arch())
except Exception as exc:
    print("ENV_LIMIT:", type(exc).__name__, str(exc))
'@ | python -
```

预期：依赖完整时输出 `arch: 90`；否则输出 `ENV_LIMIT`。后者只能证明环境不足，不能证明 `_parse_arch_str` 或 GPU kernel 错误。实验结束后清理覆盖：

```powershell
Remove-Item Env:FLASH_ATTENTION_ARCH -ErrorAction SilentlyContinue
```

### 目标 GPU 环境的冷/热 cache 实验

准备固定的一组合法输入，记录：

1. 同一进程第一次调用延迟；
2. 同一进程同 key 第二次调用延迟；
3. 新进程默认配置下第一次调用延迟；
4. 开启 `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` 后的新进程延迟；
5. 修改一个真实 key 字段（如 dtype、head dim、arch/feature），再次调用。

预期：同进程同 key 能命中内存 cache；默认新进程不能继承旧进程字典；磁盘 cache 只有在开关、目录、fingerprint 与 key 都匹配时才可能复用。不要预设具体毫秒阈值，结果必须附 GPU、CUDA、依赖版本与 workload。

## 第六关：故障情景推理

### 情景 A：FA3 CUDA extension import 失败

回答：它是否一定自动 fallback？

通过标准：指出 HIP 分支在 compiled extension 缺失时可切到 Triton；CUDA 分支直接 import `flash_attn_3._C`，不能把 HIP 行为外推到 CUDA。

### 情景 B：SM90 paged KV 比预期慢

至少检查：

- `params.pagedkv_tma` 最终值；
- `page_table && !params.pagedkv_tma` 是否让 `PagedKVNonTMA=true`；
- `Use_TMA_KV` 是否因此为 false；
- PackGQA、SplitKV、varlen、head dims 和 tile 策略是否同时变化。

通过标准：不能只用“paged KV 有随机访问”作为结论，必须落到最终分派对象。

### 情景 C：FA4 FP8 backward 报错

至少记录：真实 arch、`q/k/v/qv` dtype、各 tensor 的 `requires_grad`、descale shape/dtype/device、输出 dtype。

通过标准：指出当前 FA4 FP8 是 SM100 forward-only 边界；不能通过删除 Python 断言来假设 backward kernel 已存在。

### 情景 D：FA4 新进程仍然冷编译

至少检查：持久化开关、cache 目录、源码 fingerprint、进程用户与目录权限、key 是否相同、对象文件与锁日志。

通过标准：能区分“默认只有内存 cache”“持久化 miss”“fingerprint 自然失效”“真实 key 改变”四类原因。

## 口述终验

用五分钟回答：

> FA3 与 FA4 都没有改变 FlashAttention 的 IO-aware 数学主线，那么它们分别把优化与复杂性搬到了哪里？

合格答案必须形成因果链：

- FA3：更宽的 serving/training 入口 → heuristic → static specialization → 条件化 TMA/GMMA 与 warp specialization → scheduler → 可选 SplitKV combine。
- FA4：Python/CuTeDSL program object → compile key → 按需编译 → compiled callable → 内存/持久化 cache → forward/backward 覆盖管理。
- 两者都必须按 baseline、arch、dtype、feature、grad 谈能力，不能用版本号代替证据。

## 收官标准

只有当你能独立完成两张生命周期图、五个纠错题、源码定位表、七组静态检查和四个故障情景，才算通过本专题。然后回到 [[FlashAttention-总结复盘]]，把 FA3/FA4 与 [[FlashAttention-Attention-IO-核心概念]]、[[FlashAttention-Online-Softmax-核心概念]] 串回同一条主线。
