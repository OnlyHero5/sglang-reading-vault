---
type: module-moc
module: 25-LoRA
batch: "25"
doc_type: moc
title: "LoRA 多适配器服务"
tags:
 - sglang/batch/25
 - sglang/module/lora
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# LoRA 多适配器服务

> 阶段 VI · 扩展组件 | 状态：已完成 | Git：`70df09b` | **图谱更新点**

## 1. 本模块在全局架构中的位置

**Explain：** LoRA 模块在 ModelRunner 加载 base 模型后由 `LoRAManager` 将 Linear/MoE 层替换为 `BaseLayerWithLoRA` 包装；运行时按请求 `lora_id` 从 `LoRAMemoryPool` 取权重槽位，Triton CSGMV 批处理多 adapter。本模块与Models 通用-Models 的 base 层结构、RadixAttention-RadixAttention 的 `extra_key` LoRA namespace 紧密相关。

```
API (lora_path / lora_id)
 → TokenizerManager → Scheduler (Req.lora_id)
 → LoRAManager.load_adapter / eviction
 → ForwardBatch.lora_ids → Triton backend
 → BaseLayerWithLoRA.forward
```

---

## 2. 本模块目标

1. SGLang 如何实现 S-LoRA / Punica 风格的多 LoRA 并发？
2. `LoRAManager` 如何将 base 层替换为 `BaseLayerWithLoRA` 并管理内存池？
3. Triton / Torch backend 的 sgemv 批处理如何按 request 选 adapter？

## 文档导航

| 文件 | 内容 |
|------|------|
| [[25-LoRA-01-核心概念]] | LoRAAdapter、MemoryPool、Eviction |
| [[25-LoRA-02-源码走读]] | lora_manager、layers、backend |
| [[25-LoRA-03-数据流与交互]] | 动态加载、ForwardBatch、API |
| [[25-LoRA-04-关键问题]] | rank 限制、MoE LoRA、eviction |
| [[25-LoRA-05-checkpoint]] | 验收清单 |

## 源码范围

`srt/lora/` — lora_manager.py、layers.py、mem_pool.py、backend/、triton_ops/。

## 最关键的一段入口代码

**Explain：** `LoRAManager.__init__` 选择 backend、扫描 base 模型可挂 LoRA 的层，初始化 `LoRAMemoryPool` 与 eviction policy；`init_lora_modules` 将 Linear 替换为 LoRA 包装层。

**Code：**

```python
# 来源：python/sglang/srt/lora/lora_manager.py L57-L99
class LoRAManager:
    def __init__(
        self,
        base_model: torch.nn.Module,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        load_config: LoadConfig,
        dtype: torch.dtype,
        server_args: ServerArgs,
        lora_backend: str = "triton",
        tp_size: int = 1,
        tp_rank: int = 0,
        max_lora_rank: Optional[int] = None,
        target_modules: Optional[Iterable[str]] = None,
        lora_paths: Optional[List[LoRARef]] = None,
    ):
        self.base_model: torch.nn.Module = base_model
        if hasattr(base_hf_config, "get_text_config"):
            self.base_hf_config: AutoConfig = base_hf_config.get_text_config()
        else:
            self.base_hf_config: AutoConfig = base_hf_config
        self.max_loras_per_batch: int = max_loras_per_batch
        self.load_config: LoadConfig = load_config
        self.dtype: torch.dtype = dtype
        self.device: torch.device = next(self.base_model.parameters()).device
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.lora_added_tokens_size: Optional[int] = None
        self.enable_lora_overlap_loading: Optional[bool] = (
            server_args.enable_lora_overlap_loading
        )

        self.eviction_policy = server_args.lora_eviction_policy
        self._experts_shared_outer_override: Optional[bool] = (
            server_args.experts_shared_outer_loras
        )
        self.lora_use_virtual_experts: bool = server_args.lora_use_virtual_experts
        self.lora_strict_loading: bool = getattr(
            server_args, "lora_strict_loading", False
        )

        # LoRA backend for running sgemm kernels
        logger.info(f"Using {lora_backend} as backend of LoRA kernels.")
```

**Comment：**

- 参考论文 S-LoRA 与 Punica 的多租户批处理设计。
- `--max-loras-per-batch` 限制单 step 活跃 adapter 数；超出触发 eviction。
- 支持 MoE 层 `FusedMoEWithLoRA` 与 embedding / lm_head LoRA。

## 下一模块

→ [[26-sgl-kernel-00-MOC|sgl-kernel]]
