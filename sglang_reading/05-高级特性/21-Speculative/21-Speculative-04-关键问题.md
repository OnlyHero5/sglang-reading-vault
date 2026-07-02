---
type: batch-doc
module: 21-Speculative
batch: "21"
doc_type: faq
title: "投机解码：关键问题"
tags:
 - sglang/batch/21
 - sglang/module/speculative
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# 投机解码：关键问题

## Q1：EAGLE 与 NGRAM 如何选型？

**Explain：** EAGLE 需要额外 draft 模型权重，适合有专用 draft checkpoint 的场景，接受率通常更高；NGRAM 零额外模型，依赖历史 n-gram 匹配，适合重复模式多的 workload（代码补全、模板文本）。

| 维度 | EAGLE | NGRAM |
|------|-------|-------|
| 额外模型 | 需要 draft model | 不需要 |
| Draft KV | 有 | 无 |
| 典型参数 | `--speculative-num-steps`, `--speculative-eagle-topk` | `--speculative-ngram-capacity` |
| PD 分离 | 支持 hidden transfer | 不涉及 draft hidden |

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L127-L130
    def carries_draft_hidden_states(self) -> bool:
        """Whether the disagg prefill->decode transfer carries draft hidden
        states (EAGLE-family only; STANDALONE's vanilla draft ignores them)."""
        return self.is_eagle()
```

**Comment：** PD 部署若选 NGRAM，prefill/decode 均只需标准 KV transfer，无需 draft hidden 通道。

---

## Q2：为何必须开启 overlap 或使用 V2 Worker？

**Explain：** Spec V1 Worker 路径已移除；插件若 `supports_overlap=False` 且用户开启 overlap，会在 `create_worker` 直接失败。

**Code（错误配置）：**

```python
# 来源：python/sglang/srt/speculative/spec_registry.py L96-L99
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
```

**Code（正确做法 — 使用内置 EAGLE V2）：**

```bash
# 启动示例（读者在部署时使用）
python -m sglang.launch_server \
 --model-path meta-llama/Llama-3-8B \
 --speculative-algorithm EAGLE \
 --speculative-draft-model-path <draft-path> \
 --speculative-num-steps 5
```

**Comment：** 内置算法均已迁移 V2；自定义插件应实现 overlap 兼容的 Worker。

---

## Q3：Verify 阶段 dtype 与 stream 注意事项

**Explain：** 在 CUDA Graph / plan stream 内对 `predict` tensor 做 dtype cast 会产生跨 stream 依赖，导致 MTP acceptance 竞态。

**Code：**

```python
# 来源：python/sglang/srt/speculative/base_spec_worker.py L115-L118
        # Do NOT cast predict dtype here. The caller (e.g., _draft_extend_for_decode)
        # may run this under a plan stream; casting inside the plan stream creates a
        # cross-stream dependency that can lead to data races and break MTP acceptance.
        # The caller should cast to int64 before entering the plan stream context.
```

**Comment：** 调用方在进入 plan stream **之前** 将 token id cast 为 int64；这是 EAGLE/MTP 调试时的常见隐蔽 bug。

---

## Q4：DFLASH 与 EAGLE 的差异

**Explain：** DFLASH 支持 `supports_target_verify_for_draft()`，draft 阶段可嵌 target verify；适合特定 draft-target 协同架构。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L118-L119
    def supports_target_verify_for_draft(self) -> bool:
        return self.is_dflash()
```

**Comment：** Scheduler / model_runner 在 DFLASH 分支会启用 target-in-draft 验证逻辑；普通 EAGLE 仅在独立 verify 阶段对比 probs。

---

## Q5：Multi-Layer EAGLE 何时启用？

**Explain：** 当 `enable_multi_layer_eagle` 为真且算法为 EAGLE 时，工厂返回 `MultiLayerEagleWorkerV2`，支持多层 draft 堆叠。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L218-L223
        if self.is_eagle() and server_args.enable_multi_layer_eagle:
            from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
                MultiLayerEagleWorkerV2,
            )

            return MultiLayerEagleWorkerV2
```

**Comment：** 需额外配置 multi-layer draft 权重与 `multi_layer_eagle_utils` 中的层间传递逻辑。

---

## Q6：如何注册自定义投机算法？

**Explain：** 使用装饰器注册 factory，factory 接收 `ServerArgs` 返回 Worker 类；可选 `spec_class` 覆盖谓词。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L59-L77
    @classmethod
    def register(
        cls,
        name: str,
        *,
        supports_overlap: bool = False,
        validate_server_args: Optional[ServerArgsValidator] = None,
        spec_class: Type[CustomSpecAlgo] = CustomSpecAlgo,
    ) -> Callable[[WorkerFactory], WorkerFactory]:
        """Decorator to register a plugin speculative algorithm. The factory
        takes ``server_args`` and returns the worker class. Pass a
        ``CustomSpecAlgo`` subclass via ``spec_class`` to override any
        ``is_*()`` / ``create_worker`` method.

        Example:
            @SpeculativeAlgorithm.register("MY_SPEC", supports_overlap=False)
            def _factory(server_args):
                return MySpecWorker
        """
```

**Comment：** 插件 Worker 应继承 `BaseSpecWorker` 并实现 `forward_batch_generation`；注册名不可与 EAGLE/NGRAM 等内置名冲突。

---

## Q7：SGLang EAGLE 与 vLLM Spec Decode 架构差异？

**Explain：** 两者都实现「draft 提议 + target 验证」，但集成层级与数据结构不同。SGLang 把投机作为 Scheduler 与 `TpModelWorker` 之间的 Spec Worker 层，用 `SpecInputType` 阶段模型驱动 Attention 后端；vLLM 则在 Engine 核心里挂载 proposer，验证与主模型 forward 在同一调度框架内展开。选型时需关注：是否已有 SGLang 的 PD 分离 / overlap / adaptive spec 需求，以及 draft 树 verify 的实现路径。

| 维度 | SGLang EAGLE (V2) | vLLM Spec Decode |
|------|-------------------|------------------|
| 入口 | `SpeculativeAlgorithm.create_worker` → `EAGLEWorkerV2` | Engine `SpeculativeConfig` + proposer 插件 |
| Draft/Target | 两个独立 `TpModelWorker` / ModelRunner | draft 模型或 ngram proposer + 主模型 |
| Verify 数据结构 | `EagleVerifyInput`（树 mask + retrieve_index） | batch 扩展 / proposer 专用 metadata |
| Attention 集成 | `batch.spec_info` → FlashInfer tree mask | 引擎内 speculative metadata 路径 |
| 接受判定 | Triton `reject_sampling` kernel | 引擎侧 rejection / acceptance 逻辑 |
| 运行时调参 | `AdaptiveController` 切换 `SpecRuntimeState` | 多为静态 `num_speculative_tokens` 配置 |
| PD 分离 | `build_disagg_draft_input` 传 draft hidden | 依部署模式，非 SGLang 原生 disagg 栈 |
| 插件扩展 | `@SpeculativeAlgorithm.register` | 自定义 proposer 注册 |

**Code（SGLang verify 输入锚点）：**

```python
# 来源：python/sglang/srt/speculative/eagle_info.py L17-L21
@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
```

**Comment：** SGLang 优势在统一 Scheduler 语义（与 NGRAM/MTP/DFLASH 共用 `SpecInput` 接口）及 PD 分离下的 hidden transfer；vLLM 优势在生态内与 V1 Engine 调度深度耦合。跨框架迁移 EAGLE 时，draft 权重可复用，但 verify 阶段的 mask 构造与 KV 写回逻辑需重写。
