---
type: batch-doc
module: 11-ModelRunner
batch: "11"
doc_type: walkthrough
title: "ModelRunner · 源码走读"
tags:
 - sglang/batch/11
 - sglang/module/model-runner
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# ModelRunner · 源码走读

## 走读顺序

1. `tp_worker.py` — Scheduler 调用的 Worker 入口
2. `forward_batch_info.py` — ForwardBatch 构造
3. `model_runner.py` — 初始化、load_model、forward
4. `runner/eager_runner.py` / `runner/decode_cuda_graph_runner.py` — 执行后端

---

## 1. TpModelWorker 初始化

### 1.1 构造 ModelRunner

**Explain：** `TpModelWorker.__init__` 解析 server_args、构建 ModelConfig，再创建 `ModelRunner`；多 layer EAGLE 会创建多个 runner。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L228-L269
# 提交版本：70df09b
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # Parse args
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = is_draft_worker
        self.is_multi_layer_eagle = is_multi_layer_eagle
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        # Draft worker: target's resolved MemoryPoolConfig (forwarded to ModelRunner).
        self.memory_pool_config = memory_pool_config

        # MTP model runners
        self.model_runner_list: List[ModelRunner] = []

        self._init_model_config()
        self._init_model_runner()
```

**Comment：**

- `is_draft_worker=True` 时加载 draft 模型路径，用于投机解码。
- `model_runner_list` 仅在 multi-layer EAGLE 时使用；默认只有一个 `_model_runner`。

### 1.2 分阶段初始化：内存池 → Attention → CUDA Graph

**Explain：** Worker 启动分三步：`alloc_memory_pool` 分配 KV、`init_attention_backends` 选 FlashInfer/Triton 等、`init_cuda_graphs` capture decode 图。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L316-L355
# 提交版本：70df09b
    def alloc_memory_pool(
        self,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        """Allocate KV cache pools only (no backends or cuda graphs)."""
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool
            self.model_runner.req_to_token_pool = req_to_token_pool
        if token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.alloc_memory_pool(memory_pool_config)
        for mr in self.model_runner_list[1:]:
            mr.req_to_token_pool = self.req_to_token_pool
            mr.token_to_kv_pool_allocator = self.token_to_kv_pool_allocator
            mr.alloc_memory_pool(memory_pool_config)

        # Validation
        assert self.model_runner.max_running_requests > 0, "max_running_request is zero"
        max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        assert max_req_len > 0, "Memory pool size is too small"

    def init_attention_backends(self):
        """Initialize attention backends for all model runners."""
        self.model_runner.init_attention_backends()
        for mr in self.model_runner_list[1:]:
            mr.init_attention_backends()

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        """Capture cuda graphs for all model runners."""
        self.model_runner.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )
        for mr in self.model_runner_list[1:]:
            mr.init_cuda_graphs(capture_decode_cuda_graph=capture_decode_cuda_graph)
```

**Comment：**

- 分阶段是为在显存估算后再定 pool 大小；Attention 后端依赖 pool layout。
- EAGLE 多 runner 逐步初始化，共享同一 req_to_token_pool。

---

## 2. ModelRunner 初始化

### 2.1 __init__ 并行与投机配置

**Explain：** ModelRunner 保存 TP/PP/EP/DP 各 rank 信息，并根据 speculative_algorithm 预读 draft 层数（EAGLE/DFlash）。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L343-L410
# 提交版本：70df09b
class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dcp_size = server_args.dcp_size
        self.dcp_rank = self.tp_rank % self.dcp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
```

**Comment：**

- `ModelRunnerKVCacheMixin` 提供 KV 池 sizing、alloc 逻辑（与RadixAttention/16 衔接）。
- `use_mla_backend` 影响 Attention 后端选择（DeepSeek MLA，Models 专用）。

### 2.2 load_model

**Explain：** 构建 `LoadConfig`、调用 ModelLoader 加载权重、初始化 LoRA/量化等；是ModelLoader ModelLoader 的调用方。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L1388-L1437
# 提交版本：70df09b
    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
```

**Comment：**

- `load_format` 决定 Default / GGUF / RemoteInstance 等加载器（ModelLoader）。
- `torch.set_num_threads(1)` 避免多线程与 NCCL 加载冲突。

---

## 3. forward 主路径

### 3.1 ModelRunner.forward

**Explain：** 入口递增 `forward_pass_id`，可选 msprobe/canary，再调 `_forward_raw`；结束后收集 MoE 指标与 expert capture。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L2954-L3002
# 提交版本：70df09b
    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = profile_range(_build_step_span_name(forward_batch))

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
```

**Comment：**

- `_forward_raw` 内根据 `forward_mode` 选择 EagerRunner 或 CudaGraphRunner。
- `reinit_attn_backend` 用于动态 batch size 变化时重建 attention metadata。

### 3.2 init_cuda_graphs

**Explain：** decode 阶段 batch size 离散化后 capture 多档 graph；prefill 通常走 eager。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L900-L920
# 提交版本：70df09b（节选）
    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        """Capture cuda graphs. Requires init_attention_backends() to have run.

        Spec draft runners pass capture_decode_cuda_graph=False
        because they capture their own decode-style graphs separately.
        """

        # The eager (no-cuda-graph) phase runner, built AFTER the attention
        # backend so its __init__ can warm up kernels (run-once) and allocate the
        # fixed-max static buffer — both before the cuda-graph runners, so that
        # buffer is canonical in the shared pool and the cg runners coalesce onto
        # it. Always built: it serves both the fully-disabled case (decode/prefill
        # runners point at it) and the eager fallback when a cg runner can't run a
        # batch.
        self.eager_runner = EagerRunner(self)

        # cuda-graph capture: prefill before decode, so both coalesce onto the
        # eager buffer allocated above. (init_prefill_cuda_graph routes prefill
        # to the eager runner when the prefill graph is disabled.)
        self.init_prefill_cuda_graph()

```

**Comment：**

- Graph capture 在权重加载、Attention 后端就绪之后进行。
- CPU 设备走 `cpu_graph_runner` 单独路径。

---

## 4. forward_batch_generation 完整链路

**Explain：** PP 末 rank 调 forward 并采样；非末 rank 只做 forward 传 proxy；overlap 模式下 logits 可能异步回传。

**Code：**

```python
# 来源：python/sglang/srt/managers/tp_worker.py L506-L560
# 提交版本：70df09b（节选）
        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
                routed_experts_output=out.routed_experts_output,
                indexer_topk_output=out.indexer_topk_output,
            )

            if is_verify:
                # Skip sampling; spec_v2 worker fires its own publish post-verify.
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and forward_batch.sampling_info.grammars is not None
            ):

                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not forward_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
                if (
                    forward_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, forward_batch
                    )

```

**Comment：**

- `sample` 在 ModelRunner 内调用 SamplingBatchInfo（Sampling 展开）。
- `can_run_cuda_graph` 反馈给 Scheduler 做 pipeline overlap 决策。

---

## 5. ForwardBatch.init_new（概念衔接）

**Explain：** 从 ScheduleBatch 提取 input_ids、req_pool_indices、seq_lens 等，填充 GPU tensor；设置 forward_mode。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/forward_batch_info.py L201-L250
# 提交版本：70df09b（节选）
def compute_local_num_token_non_padded(
    global_num_token_non_padded: torch.Tensor,
    num_tokens_per_dp: int,
) -> torch.Tensor:
    """Compute local non-padded token count for this attention-TP rank.

    Converts a global count (across all TP ranks) to a local count for this rank.
    The "global" scope is within the current DP rank; DP is handled via num_tokens_per_dp.
    """
    attn_tp_rank = get_parallel().attn_tp_rank
    attn_tp_size = get_parallel().attn_tp_size
    tokens_per_rank = num_tokens_per_dp // attn_tp_size

    return torch.clamp(
        global_num_token_non_padded - tokens_per_rank * attn_tp_rank,
        0,
        tokens_per_rank,
    )


@dataclass
class DSV4OutCacheLoc:
    """Per-forward-pass KV cache allocation for DeepSeek-V4 on NPU.

    Bundles slot indices for full/SWA pools, the two compressed-KV pools
    (c4/c128), and the two compressed-state pools (c4_state/c128_state).
    Populated by the NPU V4 allocator (DSV4NPUTokenToKVPoolAllocator) when
    the model is DeepSeek-V4 on NPU; left as ``None`` on ForwardBatch
    otherwise. CUDA's DSV4 path doesn't construct this bundle (state is
    derived via translate_kv_loc_to_compress_state_loc there).

    All fields are token-level slot ids in their respective pools (NOT page
    ids). Attention backends convert to page ids via ``// page_size`` when
    constructing PA_ND block tables.

    State fields default to ``None`` so the bundle is constructible from
    paths that allocate KV but not state (or vice versa); the NPU allocator
    fills all six on real alloc, CUDA paths leave state ones None and use
    the ring-hash translation instead.
    """

    out_full_loc: torch.Tensor
    out_swa_loc: torch.Tensor
    out_c4_loc: torch.Tensor
    out_c128_loc: torch.Tensor
    out_c4_state_loc: Optional[torch.Tensor] = None
    out_c128_state_loc: Optional[torch.Tensor] = None


@dataclass
```

**Comment：**

- `out_cache_loc` 指向本 step 写入 KV cache 的 slot（与 RadixAttention / allocator 衔接）。
- `prepare_mlp_sync_batch` 处理 DP attention 的 padding 与 sync。

---

## 6. EagerRunner 与 Graph Runner 选择

**Explain：** `_forward_raw` 内：若 mode 支持且 batch size 命中已 capture 的 graph，则 replay；否则 eager forward。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/runner/eager_runner.py L30-L55
# 提交版本：70df09b（节选）
    is_cp_v2_active,
    prepare_cp_forward,
)
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_eager_registry,
)
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    create_chunked_prefix_cache_kv_indices,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    set_tc_piecewise_forward_context,
)
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import ceil_align, require_mlp_sync

logger = logging.getLogger(__name__)
```

**Comment：**

- `forward_context` 设置全局 forward batch，供 RadixAttention 层读取 KV 索引。
- 模型 `forward` 签名统一为 `(input_ids, positions, forward_batch)`（Llama/Qwen 等均如此，Models 通用）。

---

## 7. input_buffers 与 overlap

**Explain：** overlap schedule 下输入 buffer 双缓冲，Scheduler 写下一 batch 时 GPU 仍跑上一 batch。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/input_buffers.py L25-L50
# 提交版本：70df09b（节选）
    an existing entry — so the sharing *structure* is independent of
    registration order and no already-captured buffer is ever repointed.

    This pool is process-wide and governs *every* ``share_buffers()`` caller —
    including graph runners not yet on the registry (the speculative draft /
    draft-extend / frozen-kv-mtp / multi-layer-eagle runners), which register
    identically-named ``input_ids`` / ``positions`` / ``out_cache_loc`` /
    ``mrope_positions``. Cross-runner sharing is safe because those buffers are
    filled immediately before each replay and the forwards that use them are
    sequential / mutually exclusive.
    """
    key: _PoolKey = (name, new_buffer.numel(), new_buffer.dtype, new_buffer.device)
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    return canonical.as_strided(new_buffer.size(), new_buffer.stride())


def share_input_buffers_in(obj) -> None:
    """Pool every tensor buffer on ``obj`` (dataclass / ``SimpleNamespace``)
    through the process-wide pool, in place. No-op on NPU; recurses into dict /
    dataclass buffer fields (``pp_proxy_tensors`` / ``ngram_embedding_info``)."""
    if is_npu():
        return

```

**Comment：**

- 预分配最大 batch 的 buffer，避免每 step cuda malloc。
- WAR barrier 由 `war_fastpath_runner` 的 read-done event 同步（见 BaseTpWorker）。

---

## 8. cuda_graph_config

**Explain：** 配置哪些 batch size 需要 capture、是否 enable padding bucket 等。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/cuda_graph_config.py L15-L40
# 提交版本：70df09b（节选）
cuda_graph_config, and the --cuda-graph-config JSON CLI parser.

Module-level imports are pure stdlib — no torch / sglang.srt deps — so
ServerArgs can import everything here without pulling in backend
classes. check_cuda_graph_backend lazy-imports get_global_server_args
inside the function body to preserve that invariant.
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class Phase:
    """The two phases of model forward."""

    DECODE = "decode"
    PREFILL = "prefill"
    ALL = (DECODE, PREFILL)


class Backend:
    """CUDA graph capture backends a phase can use."""

```

**Comment：**

- `capture_bs` 通常为 `[1,2,4,8,...]` 直到 max running requests。
- deduplication mixin 合并 shape 相同的 graph 减少显存（runner_backend 子目录）。
