---
type: batch-doc
module: 21-Speculative
batch: "21"
doc_type: walkthrough
title: "投机解码 · 源码走读"
tags:
 - sglang/batch/21
 - sglang/module/speculative
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# 投机解码 · 源码走读

## 走读顺序

1. `spec_info.py` — 算法枚举与 Worker 工厂
2. `spec_registry.py` — 插件注册
3. `base_spec_worker.py` — Worker 基类与 EAGLE draft 准备
4. `eagle_worker_v2.py` — EAGLE 主 Worker
5. `ngram_worker.py` — NGRAM 无 draft KV 路径
6. `reject_sampling.py` — Triton verify kernel

---

## 1. spec_info.py — from_string 与 create_worker

### 1.1 字符串解析

**Explain：** CLI `--speculative-algorithm EAGLE` 最终调用 `from_string`；未知名称抛 ValueError，None 映射为 NONE。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L43-L57
    @classmethod
    def from_string(
        cls, name: Optional[str]
    ) -> Union[SpeculativeAlgorithm, CustomSpecAlgo]:
        if name is None:
            return cls.NONE
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            pass
        spec = _get_registered_spec(upper)
        if spec is not None:
            return spec
        raise ValueError(f"Unknown speculative algorithm name: {name}")
```

**Comment：** 返回类型可能是枚举或 `CustomSpecAlgo`，调用方统一用 `is_eagle()` 等接口，禁止 isinstance 分支。

### 1.2 has_draft_kv 语义

**Explain：** NGRAM 的候选树只存在于 verify mask，不写 draft KV；内存分配与 page 对齐逻辑因此不同。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L121-L125
    def has_draft_kv(self) -> bool:
        """Whether the draft phase writes KV chains. NGRAM does not (its tree
        lives only in the verify mask), so per-decode KV sizing needs no
        per-topk page rounding; see get_alloc_len_per_decode."""
        return not self.is_ngram()
```

**Comment：** Scheduler 在计算每步 KV 分配长度时调用；NGRAM 跳过 topk 相关的 page 舍入。

---

## 2. spec_registry.py — CustomSpecAlgo

**Explain：** 插件算法默认所有 `is_*()` 为 False，仅 `is_speculative()` 为 True；`create_worker` 检查 overlap 支持。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_registry.py L92-L111
    def handle_server_args(self, server_args: ServerArgs) -> None:
        pass

    def create_worker(self, server_args: ServerArgs) -> Type:
        if not server_args.disable_overlap_schedule and not self.supports_overlap:
            raise ValueError(
                f"Speculative algorithm {self.name} does not support overlap scheduling."
            )
        if not self.supports_overlap:
            # Reached only when overlap is disabled, so the algorithm really
            # does run synchronously on the V2 schema below.
            logger.warning(
                "Speculative algorithm %s is registered with "
                "supports_overlap=False, which is deprecated: the spec V1 "
                "worker path has been removed, and the algorithm now runs on "
                "the V2 scheduler schema with overlap disabled (synchronous). "
                "Migrate the plugin worker to support overlap scheduling.",
                self.name,
            )
        return self.factory(server_args)
```

**Comment：** overlap 开启但插件不支持时会直接报错；关闭 overlap 时仅 warning 后同步运行。

---

## 3. base_spec_worker.py — Draft Extend 准备

**Explain：** EAGLE draft extend 将 predict token 写入 batch，设置 `ForwardMode.EXTEND`；注意不在 plan stream 内 cast dtype，避免跨 stream 竞态。

**Code：**

```python
# 来源：python/sglang/srt/speculative/base_spec_worker.py L92-L120
    def prepare_for_draft_extend(
        self,
        draft_extend_input: EagleDraftExtendInput,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )
        from sglang.srt.utils.async_probe import maybe_detect_oob
        from sglang.srt.utils.common import is_npu

        bs = len(batch.seq_lens)
        extend_num_tokens = bs * num_draft_tokens
        # When seq_lens_cpu is absent, stay on GPU-only path -- no .tolist()/.cpu().
        gpu_only = batch.seq_lens_cpu is None

        batch.spec_info = draft_extend_input
        # Do NOT cast predict dtype here. The caller (e.g., _draft_extend_for_decode)
        # may run this under a plan stream; casting inside the plan stream creates a
        # cross-stream dependency that can lead to data races and break MTP acceptance.
        # The caller should cast to int64 before entering the plan stream context.
        batch.input_ids = predict
        maybe_detect_oob(
```

**Comment：** `batch.spec_info` 驱动 Attention 后端识别 EAGLE_DRAFT_EXTEND；后续构造 `ForwardBatch` 并可选走 CUDA Graph。

---

## 4. eagle_worker_v2.py — 类结构与依赖

**Explain：** `EAGLEWorkerV2` 继承 `BaseSpecWorker` 与 `EagleDraftWorkerBase`，组合 Target `TpModelWorker` 与独立 Draft ModelRunner；集成 adaptive spec 与多种 Attention 后端。

**Code：**

```python
# 来源：python/sglang/srt/speculative/eagle_worker_v2.py L54-L76
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    _eagle_prefill_tail_tokens,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
```

**Comment：** draft 与 verify 共用 `eagle_utils` 中的树构建与采样；`DraftBackendFactory` 按硬件选择 FlashInfer / Triton 等 draft attention。

---

## 5. ngram_worker.py — 语料库与初始化

**Explain：** NGRAM 不加载 draft 模型，而是用 C++ `NgramCorpus` 从历史 token 序列 BFS 匹配 draft 树；可加载 external corpus 文件。

**Code：**

```python
# 来源：python/sglang/srt/speculative/ngram_worker.py L37-L87
class NGRAMWorker(BaseSpecWorker):
    def alloc_memory_pool(self, **kwargs):
        # The target memory pool does not exist yet when __init__ runs.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            self._target_worker.get_memory_pool()
        )
        self.max_batch_size = self.model_runner.max_running_requests
        self._init_preallocated_tensors()

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.enable_overlap = not server_args.disable_overlap_schedule
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        # req_to_token_pool / token_to_kv_pool_allocator are set in
        # alloc_memory_pool(), after the target pools are allocated.
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self.adaptive_controller = None
        # rids of the last decode batch; used to erase corpus match state for
        # requests that left the batch (see forward_batch_generation).
        self._prev_decode_rids: set = set()

        self.ngram_corpus = NgramCorpus(
            min_bfs_breadth=server_args.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_ngram_max_bfs_breadth,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=server_args.speculative_ngram_max_trie_depth,
            draft_token_num=server_args.speculative_num_draft_tokens,
            external_sam_budget=server_args.speculative_ngram_external_sam_budget,
            external_corpus_max_tokens=server_args.speculative_ngram_external_corpus_max_tokens,
        )
```

**Comment：** `alloc_memory_pool` 延迟到 Target pool 就绪后调用；语料在 decode 过程中增量更新，请求离批时 erase 状态。

---

## 6. reject_sampling.py — Classic Verify Kernel

**Explain：** Triton kernel 实现标准 speculative sampling：逐步比较 target prob `p` 与 draft prob `q`，`coin * q < p` 则接受；全部拒绝后从 residual 分布 `max(p-q, 0)` 采样 bonus token。

**Code：**

```python
# 来源：python/sglang/srt/speculative/reject_sampling.py L48-L87
    # Verification Loop
    step = 1
    continue_verifying = 1

    while (step < NUM_SLOTS) and (continue_verifying == 1):
        draft_token = tl.load(cand_ptr_base + step * stride_cand_s)

        offset_prob = (
            (pid * stride_tp_b)
            + (cur_prob_row * stride_tp_s)
            + (draft_token * stride_tp_v)
        )
        offset_draft = (
            (pid * stride_dp_b)
            + (cur_prob_row * stride_dp_s)
            + (draft_token * stride_dp_v)
        )

        p = tl.load(TargetProbs + offset_prob)
        q = tl.load(DraftProbs + offset_draft)

        coin = tl.load(uni_ptr_base + (step - 1) * stride_uni_s)

        if coin * q < p:
            num_accept += 1
            cur_prob_row = step
            tl.store(Predicts + last_accepted_global_idx, draft_token)

            curr_global_idx = tl.load(idx_ptr_base + step * stride_idx_s)
            tl.store(
                AcceptIndex + pid * stride_idx_b + num_accept * stride_idx_s,
                curr_global_idx,
            )
            last_accepted_global_idx = curr_global_idx

            step += 1
        else:
            continue_verifying = 0

    tl.store(AcceptTokenNum + pid, num_accept)
```

**Comment：** 每个 batch 行（pid）独立验证；`AcceptTokenNum` 供 Scheduler 决定本步实际推进的 decode 长度。Final sampling 段（L89+）处理全接受或部分拒绝后的 bonus token。

---

## 7. duplicate_prefix_tail_to_draft_branches

**Explain：** EAGLE topk>1 时，各 draft 分支的首页需复制 prefix 尾页 KV，否则 expand 读错 block。

**Code：**

```python
# 来源：python/sglang/srt/speculative/base_spec_worker.py L22-L36
def duplicate_prefix_tail_to_draft_branches(
    token_to_kv_pool,
    rows: torch.Tensor,
    prefix_base: torch.Tensor,
    last_page: torch.Tensor,
    num_new_pages: torch.Tensor,
    topk: int,
    page_size: int,
) -> None:
    """Copy the prefix partial-tail page into each branch's first-page holes (page>1 + topk>1).

    The draft-decode expand pass reads each branch's own draft page by block id
    (cache_loc // page_size), so branch b>=1's hole slots [0, last_page) must hold the
    real prefix tail (branch 0's first page already is it). Mirrors V1 #7725.
    """
```

**Comment：** 镜像 V1 PR #7725 行为；仅复制 `[0, last_page)` 区间，不覆盖分支自有 draft slot。

---

## 8. handle_server_args 钩子

**Explain：** 各算法在 ServerArgs 解析后 mutate 专用字段（draft 模型路径、步数、topk 等）。

**Code：**

```python
# 来源：python/sglang/srt/speculative/spec_info.py L162-L181
    def handle_server_args(self, server_args: ServerArgs) -> None:
        """Hook for per-algorithm server args mutation.

        In-place updated.
        """
        from sglang.srt.arg_groups.speculative_hook import (
            _handle_dflash,
            _handle_eagle_family,
            _handle_frozen_kv_mtp,
            _handle_ngram,
        )

        if self.is_dflash():
            _handle_dflash(server_args)
        elif self.is_frozen_kv_mtp():
            _handle_frozen_kv_mtp(server_args)
        elif self.is_eagle() or self.is_standalone():
            _handle_eagle_family(server_args)
        elif self.is_ngram():
            _handle_ngram(server_args)
```

**Comment：** 在 `prepare_server_args` 流程末尾调用；确保 draft checkpoint 路径与 `speculative_num_steps` 等默认值正确。
