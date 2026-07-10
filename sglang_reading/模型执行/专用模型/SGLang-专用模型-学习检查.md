---
title: "专用模型 · 学习检查"
type: exercise
framework: sglang
topic: "专用模型"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 专用模型 · 学习检查

## 验收目标

读完本专题后，合格状态不是能背出 `deepseek_v2.py` 的类名，而是能把 DeepSeek 专用模型拆成四本账：类账、Attention 账、Expert 账、权重账。遇到 MLA/MHA 选路、shared expert fusion、DSA top-k、CP shape、权重 remap 问题时，能先定位到正确接缝。

## 1. 主线复述

- [ ] 能解释 `DeepseekV2ForCausalLM`、`DeepseekV3ForCausalLM`、`DeepseekV32ForCausalLM` 为什么共用实现但按 `EntryClass` 分别注册。
- [ ] 能解释 `ForCausalLM.forward` 只补 CP metadata 和 attention TP context，仍保持 PP last rank 才产出 logits。
- [ ] 能解释 `DeepseekV2Model.forward` 如何在 PP proxy 中额外传递 DSA `topk_indices`。
- [ ] 能解释 `DeepseekV2DecoderLayer` 如何把 `LayerCommunicator`、`DeepseekV2AttentionMLA`、`DeepseekV2MoE/MLP` 串起来。
- [ ] 能解释 `DeepseekV2WeightLoaderMixin` 为什么要做 stage skip、shared expert remap、fused qkv_a 和 post-load `kv_b_proj` 拆分。

## 2. 画图验收

| 图 | 必须出现 | 常见错误 |
|----|----------|----------|
| Attention 选路图 | `ForwardBatch.forward_mode`、prefill/decode backend、`AttentionBackendRegistry`、`AttnForwardMethod` | 直接把 backend 名等同于 MLA/MHA |
| Expert 形态图 | `n_routed_experts`、`n_shared_experts`、`num_fused_shared_experts`、DeepEP `moe_ep_size` | 看到 expert 数变大就判断 checkpoint 错 |
| PP/DSA 状态图 | `hidden_states`、`residual`、`topk_indices`、`PPProxyTensors` | 忽略 DSA skip-topk 跨 stage 状态 |
| 权重装载图 | 原始 checkpoint name、stage skip、shared remap、expert mapping、fused qkv_a、`w_kc/w_vc` | 只按同名参数查 `params_dict` |

## 3. 症状定位验收

| 症状 | 第一源码入口 | 要验证的判断 |
|------|--------------|--------------|
| prefill 走 MHA chunked KV | [[SGLang-专用模型-源码走读]] 第 5 节 | prefix 长度和 backend handler 是否显式选择 MHA |
| shared experts fusion 关闭 | [[SGLang-专用模型-排障指南]] Q4 | disable reason 是配置、硬件、量化、SBO/TBO 还是 DeepEP 默认 |
| `tp_size > n_routed_experts` | [[SGLang-专用模型-排障指南]] Q3 | 是否把 TP 当作 expert parallel 使用 |
| PP 下缺 `topk_indices` | [[SGLang-专用模型-数据流]] 第 2 节 | 上一 stage 是否应该把 DSA top-k index 放入 proxy |
| 权重名找不到 | [[SGLang-专用模型-排障指南]] Q7 | 是否 stage 外跳过、shared expert remap、fused qkv_a 延迟写入 |
| CP shape 异常 | [[SGLang-专用模型-数据流]] 第 3 节 | metadata、split 和 gather 三步是否一致 |

## 4. 源码证据验收

- [ ] 能指出 `EntryClass` 中三个类名如何接入 Registry。
- [ ] 能指出 `dispatch_attn_forward_method` 如何从 `ForwardBatch` 选择 prefill/decode backend。
- [ ] 能指出 backend handler 为什么会把某些 prefill 送到 `MHA_CHUNKED_KV` 或 `MHA_ONE_SHOT`。
- [ ] 能指出 `_is_layer_sparse` 如何决定 dense MLP 与 MoE。
- [ ] 能指出 DeepEP fusion 下 `num_experts_for_moe` 和 `top_k_for_moe` 如何变化。
- [ ] 能指出 `DeepseekV2Model.forward` 何时在 `PPProxyTensors` 中加入 `topk_indices`。
- [ ] 能指出 `DeepseekV2WeightLoaderMixin` 的 shared expert remap 与 fused qkv_a 缓存逻辑。
- [ ] 能指出 `post_load_weights` 为什么要生成 `w_kc` 和 `w_vc`。

## 5. 运行验证验收

- [ ] 在 attention 入口记录 `current_attention_backend` 和 `AttnForwardMethod`，能解释实际路径。
- [ ] 在 MoE init 记录 `num_experts_for_moe`、`top_k_for_moe`、`num_fused_shared_experts`，能解释 expert slot 数。
- [ ] 在 PP 边界记录 `PPProxyTensors.tensors.keys()`，能确认 DSA `topk_indices` 是否跨 stage 传递。
- [ ] 在 CP 路径记录 `forward_batch.attn_cp_metadata`、split 后 token 数、gather 后 hidden shape。
- [ ] 在 weight loader 记录原始 name、remap 后 name、是否 stage skip、是否进入 `cached_a_proj`。

## 后续边界

- expert dispatch、combine、EPLB、DeepEP all-to-all 细节放到 [[SGLang-MoE]]。
- RadixAttention、prefix cache、backend metadata 放到 [[SGLang-RadixAttention]] 与 [[SGLang-Attention]]。
- ModelLoader 的文件发现、iterator、mmap、量化后处理放到 [[SGLang-ModelLoader]]。
- NextN speculative 的完整草稿模型链路放到 [[SGLang-Speculative]]。
