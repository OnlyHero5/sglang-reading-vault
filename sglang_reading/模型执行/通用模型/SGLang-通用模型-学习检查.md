---
title: "通用模型 · 学习检查"
type: exercise
framework: sglang
topic: "通用模型"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-11
---
# 通用模型 · 学习检查

## 验收目标

读完本专题后，读者不应该只记住“有哪些模型文件”。合格状态是：遇到 architecture 不支持、PP stage 输出异常、Qwen3 attention shape 错、KV cache 重复写、权重名找不到时，能立刻判断该查类账、执行账还是权重账。

## 1. 三本账复述

- [ ] 能用 2 分钟解释类账：模型模块先被扫描并以 `EntryClass.__name__` 注册；随后 `HF config.architectures → get_model_architecture → ModelRegistry.resolve_model_cls → *ForCausalLM`。不要把运行时 resolve 说成再次 import/inspect 模型类。
- [ ] 能用 2 分钟解释执行账：`input_ids/positions/ForwardBatch → embedding 或 PPProxyTensors → decoder layers → RadixAttention → LogitsProcessor 或 PPProxyTensors`。
- [ ] 能用 2 分钟解释权重账：`(name, tensor) → load_weights → 前缀兼容/跳过规则/stacked_params_mapping → param.weight_loader`。
- [ ] 能明确说出边界：Registry 不管 QKV fused，model forward 不管 checkpoint 文件查找，`load_weights` 不决定 attention backend。

## 2. 画图验收

在纸上或白板上画三张图，每张图必须包含“输入对象、关键函数、输出对象”。

| 图 | 必须出现 | 失败信号 |
|----|----------|----------|
| 类账图 | `architectures`、`get_model_architecture`、`ModelRegistry`、`EntryClass.__name__`、fallback Transformers | 把文件名或 Python module 名当成 architecture key |
| 执行账图 | `ForwardBatch`、PP first/middle/last rank、`PPProxyTensors`、`RadixAttention`、`LogitsProcessor` | 认为每个 PP rank 都会产出 logits |
| 权重账图 | checkpoint name、`params_dict`、`stacked_params_mapping`、stage skip、`weight_loader` | 在 ModelLoader 层解释 `q_proj` 为什么不在参数表 |

## 3. 症状定位验收

| 症状 | 第一源码入口 | 要验证的判断 |
|------|--------------|--------------|
| native 模型没命中 | [[SGLang-通用模型-源码走读]] 第 1-3 节 | `architectures` 是否命中 `EntryClass.__name__`，是否被强制走 Transformers |
| PP 中间 rank 没 logits | [[SGLang-通用模型-数据流]] 第 3-4 节 | 当前 rank 是否是 last rank，输出是 hidden states 还是 `PPProxyTensors` |
| Qwen3 attention head shape 错 | [[SGLang-通用模型-源码走读]] 第 10-14 节 | QKV/O projection 使用 attention TP，O projection 不立即 reduce；Q/K 在 RoPE 前做 QK-Norm，layout 由 communicator 接回 MLP |
| Qwen3 decode KV cache 异常 | [[SGLang-通用模型-排障指南]] Q5 | 是否同时满足 ROCm、Aiter、MRoPE 与 decode 条件；若 fused 路径已写 KV，后续 `save_kv_cache` 是否关闭 |
| `Parameter <name> not found in params_dict` | [[SGLang-通用模型-排障指南]] Q6-Q7 | 前缀兼容、stage skip、tied embedding skip、QKV/gate-up mapping 是否命中 |

## 4. 源码证据验收

- [ ] 能指出 `get_model_architecture` 里 native、fallback、unsupported 三个分支各自的触发条件。
- [ ] 能指出 Registry 的“扫描注册”和“按 key resolve”是两个阶段，并解释模块 import warning 为什么会让 native key 消失。
- [ ] 能指出 `Qwen3Model` 为什么复用 `Qwen2Model` 骨架，以及 `decoder_layer_type` 在哪里替换。
- [ ] 能指出 `make_layers` 如何保留完整层号，同时用 `PPMissingLayer` 占住非本 stage 的位置。
- [ ] 能指出 Llama attention 的 QKV prepare 顺序和 Qwen3 attention 的额外 QK-Norm、attention TP 差异。
- [ ] 能指出 Llama、Qwen2、Qwen3 的 `load_weights` 在前缀处理、PP tied embedding、skip 规则、fused 参数 mapping 和缺失参数行为上的共同点与差异。

## 5. 运行验证验收

- [ ] 启动后能检查 `_resolved_model_arch` 与 `_resolved_model_impl`，确认当前模型是 native 还是 Transformers fallback。
- [ ] 遇到 PP 输出问题时，先记录当前 rank 的 `pp_group.is_first_rank`、`pp_group.is_last_rank`，再判断应该看到 embedding、hidden states、`PPProxyTensors` 还是 logits。
- [ ] 遇到 Qwen3 attention 问题时，先记录 `tp_size` 与 `attn_tp_size/attn_tp_rank`，再看 QKV/O projection、MLP 和 `LayerCommunicator` 分别使用哪条并行边界；只有满足 ROCm/Aiter/MRoPE 条件时才检查 fused mRoPE。
- [ ] 遇到 PP + `tie_word_embeddings` 时，能验证 last rank 的 `lm_head` 是共享模块还是独立参数，以及 embedding tensor 是否由 `load_weights` 补写过去。
- [ ] 遇到权重加载问题时，先打印原始 checkpoint name 与 remap 后的 param name，再判断是 skip、前缀兼容、fused shard 还是参数缺失。

## 后续边界

- DeepSeek MLA、MoE、DSA、多模态专用模型放到 [[SGLang-专用模型]]。
- `RadixAttention.forward`、prefix cache、backend metadata 放到 [[SGLang-RadixAttention]] 与 [[SGLang-Attention]]。
- 权重文件发现、下载、iterator、mmap、量化后处理放到 [[SGLang-ModelLoader]]。
