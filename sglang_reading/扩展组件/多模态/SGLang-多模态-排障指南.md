---
title: "多模态 · 排障指南"
type: troubleshooting
framework: sglang
topic: "多模态"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# 多模态 · 排障指南

## 你为什么要读

多模态故障的报错点经常晚于根因。最有效的办法不是从 CUDA 栈向上猜，而是逐站检查四个不变量：媒体数量、prompt 顺序、placeholder span、feature/embedding 身份。

## 先做五分钟分层

| 现象 | 第一入口 | 先确认 |
|---|---|---|
| 启动时报 unsupported processor/model | `multimodal_processor.py`、`server_args.py` | 架构名、model backend、encoder disaggregation 支持表 |
| 下载/解码失败 | 模型专用 processor、媒体 loader | URL/Base64、格式、超时、像素/帧预算 |
| shape、grid、position mismatch | `base_processor.py`、`qwen_vl.py` | 占位符顺序、grid、offset、展开 token 数 |
| Scheduler 反序列化或显存异常 | `schedule_batch.py`、CUDA IPC utils | transport mode、pool、驻留策略、重建 device |
| 结果偶发错误但长度相同 | ViT CUDA Graph runner | 同 `S` 是否有不同分段 metadata |
| encoder-only 超时或 embedding 丢失 | `encode_server.py` | backend、URL 注册、DP 映射、`/send` 生命周期 |

## 症状 1：找不到 Processor 或 Transformers backend 不兼容

**可能原因**

- `hf_config.architectures` 没有匹配已注册模型类名；
- 外部 processor 包没有导入，或覆盖注册失败；
- `model_impl=transformers`，但 processor 没声明 `supports_transformers_backend`；
- encoder disaggregation 使用了不在当前支持表中的架构。

**源码入口**

- `managers/multimodal_processor.py::get_mm_processor`
- `managers/tokenizer_manager.py::init_tokenizer_and_processor`
- `server_args.py::_handle_encoder_disaggregation`

**操作**

```powershell
rg -n "class .*Processor|models =|supports_transformers_backend" sglang/python/sglang/srt/multimodal/processors
rg -n "architectures|PROCESSOR_MAPPING|get_mm_processor" sglang/python/sglang/srt/managers/multimodal_processor.py
```

**预期**：模型架构类名能唯一命中 processor；若走 Transformers 模型实现，该 processor 明确允许该 backend。

## 症状 2：媒体数量正确，但 token 或 embedding 仍错位

**可能原因**

- prompt 的 image/video/audio 顺序与调用方拼接媒体的假设不同；
- 把 `organize_results()` 的 IMAGE→VIDEO→AUDIO 分组误当成 prompt 必须分组；
- `grid_thw`、audio length 或 spatial merge 计算错；
- offset 指向的是展开前位置或被后续截断破坏。

**操作**

1. 打印 tokenize 后的特殊 token 位置。
2. 分别打印 image、video、audio item 数量与每项 grid/长度。
3. 逐个 span 计算 `end - start`，与该项预期视觉/音频 token 数比较。
4. 暂停 `allow_auto_truncate`，复现一次。

**预期**：prompt 扫描顺序保持不变；每种 modality 的独立游标都刚好消费完；所有 span 与 item 一一对应。

## 症状 3：开启自动截断后才出现 shape mismatch

**根因线索**

当前 `_validate_one_request()` 只执行：

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L958-L966
        if input_token_num >= self.context_len:
            if self.server_args.allow_auto_truncate:
                logger.warning(
                    f"The input ({input_token_num} tokens) is longer than the "
                    f"model's context length ({self.context_len} tokens). "
                    "Truncating the input."
                )
                del input_ids[_max_req_len:]
                input_token_num = len(input_ids)
```

它没有同步修改 `mm_inputs`。

**操作**

- 关闭 auto truncate；
- 在 API 层按文本 token + 视觉 token 预算提前拒绝；
- 若业务必须截断，只在 processor 前截文本，并确保不切断媒体标记结构。

**预期**：禁用后错位消失；超长请求得到明确的长度错误，而不是晚到的模型 shape 错误。

## 症状 4：开启 CUDA IPC 后显存反而上升

**可能原因**

- 每 tokenizer worker 的 pool 最少 128 MiB，总额可能超过配置预算；
- consumer 会再分配 target tensor 并 device copy，峰值同时包含共享 slice 与目标 tensor；
- TP consumer 尚未全部完成，producer chunk 不能回收；
- `keep_mm_feature_on_device=true` 让 fallback tensor 继续驻留 GPU。

**操作**

1. 记录 `tokenizer_worker_num` 与日志中的 per-worker pool MiB。
2. 对比关闭 IPC、关闭 `keep_mm_feature_on_device` 的峰值。
3. 检查 consumer 是否都递增共享完成计数。
4. 区分稳态占用与大请求瞬时峰值。

**预期**：实际池总额按 `worker_num × max(total/worker_num, 128MiB)` 估算；consumer copy 完成后 chunk 才逐步可回收。

## 症状 5：IPC pool 满后行为与预期不一致

不要假设“必回 CPU”。先核对 `keep_mm_feature_on_device`：

- false：普通路径会倾向把 feature 搬到 CPU；
- true：feature 可继续留在 CUDA tensor，即使没有装进 pool；
- Scheduler 侧若启用 hash buffer，还可能短暂上 GPU 后再回 CPU。

**预期**：日志、tensor device 与配置相符；同一请求的身份/hash 不因 transport fallback 改变。

## 症状 6：ViT CUDA Graph 只在某些多图组合下给错结果

**高概率原因**

graph key 只有总 `S`。单图 1024 token 与两图 512+512 token 可能命中同一个 key，但 `cu_seqlens`、window 分段不同。

**操作**

1. 构造总 `S` 相同、分段不同的两个请求。
2. 分别跑 eager 与 graph。
3. 比较 embedding 的 max/mean error 和最终 logits。
4. 若不一致，禁用 ViT graph 或扩展 graph key，使其包含足以区分布局的 metadata。

**预期**：同一请求 eager/graph 一致；不同布局不会静默复用错误 metadata。

## 症状 7：hash 命中异常或相同图片不能复用前缀

**可能原因**

- 外部 `mm_hashes` 数量与 item 数不一致；
- hash 字符串不是合法十六进制；
- 路由器与 SGLang 对媒体规范化方式不同；
- 把 pad value 当成 image token id 使用；
- `SGLANG_MM_SKIP_COMPUTE_HASH` 导致每次使用随机 UUID。

**操作**

- 记录规范化后 item 数、hash 与 pad value；
- 对同一内容重复请求，确认 hash 稳定；
- 对字节不同但视觉相同的输入，明确你需要“字节身份”还是“语义身份”。

**预期**：路由器计算的内容身份与 SGLang prefix key 一致；pad 位于普通词表之外，仅服务缓存寻址。

## 症状 8：encoder-only / language-only 请求挂住

**检查顺序**

1. `encoder_only` 与 `language_only` 是否误同时开启；
2. transfer backend 是 `zmq_to_tokenizer`、`zmq_to_scheduler` 还是 Mooncake；
3. 静态 `encoder_urls` 是否为空；若为空，bootstrap 注册是否成功；
4. DP 模式是否满足 encoder 的 `dp_size > 1, tp_size == 1` 约束；
5. `zmq_to_scheduler` 的 DP 模式是否显式提供 embedding port；
6. Mooncake `/encode` 是否已返回 metadata，但 `/send` 尚在等 ready event；
7. `req_id → dp_rank` 映射是否过期或 worker 已被 watchdog 判死。

**预期**：encode 与后续 send 落在同一 DP worker；超时会清理 stale mapping，而不是无限等待。

## 症状 9：健康检查通过，但业务请求仍失败

非 DP encoder 在 `embedding_to_send` 非空时直接认为“繁忙即存活”，不会再跑 dummy encode。DP worker 忙时也可能只报告健康，不占用 GPU 做探针。

**结论**：`/health_generate` 是服务存活信号，不是每次都覆盖 processor→ViT→传输→LLM 的端到端验证。

**操作**：另设低频合成业务探针，覆盖真实 modality、真实 transfer backend 和完整生成链路。

## 故障记录模板

```text
症状：
请求形态：text / image / video / audio 数量与顺序
模型与 baseline：
关键参数：transport、worker_num、keep_on_device、auto_truncate、ViT graph、EPD backend
最后一个正确边界：
第一个错误边界：
源码入口：
操作：
预期：
结果：
```
