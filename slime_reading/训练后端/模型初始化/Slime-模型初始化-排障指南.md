---
title: "模型初始化 · 排障指南"
type: troubleshooting
framework: slime
topic: "模型初始化"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-13
---
# 模型初始化 · 排障指南

本篇按症状排障。先判断问题属于 provider 选择、checkpoint load、optimizer/scheduler，还是 forward-only 采集。

## 症状速查

| 症状 | 最可能原因 | 源码入口 | 验证方法 |
|------|------------|----------|----------|
| 初始化 assert load/pretrained | `args.load` 与 `args.pretrained_checkpoint` 都为空 | `model.py` L291-L294 | 检查 arguments bridge/legacy load 规则 |
| critic value head shape mismatch | 从 actor checkpoint 恢复 critic | `model.py` L125-L180 | 看 warning，确认是否 reinit output layer |
| custom provider critic 报 hidden_size | provider 没暴露 `model.config.hidden_size` | `model_provider.py` L66-L83 | 检查 custom provider 返回对象 |
| 只训练部分参数无效 | freeze pattern 在 optimizer 创建后才改或 pattern 不匹配 | `model_provider.py` L245-L286 | 打印 `requires_grad` 参数名 |
| Stateless Adam 报错 | 非 Adam 或未设置 `--no-save-optim` | `model.py` L304-L316 | 修正 optimizer 配置 |
| forward-only 无结果 | 非 last PP stage 查看输出 | `model.py` L487-L506 | 在 pipeline last stage 打印 keys |
| logprob 顺序和 reward 对不上 | dynamic batch 没还原顺序 | `model.py` L497-L505 | 检查 `micro_batch_indices` |
| LR decay 终点不准 | `train_iters` 是估算 | `model.py` L182-L235 | 显式设置 `--lr-decay-iters` |
| 只给 pretrained checkpoint 仍在 loader 失败 | initialize 最终仍硬读 `args.load` | `model.py` L990-L1000、`checkpoint.py` L97-L120 | 打印最终 `args.load` 并验证非空目录 |
| forward-only 异常后 dropout 状态不对 | eval→train 无 finally | `model.py` L447-L506 | 异常后检查每个 chunk `.training` |

## Bridge 和 legacy load 怎么区分？

Bridge 模式如果 `args.load` 指向 Megatron checkpoint，就按 checkpoint 恢复；否则从 HF/ref/hf_checkpoint 路径设置 load，并把 `start_rollout_id` 置 0。

legacy/raw 模式如果没有有效 Megatron tracker，会设置 finetune 语义，跳过 optimizer/RNG load，并把 `load` 指向 `ref_load`。

源码入口：来源：slime/utils/arguments.py L1763-L1785

排查时先看最终解析后的 `args.load`，不要只看命令行原始值。

## critic output layer 为什么会被重置？

actor checkpoint 的 LM head shape 通常是 `[vocab, hidden]`，critic value head 是 `[1, hidden]`。如果 checkpoint metadata 中缺少或 shape 不匹配，Slime 会 warning 并在 load 后重新初始化 critic output layer。

源码入口：

- 来源：slime/backends/megatron_utils/model.py L125-L180
- 来源：slime/backends/megatron_utils/model.py L990-L1004

如果使用 fp16/bf16，重置后还会 `optimizer.reload_model_params()`，刷新 optimizer 的 model/main-param 内容。

但自动重置只检查带 `.metadata` 的 checkpoint，并且发生在 load 成功之后。旧格式/HF 目录不会进入这项 metadata 判断；严格 load 若先因 shape mismatch 失败，也无法靠后置 reinit 修复。排障时先区分“检测到了需要重置”和“loader 已经容忍并完成加载”。

## `only_train_params_name_list` 和 `freeze_params_name_list` 能同时用吗？

不能。一个是 allowlist，一个是 blocklist，同时设置会让参数训练语义不清楚，参数校验直接报错。

源码入口：来源：slime/utils/arguments.py L1977-L1978

如果只想训练 LoRA 或少量层，用 allowlist；如果只是冻结 embedding 或特定层，用 blocklist。

再加两项验证：先编译每个 regex；构模后统计 `requires_grad=True` 的数量。allowlist 零命中会冻结全部参数，blocklist 零命中却不会报错。

## `forward_only` 为什么只在 last PP stage 有输出？

Megatron pipeline 只有 last stage 拿到最终 logits/value head 输出。`forward_only` 在 last stage 汇总 `forward_data_store`，其他 stage 返回空 dict。

源码入口：来源：slime/backends/megatron_utils/model.py L487-L506

排查 ref/teacher/old actor logprob 时，要在 last PP rank 查看 `rollout_data`。

## Stateless Adam 的边界是什么？

Stateless Adam 只替换 Adam class，并禁止保存 Adam moment states；因此必须：

- `optimizer == "adam"`
- `--no-save-optim`

源码入口：来源：slime/backends/megatron_utils/model.py L304-L316

它不是通用 optimizer 兼容层，也不改变模型 provider 或 checkpoint 结构。

## 自定义 provider 要遵守什么契约？

至少要支持 Megatron 调用协议：

- `pre_process`
- `post_process`
- 可选 `vp_stage`

当前 Slime custom wrapper 不透传 `config`、`pg_collection`；需要这些新版 provider 参数时不能仅在自定义函数签名中声明，必须调整 wrapper 并做 PP/VPP 集成测试。

## 为什么 pretrained checkpoint 非空，仍报 `args.load`？

`setup_model_and_optimizer` 的前置断言接受 `load` 或 `pretrained_checkpoint` 任一非空，但 `initialize_model_and_optimizer` 随后总会调用 Slime `load_checkpoint`。该函数直接对 `args.load` 做 `Path(...)`、存在性与非空目录检查。因此当前完整链路的权威输入是解析后的 `args.load`。

操作：打印参数校验后的 `args.load`、目录类型和 tracker；Bridge HF 目录应存在且非空，Megatron 目录应有 tracker 或直接是 `iter_XXXXXXX`。预期不能用一个仅存在于 `pretrained_checkpoint` 的路径绕过该检查。

如果用于 critic，还要让返回模型暴露：

- `model.config.hidden_size`
- 可替换的 `model.output_layer`

源码入口：来源：slime/backends/megatron_utils/model_provider.py L66-L83

## debug rollout-only 会初始化模型吗？

不会进入训练模型初始化路径。`debug_rollout_only` 在 actor 初始化外层短路，训练后端不需要构建 model/optimizer。

这个边界属于 [[Slime-Megatron-Actor初始化]]；本专题只覆盖实际进入 `initialize_model_and_optimizer` 的路径。

## 改初始化路径前跑什么？

轻量参数测试：

```powershell
python -m pytest slime/tests/test_megatron_argument_validation.py
python -m pytest slime/tests/utils/test_megatron_server_arguments.py
```

文档检查：

```powershell
node maintenance/audit_source_evidence.mjs
node maintenance/audit_wikilinks.mjs
```

完整模型初始化需要 Megatron/CUDA/checkpoint 环境；本地 Windows 缺依赖时不要把单测收集失败误判成初始化逻辑失败。
