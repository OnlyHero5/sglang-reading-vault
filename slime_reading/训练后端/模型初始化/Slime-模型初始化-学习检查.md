---
title: "模型初始化 · 学习检查"
type: exercise
framework: slime
topic: "模型初始化"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# 模型初始化 · 学习检查

## 读者能做什么

- [ ] 能画出 `args/checkpoint -> provider -> get_model -> optimizer/scheduler -> load_checkpoint -> ready model`。
- [ ] 能区分 custom、Bridge、legacy provider 三条路径。
- [ ] 能说明 actor LM head 与 critic value head 的差异。
- [ ] 能解释 freeze/only-train 为什么必须发生在 optimizer 创建前。
- [ ] 能复述 `setup_model_and_optimizer` 和 `initialize_model_and_optimizer` 的边界。
- [ ] 能说明 `forward_only` 与 `train_one_step` 的分工。
- [ ] 能解释为什么 forward-only 结果只在 pipeline last stage 聚合。
- [ ] 能说明 `pretrained_checkpoint` 通过 setup 断言为何仍不能替代最终 `args.load`。
- [ ] 能区分 Megatron 完整恢复与 Bridge HF 仅模型权重/iteration 0。

## 源码入口自测

- [ ] provider 选择：`slime/backends/megatron_utils/model_provider.py` L61-L240。
- [ ] freeze 包装：`slime/backends/megatron_utils/model_provider.py` L245-L286。
- [ ] optimizer/scheduler setup：`slime/backends/megatron_utils/model.py` L182-L318。
- [ ] critic head reinit：`slime/backends/megatron_utils/model.py` L125-L180。
- [ ] 初始化收口：`slime/backends/megatron_utils/model.py` L968-L1007。
- [ ] forward-only：`slime/backends/megatron_utils/model.py` L344-L506。
- [ ] actor 消费初始化结果：`slime/backends/megatron_utils/actor.py` L83-L168。

## 可执行验证

- [ ] 用 `rg -n 'get_model_provider_func|wrap_model_provider_with_freeze|freeze_model_params' slime/slime/backends/megatron_utils/model_provider.py` 定位 provider 选择与 freeze 时机。
- [ ] 用 `rg -n 'setup_model_and_optimizer|initialize_model_and_optimizer|forward_only' slime/slime/backends/megatron_utils/model.py` 区分装配、checkpoint 恢复和无梯度采集边界。
- [ ] 可用依赖环境下运行 `python -m pytest slime/tests/test_megatron_argument_validation.py`。
- [ ] 可用依赖环境下运行 `python -m pytest slime/tests/utils/test_megatron_server_arguments.py`。

预期：静态定位能串出 `provider → get_model → optimizer/scheduler → load_checkpoint`；轻量测试通过只证明参数约束，不能替代 Megatron、CUDA、checkpoint 和分布式环境下的完整初始化验证。

## 排障演练

- [ ] 构造 critic 从 actor checkpoint 恢复的路径，能指出 output layer reinit。
- [ ] 构造 custom provider critic 路径，能指出 `config.hidden_size` 契约。
- [ ] 构造 dynamic batch forward-only 路径，能指出结果顺序在哪里恢复。
- [ ] 构造 stateless Adam 路径，能指出两个必需条件。
- [ ] 构造 allowlist 零命中、非法 regex 与 blocklist 零命中三种 freeze 反例。
- [ ] 构造 forward-only hook 抛异常，能指出 eval mode 与进度条缺少 finally 恢复。
- [ ] 构造 dynamic batch index 缺项/重复，能说明 `zip(strict=False)` 不保证完整 permutation。

## 迁移结论

这组文档读懂后，再读 [[Slime-训练步骤]]：本专题解释模型对象、optimizer 和 scheduler 如何产生，训练步骤专题解释这些对象如何执行 forward、backward 与 optimizer step。
