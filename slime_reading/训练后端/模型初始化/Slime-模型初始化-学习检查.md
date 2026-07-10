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
updated: 2026-07-10
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

## 源码入口自测

- [ ] provider 选择：`slime/backends/megatron_utils/model_provider.py` L61-L240。
- [ ] freeze 包装：`slime/backends/megatron_utils/model_provider.py` L245-L286。
- [ ] optimizer/scheduler setup：`slime/backends/megatron_utils/model.py` L182-L318。
- [ ] critic head reinit：`slime/backends/megatron_utils/model.py` L125-L180。
- [ ] 初始化收口：`slime/backends/megatron_utils/model.py` L968-L1007。
- [ ] forward-only：`slime/backends/megatron_utils/model.py` L344-L506。
- [ ] actor 消费初始化结果：`slime/backends/megatron_utils/actor.py` L83-L168。

## 可执行验证

- [ ] 运行 `node maintenance/audit_source_evidence.mjs --note slime_reading/训练后端/模型初始化/Slime-模型初始化-源码走读.md`，确认源码引用可追踪。
- [ ] 运行 `node maintenance/audit_wikilinks.mjs`，确认双链无断链。
- [ ] 可用依赖环境下运行 `python -m pytest slime/tests/test_megatron_argument_validation.py`。
- [ ] 可用依赖环境下运行 `python -m pytest slime/tests/utils/test_megatron_server_arguments.py`。

## 排障演练

- [ ] 构造 critic 从 actor checkpoint 恢复的路径，能指出 output layer reinit。
- [ ] 构造 custom provider critic 路径，能指出 `config.hidden_size` 契约。
- [ ] 构造 dynamic batch forward-only 路径，能指出结果顺序在哪里恢复。
- [ ] 构造 stateless Adam 路径，能指出两个必需条件。

## 迁移结论

这组文档读懂后，再读 [[Slime-训练步骤]] 会更顺：19 讨论的是已经初始化好的模型如何执行 forward/backward/step；18 讨论的是这个模型对象、optimizer 和 scheduler 是如何来的。
