---
title: "自定义扩展 · 排障指南"
type: troubleshooting
framework: slime
topic: "自定义扩展"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 自定义扩展 · 排障指南

## 你为什么要读

本篇按症状排障。Customization 的大多数问题不是算法错，而是 hook 选错层级、签名不对、返回对象形状不符合后续消费方。

## 1. path 一启动就报 import 错

先看 path 是否是 `package.module.attr`，不要写文件路径或带 `.py` 后缀。`load_function` 只做 `importlib.import_module` 加 `getattr`，不会做 fallback。源码入口：`slime/utils/misc.py` L37-L45。

如果同一个 path 在本地 Python 能 import，但 Slime 里不能 import，通常是运行时 `PYTHONPATH`、Ray runtime_env 或插件包安装位置不一致。

## 2. custom generate 没被调用

检查三个优先级：

1. `sample.generate_function_path`
2. `args.custom_generate_function_path`
3. 内置 `generate`

源码入口：`slime/rollout/sglang_rollout.py` L249-L260。

如果 sample 级 path 存在，它会覆盖全局 path。eval 场景还要确认你的函数是否声明了 `evaluation` 参数；只有签名里出现这个参数，Slime 才会传入 eval 标志。

## 3. custom generate 报 await 或返回值错误

`custom_generate` 必须是 async callable，因为调用点无条件 `await`。返回值应是 `Sample` 或 `list[Sample]`。

最小返回对象要保证：

- `tokens` 是 token id list。
- `response` 是字符串。
- `response_length` 与响应 token 数一致。
- `status` 表示完成或截断等状态。
- reward 若已经在 generate 内算好，后续 RM 逻辑要能识别。

契约测试入口：`tests/plugin_contracts/test_plugin_generate_contracts.py`。

## 4. fan-out 后 group reward 或 advantage 异常

检查兄弟样本是否共享同一个 `rollout_id`。fan-out 表示一个 prompt 拆成多个训练段，不是多个独立 prompt。

正确原则：

```python
rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
```

然后把这个 `rollout_id` 写回每个兄弟 sample。否则 GRPO group、train step 切分、reward normalization 和 metrics 都会错位。

## 5. custom RM 在 group 模式下返回错位

普通 RM 是 `args, sample` 到一个 float；group RM 是 `args, samples` 到一个 float list。batch 返回长度必须等于输入 `samples` 长度。源码入口：`slime/rollout/rm_hub/__init__.py` L97-L99，官方签名见 `docs/en/get_started/customization.md` L131-L136。

如果你同时支持两种模式，建议写两个函数或在函数里显式区分输入类型，不要让单样本逻辑在 group 模式下隐式复用。

## 6. filter 后样本数或 mask 对不上

`rollout_sample_filter` 不应该删除列表元素，而是原地设置 `Sample.remove_sample`。官方签名见 `docs/en/get_started/customization.md` L209-L211。

直接删除样本会改变 group 结构，后续 `n_samples_per_prompt`、rollout_id、reward 对齐和 DP schedule 都可能被破坏。若目标是“不参与 loss”，设置 `remove_sample` 才是正确边界。

## 7. custom loss path 配了但没生效

检查是否设置了 `--loss-type custom_loss`。很多 loss 相关 path 只是提供实现，实际分支仍由 loss type 或 reducer 配置决定。若只是改变 policy loss 归约，优先确认是否应该用 `--custom-pg-loss-reducer-function-path`，不要直接重写完整 loss。

pg loss reducer 的官方输入见 `docs/en/get_started/customization.md` L288-L294。

## 8. rollout_data_postprocess 之后训练崩在 shape

这个 hook 在 advantage/return 之后、日志和训练之前调用。源码入口：`slime/backends/megatron_utils/actor.py` L511-L512。

排查顺序：

1. 每个字段的 batch 维长度是否一致。
2. `tokens`、`response_lengths`、`loss_masks` 是否仍能互相解释。
3. 新增字段是否被后续 collate、logger 或 loss reducer 期待为 tensor。
4. 多 rank 情况下是否所有 rank 都做了同样的字段变换。

这个 hook 适合小范围修正，不适合重建整个 rollout 语义。

## 9. DataSource 续训或 partial rollout 后乱序

自定义 DataSource 必须支持 `get_samples`、`add_samples`、`save`、`load` 和 `__len__`。官方要求见 `docs/en/get_started/customization.md` L387-L401。

如果只实现单向读取，partial abort、buffer 回填和 checkpoint 恢复都会缺状态。返回形状也必须是 `list[list[Sample]]`，扁平 list 会让 group RM 和 `n_samples_per_prompt` 语义错位。

## 10. harness 里改了 model_label 但模型没变

`HarnessContext.model_label` 是外部 CLI 看到的模型名。Slime adapter 忽略它，实际 serving 的模型由后端 SGLang engine 加载的权重决定。源码入口：`slime/agent/harness/common.py` L37-L48。

因此，改 harness 配置只能影响 CLI 请求里的 model 字段，不能改变 rollout engine 的模型权重。真正的模型切换要看启动参数、权重同步或 SGLang engine 配置。

## 11. XML tool call 没被解析

优先确认是否已经被标准 `FunctionCallParser` 解析；XML fallback 只有在标准 parser 没产出 tool use 且有 tools schema 时才运行。源码入口：`slime/agent/parsing.py` L67-L85 和 L99-L110。

fallback 还要求 tool name 出现在 schema 中。模型输出了未知工具名时，解析器会保守跳过，避免把普通文本误识别成工具调用。

## 12. contract tests 应该怎么用

本专题相关的最低成本检查是：

```powershell
python -m pytest tests/plugin_contracts/test_plugin_rollout_contracts.py tests/plugin_contracts/test_plugin_generate_contracts.py tests/plugin_contracts/test_plugin_path_loading_contracts.py tests/plugin_contracts/test_plugin_runtime_hook_contracts.py -q
```

也可以直接给单个测试文件传你的 path，例如 rollout 函数或 custom generate path。契约测试只验证边界形状，不验证业务 reward 是否合理；业务正确性仍需要小规模 rollout 回放。
