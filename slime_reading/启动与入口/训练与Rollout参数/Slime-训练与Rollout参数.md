---
title: "训练与Rollout参数"
type: map
framework: slime
topic: "训练与Rollout参数"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/map
  - source-reading
updated: 2026-07-10
---
# 训练与Rollout参数

这组文档解决一个具体问题：Ray 资源事实已经由 [[Slime-Ray参数]] 收口，剩下这些 train、rollout、data、algo、reward、plugin 参数如何真正挂进 RL 闭环。

旧读法容易把 `arguments.py` 当成参数清单。更稳的读法是把它看成“运行契约编译器”：CLI 字段先表达意图，`slime_validate_args` 补默认值和互斥约束，RolloutManager、SGLang rollout、Megatron actor、loss 函数再把这些字段变成可调用对象、样本账、训练 batch、权重同步和插件 hook。

## 读完能解决什么

| 你遇到的问题 | 本专题给你的抓手 |
|--------------|------------------|
| 自定义 rollout 函数没生效 | 看 `rollout_function_path` 在 RolloutManager 初始化时如何被 `load_function` 加载 |
| `eval_function_path` 没传却有值 | 看 validate 如何继承 `rollout_function_path` |
| `global_batch_size` 和 rollout 数对不上 | 用 `rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout` 推导 |
| custom generate、dynamic filter、sample filter 混淆 | 区分 per-sample 生成钩子、组级采样过滤、loss 参与过滤 |
| custom loss / advantage 不生效 | 看 Megatron loss 中的 `load_function` 消费点 |
| disk/delta 权重同步启动失败 | 看 validate 对 shared dir、transport、colocate、本地 checkpoint 的约束 |

## 主线地图

```mermaid
flowchart LR
    CLI["CLI / YAML<br/>训练语义意图"]
    VAL["slime_validate_args<br/>默认值与互斥约束"]
    RM["RolloutManager<br/>加载数据源和 rollout 函数"]
    SG["SGLang rollout<br/>生成、RM、filter"]
    DATA["train_data<br/>样本账转训练账"]
    ACT["Megatron actor<br/>logprob / train / hook"]
    LOSS["loss.py<br/>advantage 与 loss"]
    WS["WeightSync<br/>full / delta / disk / nccl"]

    CLI --> VAL
    VAL --> RM
    RM --> SG
    SG --> DATA
    DATA --> ACT
    ACT --> LOSS
    ACT --> WS
```

这条链路的核心不是“有哪些参数”，而是“哪个参数在哪个运行点改变行为”。

## 阅读顺序

| 文档 | 读者任务 |
|------|----------|
| [[Slime-训练与Rollout参数-核心概念]] | 建立样本账、函数路径、算法开关、后端透传四类心理模型 |
| [[Slime-训练与Rollout参数-源码走读]] | 沿一次 rollout 到 train 的主线读参数消费点 |
| [[Slime-训练与Rollout参数-数据流]] | 对照 path 参数、batch 参数、算法参数如何流向运行时 |
| [[Slime-训练与Rollout参数-排障指南]] | 按症状排查自定义函数、batch 推导、loss/advantage、权重同步 |
| [[Slime-训练与Rollout参数-学习检查]] | 用推导题验证自己能从配置推出运行行为 |

## 源码范围

| 源码入口 | 本专题关注点 |
|----------|--------------|
| `slime/utils/arguments.py` L107-L155 | 训练环境、Megatron to HF、full/delta weight sync |
| `slime/utils/arguments.py` L304-L340 | `hf_checkpoint`、`model_name`、`rollout_function_path` |
| `slime/utils/arguments.py` L441-L562 | dynamic filter、partial rollout、custom generate、buffer filter、update interval、rollout data postprocess |
| `slime/utils/arguments.py` L592-L725 | data source、prompt dataset、rollout batch、global batch 推导入口 |
| `slime/utils/arguments.py` L766-L775 与 L1908-L1919 | eval function 默认值和 global batch validate |
| `slime/utils/arguments.py` L902-L967 与 L1796-L1835 | loss、custom advantage、算法互斥与默认修正 |
| `slime/utils/arguments.py` L1316-L1458 | reward、sample conversion、buffer/sample filters、Megatron hooks |
| `slime/ray/rollout.py` L424-L451 与 L686-L723 | RolloutManager 加载 callable、reward postprocess、samples to train data |
| `slime/rollout/sglang_rollout.py` L249-L261 与 L394-L467 | custom generate、dynamic filter、sample filter、all-samples process |
| `slime/backends/megatron_utils/loss.py` L715-L764 与 L1264-L1274 | custom advantage 与 custom loss 的实际消费 |
| `slime/backends/megatron_utils/actor.py` L180-L184 与 L583-L620 | rollout data postprocess 与 weight updater |

刻意不展开：每个 loss 的数学推导、RM 具体打分逻辑、DataSource 续训游标、WeightSync 传输细节。这些在对应专题深入。

## 本专题不变量

| 不变量 | 为什么重要 |
|--------|------------|
| path 参数都必须是 `module.attr` | `load_function` 只做 `rpartition(".")` 后 import module 再取 attribute |
| `rollout_function_path` 替换整条 rollout 函数 | 它返回 `RolloutFnTrainOutput` 或 eval 输出，不只是单 sample generate |
| `custom_generate_function_path` 只替换默认 rollout 内部的 per-sample generate | 它不接管 dynamic filter、abort、sample filter 等外层循环 |
| `rollout_batch_size` 是 prompt group 数 | 训练样本数通常还要乘 `n_samples_per_prompt` |
| `num_steps_per_rollout` 会反推 `global_batch_size` | 显式给错会在 validate 阶段 assert |
| `loss_type=custom_loss` 必须配 `custom_loss_function_path` | 实际 loss 选择发生在 Megatron loss 函数里 |

## 运行验证入口

优先跑 contract tests，它们专门固定这些 path 参数的签名和调用点：

```powershell
python -m pytest tests/plugin_contracts -q
python -m pytest tests/test_megatron_argument_validation.py -q
```

如果环境缺少训练依赖，至少运行相关静态审计和可 import 的 contract tests；失败原因要区分“依赖收集失败”和“契约断言失败”。

下一篇先读 [[Slime-训练与Rollout参数-核心概念]]。
