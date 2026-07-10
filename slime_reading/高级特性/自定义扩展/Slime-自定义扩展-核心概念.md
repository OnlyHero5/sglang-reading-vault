---
title: "自定义扩展 · 核心概念"
type: concept
framework: slime
topic: "自定义扩展"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-10
---
# 自定义扩展 · 核心概念

## 你为什么要读

本页先给 Slime 自定义 hook 的边界地图。读完后，你应该能判断一个需求该放在 rollout 外循环、单样本生成、奖励、过滤、训练数据转换，还是 Megatron actor 内部，而不是一开始就替换最大粒度入口。

读 Customization 时先别背 17 个参数，先记住一个判断框架：这个 hook 替换的是外层编排、单样本生成、奖励、过滤、训练数据转换，还是训练 actor 内部行为。边界不同，函数看到的对象、允许的副作用和失败后果完全不同。

## 1. import-path 槽位

所有 `--*-path` 的共同入口是 `load_function`。它只负责把字符串解析成 Python 对象，不负责验证签名。

源码依据：`slime/utils/misc.py` L37-L45

```python
def load_function(path):
    """
    Load a function from a module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
```

这意味着 path typo、模块不可 import、属性不存在会直接抛错；但“函数是不是 async”“参数是不是对齐”“返回值是不是 `Sample`”要靠调用点和 contract tests 约束。

## 2. 外层 rollout 家族

外层 rollout hook 负责整段采样编排。它的能力最大，责任也最大。

| 参数 | 看见什么 | 必须维持什么 |
|------|----------|--------------|
| `--rollout-function-path` | `args, rollout_id, data_source, evaluation` | train/eval 返回结构、样本组形状、metrics |
| `--eval-function-path` | eval 版同签名输入 | eval dataset 的 `rewards/truncated/samples` 对齐 |
| `--data-source-path` | 构造时拿 `args` | `get_samples/add_samples/save/load/__len__` |

源码依据：`docs/en/get_started/customization.md` L58-L59 给出 `generate_rollout(args, rollout_id, data_source, evaluation=False)`；L387-L401 要求 DataSource 支持取样、回填、保存、加载和长度统计。

只有当默认 rollout 外循环无法表达你的调度方式时，才应替换 `--rollout-function-path`。如果只是每个 sample 怎么生成，应该用下一层。

## 3. 单样本 generate 家族

`custom_generate` 是 agentic workflow 最常用的入口。它替换的是“一个 `Sample` 如何生成响应”，而不是整个 rollout 调度器。

源码依据：`docs/en/get_started/customization.md` L79-L79

```python
async def custom_generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]
```

返回 `list[Sample]` 时是在做 fan-out：一个 prompt 产生多个训练片段。兄弟样本必须共享同一个 `rollout_id`，否则 group reward、advantage、train step 分组和日志都会把它们当成独立 rollout。

## 4. reward 与过滤家族

reward hook 有两种形态：

- 单样本：`async def custom_rm(args, sample: Sample) -> float`
- batch：`async def batched_custom_rm(args, samples: list[Sample]) -> list[float]`

源码依据：`docs/en/get_started/customization.md` L131-L136

过滤 hook 不都一样：

| 参数 | 作用对象 | 返回或副作用 |
|------|----------|--------------|
| `--dynamic-sampling-filter-path` | 一个 group 的 samples | `DynamicFilterOutput(keep, reason)` |
| `--buffer-filter-path` | buffer 与 num_samples | 返回被选中的 group 列表 |
| `--rollout-sample-filter-path` | 当前 groups | 原地设置 `Sample.remove_sample` |
| `--rollout-all-samples-process-path` | 全量 groups 与 data_source | 原地处理或写 metrics |

源码依据：`docs/en/get_started/customization.md` L168-L172 说明 `DynamicFilterOutput`；L209-L211 说明 sample filter 通过副作用标记样本。

## 5. 训练数据与训练侧家族

这组 hook 位于 `Sample` 转训练 batch 或 Megatron actor 内部：

| 参数 | 作用阶段 | 常见用途 |
|------|----------|----------|
| `--custom-reward-post-process-path` | advantage 前 | reward shaping、raw reward 保留 |
| `--custom-convert-samples-to-train-data-path` | `Sample` 到 train data | 自定义 `tokens/rewards/loss_masks` 等字段 |
| `--rollout-data-postprocess-path` | actor 内 advantage/return 后、训练前 | 根据 logprob、mask、metadata 再改 batch |
| `--custom-pg-loss-reducer-function-path` | policy loss reduce | Dr.GRPO、固定分母、per-token/per-sample 归约 |
| `--custom-loss-function-path` | loss 计算 | 新训练目标，需配合 `--loss-type custom_loss` |

源码依据：`docs/en/get_started/customization.md` L288-L294 给出 pg loss reducer 输入；`slime/backends/megatron_utils/actor.py` L511-L512 显示 `rollout_data_postprocess` 在训练前调用。

## 6. Megatron hooks

Megatron hook 不是用来改 rollout 数据形状的，它们是在训练栈内部插入动作。

源码依据：`docs/en/get_started/customization.md` L421-L443

```python
def custom_init(args) -> None

def custom_hook(args, model, store_prefix) -> None

def custom_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler) -> None
```

init hook 只拿 `args`；logprob 前 hook 拿模型和存储前缀；train step 前 hook 还能拿 optimizer 与 scheduler。它们靠近分布式训练状态，副作用必须保证所有 rank 一致。

## 7. Agent parsing 与 harness

Agent 相关代码分两层：

- `slime/agent/parsing.py` 把模型原始文本解析成 visible text、reasoning 和 tool uses。
- `slime/agent/harness/*` 在 sandbox 内安装并运行 Claude Code、Codex 这类外部 CLI。

源码依据：`slime/agent/parsing.py` L67-L85 使用 SGLang `FunctionCallParser`；L99-L110 提供 XML tool fallback。`slime/agent/harness/common.py` L107-L121 则把 CLI 运行轨迹写入 `.harness/trajectory.jsonl`。

harness 的 `model_label` 只是 CLI 看到的名字，真实模型仍由 Slime adapter 后面的 SGLang engine 决定。这个点排障时很重要：改 harness 配置不等于切换训练模型。

## 8. 运行验证

这页按 hook 家族组织，验证时要同时看文档参数、rollout manager、actor 前处理、loss 分支和 agent harness：

```powershell
rg -n 'rollout_function_path|custom_generate|custom_rm|DynamicFilterOutput|custom_convert_samples_to_train_data|rollout_data_postprocess|custom_pg_loss_reducer|custom_loss_function|FunctionCallParser|trajectory\.jsonl' slime/docs/en/get_started/customization.md slime/slime/ray/rollout.py slime/slime/backends/megatron_utils/actor.py slime/slime/backends/megatron_utils/loss.py slime/slime/agent/parsing.py slime/slime/agent/harness/common.py
```

预期输出应分布在 docs、rollout、actor、loss 和 agent 目录。若只命中文档而不命中执行代码，本页要标注该 hook 是否已退化为文档入口。
