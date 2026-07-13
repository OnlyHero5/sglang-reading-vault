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
updated: 2026-07-13
---
# 自定义扩展 · 排障指南

## 你为什么要读

本篇按症状排障。Customization 的大多数问题不是算法错，而是 hook 选错层级、签名不对、返回对象形状不符合后续消费方。

## 1. path 一启动就报 import 错

先看 path 是否是 `package.module.attr`，不要写文件路径或带 `.py` 后缀。`load_function` 只做 `importlib.import_module` 加 `getattr`，不会做 fallback，也不会检查取到的对象是否 callable。源码入口：`slime/utils/misc.py` L37-L45。

如果同一个 path 在本地 Python 能 import，但 Slime 里不能 import，通常是运行时 `PYTHONPATH`、Ray runtime_env 或插件包安装位置不一致。还要检查模块顶层是否依赖只存在于 driver 的环境变量、GPU 库或当前目录；插件可能在不同 Ray actor 中分别 import，顶层副作用必须可重复。

## 2. custom generate 没被调用

检查三个优先级：

1. `sample.generate_function_path`
2. `args.custom_generate_function_path`
3. 内置 `generate`

源码入口：`slime/rollout/sglang_rollout.py` L249-L260。

如果 sample 级 path 存在，它会覆盖全局 path。eval 场景还要确认你的函数是否显式声明了精确参数名 `evaluation`；只有 `inspect.signature(fn).parameters` 中出现它，Slime 才会传入 eval 标志。仅写 `**kwargs` 不够。该 path 每次 sample 调用都会重新解析，不能假设 Slime 在启动期缓存了 callable。

## 3. custom generate 报 await 或返回值错误

`custom_generate` 必须是 async callable，因为调用点无条件 `await`。源码不会在返回后立刻做完整类型检查；返回值应是 `Sample` 或 `list[Sample]`，错误形状往往要到 RM、filter、日志或训练转换时才暴露。

最小返回对象要保证：

- `tokens` 是 token id list。
- `response` 是字符串。
- `response_length` 与响应 token 数一致。
- `status` 表示完成或截断等状态。
- 非 group RM 下，若返回完成态/截断态样本，reward 必须已有值，或让后续 RM 能补齐。

契约测试入口：`tests/plugin_contracts/test_plugin_generate_contracts.py`。

## 4. fan-out 后 group reward、abort 或 filter 异常

先区分两类故障。若训练归约、step 数或 advantage 错位，检查兄弟样本是否显式共享同一个非空 `rollout_id`；`RolloutManager` 会在拍平前验证深层 sibling。若报错是 `'list' object has no attribute reward/response/remove_sample`，则是默认路径把 fan-out 的嵌套元素误当成 `Sample`。

正确原则：

```python
rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
```

然后把这个 `rollout_id` 写回每个兄弟 sample。不要只用 `sample.index` 兜底后留在局部变量里；深层 fan-out sibling 的 `rollout_id=None` 会被主动拒绝。

当前基线还有三条结构性限制：

- group RM 把 `list[list[Sample]]` 直接传给 batched RM，随后对外层 list 写 `.reward`，fan-out 时会失败。
- partial abort 遍历 group 后直接读 `.response/.metadata`，fan-out 时也会失败。
- dynamic/sample/all-samples filter 在拍平前运行，插件必须自己支持嵌套形状。

因此 fan-out 场景优先使用非 group RM，并为 filter/abort 写端到端测试；若必须组合 group RM，当前基线通常需要自定义完整 rollout 或上游修复，不能靠配置消除形状冲突。

## 5. custom RM 在 group 模式下返回错位

普通 RM 是 `args, sample` 到一个 float；group RM 是 `args, samples` 到一个 float list。业务契约要求 batch 返回长度等于输入 `samples` 长度，但源码回填使用 `zip(..., strict=False)`：少返回会静默留下未赋 reward，多返回会被静默丢弃。源码入口：`slime/rollout/sglang_rollout.py` L272-L276、L326-L331；官方签名见 `docs/en/get_started/customization.md` L131-L136。

如果你同时支持两种模式，建议写两个函数或在函数里显式区分输入类型，并在 RM 内 `assert len(rewards) == len(samples)`。还要注意 path 优先级不对称：单样本 RM 支持 `sample.custom_rm_path` 覆盖全局 path，group RM 只看全局 `args.custom_rm_path`。`asyncio.gather` 中任一单样本 RM 异常会让整批失败，没有逐样本隔离。

## 6. filter 后样本数或 mask 对不上

`rollout_sample_filter` 不应该删除列表元素，而是原地设置 `Sample.remove_sample`。官方签名见 `docs/en/get_started/customization.md` L209-L211。

直接删除样本会改变 group 结构，后续 `n_samples_per_prompt`、rollout_id、reward 对齐和 DP schedule 都可能被破坏。若目标是“不参与 loss”，设置 `remove_sample` 才是正确边界。fan-out 时输入可能嵌套一层；若插件按 `for sample in group` 直接写属性，先递归拍平或拒绝该组合。

## 7. custom loss path 配了但没生效

检查是否设置了 `--loss-type custom_loss`。很多 loss 相关 path 只是提供实现，实际分支仍由 loss type 或 reducer 配置决定。若只是改变 policy loss 归约，优先确认是否应该用 `--custom-pg-loss-reducer-function-path`，不要直接重写完整 loss。

pg loss reducer 的官方输入见 `docs/en/get_started/customization.md` L288-L294。

## 8. rollout_data_postprocess 之后训练崩在 shape

这个 hook 在 advantage/return 之后、日志和训练之前调用。源码入口：`slime/backends/megatron_utils/actor.py` L511-L512。实际签名是 `(args, rollout_id, rollout_data)`；官方 customization 文档中的二参数示例已经漂移，不能照抄。

排查顺序：

1. 每个字段的 batch 维长度是否一致。
2. `tokens`、`response_lengths`、`loss_masks` 是否仍能互相解释。
3. 新增字段是否被后续 collate、logger 或 loss reducer 期待为 tensor。
4. 多 rank 情况下是否所有 rank 都做了同样的字段变换。

这个 hook 适合小范围修正，不适合重建整个 rollout 语义。

## 9. DataSource 续训或 partial rollout 后乱序

自定义 DataSource 应支持 `get_samples`、`add_samples`、`save`、`load` 和 `__len__`。官方要求见 `docs/en/get_started/customization.md` L387-L401；抽象基类也列出五项方法，但运行时并不会验证你的状态语义是否完整。

如果只实现单向读取，partial abort、buffer 回填和 checkpoint 恢复都会缺状态。默认返回形状是 `list[list[Sample]]`，扁平 list 会让 group RM 和 `n_samples_per_prompt` 语义错位。`RolloutDataSourceWithBuffer.add_samples` 还主动断言每个 group 长度等于 `n_samples_per_prompt`，所以 fan-out 结果若在 partial abort 中回填，并不天然满足 buffer 契约。

## 10. harness 里改了 model_label 但模型没变

`HarnessContext.model_label` 是外部 CLI 看到的模型名。Slime adapter 忽略它，实际 serving 的模型由后端 SGLang engine 加载的权重决定。源码入口：`slime/agent/harness/common.py` L37-L48。

因此，改 harness 配置只能影响 CLI 请求里的 model 字段，不能改变 rollout engine 的模型权重。真正的模型切换要看启动参数、权重同步或 SGLang engine 配置。

## 11. XML tool call 没被解析

优先确认是否已经被标准 `FunctionCallParser` 解析；XML fallback 只有在标准 parser 没产出 tool use 且有 tools schema 时才运行。源码入口：`slime/agent/parsing.py` L67-L85 和 L99-L110。

fallback 还要求 tool name 出现在 schema 中。模型输出了未知工具名时，原文本会保留而不会生成 tool use；XML 参数值全部是字符串。标准 function parser 的参数 JSON 错误会标记 `ill_formed`，但 reasoning parser 异常不会在这一层被兜底，两类解析失败语义并不对称。

## 12. contract tests 应该怎么用

本专题相关的最低成本检查是：

```powershell
python -m pytest tests/plugin_contracts/test_plugin_rollout_contracts.py tests/plugin_contracts/test_plugin_generate_contracts.py tests/plugin_contracts/test_plugin_path_loading_contracts.py tests/plugin_contracts/test_plugin_runtime_hook_contracts.py -q
```

也可以直接给单个测试文件传你的 path，例如 rollout 函数或 custom generate path。契约测试只覆盖部分正常边界，不验证业务 reward、Ray 多进程 import、同步/异步错配、RM 长度错位或 fan-out 组合安全；业务正确性仍需要小规模 rollout 回放。
