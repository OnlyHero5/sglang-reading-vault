---
title: "自定义扩展 · 学习检查"
type: exercise
framework: slime
topic: "自定义扩展"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# 自定义扩展 · 学习检查

这一页验收读者是否真正理解 hook 边界，并能在接入自定义逻辑前验证 import、签名、返回结构和运行时不变量。

## 1. 读者自测

- [ ] 能说明为什么 agentic workflow 通常先选 `custom_generate + custom_rm`，而不是直接替换 `rollout_function`。
- [ ] 能说出 `load_function` 做什么、不做什么。
- [ ] 能区分 `Sample`、`list[Sample]`、`list[list[Sample]]`、`train_data`、`rollout_data`。
- [ ] 能解释 fan-out 时为什么兄弟样本要共享 `rollout_id`。
- [ ] 能说明 `rollout_sample_filter` 为什么设置 `remove_sample`，而不是删除样本。
- [ ] 能说出 batch RM 的返回长度约束。
- [ ] 能解释为什么“应当等长”不等于源码会拒绝错长 reward。
- [ ] 能说明 fan-out 在训练 filter 前是嵌套形状、到 `RolloutManager` 才递归拍平。
- [ ] 能指出 group RM、partial abort 与 fan-out 的当前组合风险。
- [ ] 能说明为什么 `custom_generate(**kwargs)` 收不到 `evaluation`。
- [ ] 能说明 `rollout_data_postprocess` 的调用时机和 shape 风险。
- [ ] 能用 `tests/plugin_contracts/` 检查自己的插件 path。

## 2. 插件契约测试

在 `slime/` upstream 目录内运行：

```powershell
python -m pytest tests/plugin_contracts/test_plugin_rollout_contracts.py tests/plugin_contracts/test_plugin_generate_contracts.py tests/plugin_contracts/test_plugin_path_loading_contracts.py tests/plugin_contracts/test_plugin_runtime_hook_contracts.py -q
```

按自定义 path 单独验证时，可直接调用对应测试文件：

```powershell
python tests/plugin_contracts/test_plugin_generate_contracts.py --custom-generate-function-path my_project.generate.custom_generate -q
python tests/plugin_contracts/test_plugin_rollout_contracts.py --rollout-function-path my_project.rollout.generate_rollout -q
python tests/plugin_contracts/test_plugin_path_loading_contracts.py --custom-rm-path my_project.reward.custom_rm -q
```

这些命令验证的是若干 import path、签名、正常返回结构和副作用。它们没有覆盖所有真实组合；训练效果、reward 质量、fan-out 组合和分布式一致性还需要专门测试与小规模任务回放。

## 3. 契约定位验收

在 vault 根目录运行：

```powershell
rg -n 'def load_function|custom_generate_function_path|rollout_function_path|custom_rm_path' slime/slime
rg -n 'rollout_id|remove_sample|rollout_data_postprocess' slime/slime
rg -n 'run_contract_test_for_file|path_args' slime/tests/plugin_contracts
```

期望结果：

- 能指出 path 字符串在哪里被解析成 callable，以及 import 成功为什么仍不能证明契约正确。
- 能区分替换整轮 rollout、替换单样本 generate、替换 reward 三种扩展半径。
- 能定位 fan-out 的 `rollout_id`、过滤的 `remove_sample` 和 actor 侧 postprocess 三个高风险边界。
- 能解释 contract tests 只验证接口与副作用，不验证 reward 质量、训练收敛或分布式一致性。
- 能从源码指出 RM 回填为什么可能静默错位，以及官方 `rollout_data_postprocess` 示例为何不能替代实际调用点。

## 4. 边界复现实验

下面五项不要求启动 GPU。先在 vault 根目录运行这个静态边界检查；它不 import Slime，因此适合本机缺 Ray/SGLang 时使用：

```powershell
@'
from pathlib import Path

rollout = Path("slime/slime/rollout/sglang_rollout.py").read_text(encoding="utf-8")
actor = Path("slime/slime/backends/megatron_utils/actor.py").read_text(encoding="utf-8")
docs = Path("slime/docs/en/get_started/customization.md").read_text(encoding="utf-8")

assert '"evaluation" in inspect.signature(custom_generate_func).parameters' in rollout
assert "zip(samples_need_reward, rewards, strict=False)" in rollout
assert "zip(group, rewards, strict=False)" in rollout
assert "sample.reward = reward" in rollout
assert "for sample in group:" in rollout and "if sample.response" in rollout
assert "self.rollout_data_postprocess(self.args, rollout_id, rollout_data)" in actor
assert "def postprocess_function(args, samples: list[list[Sample]])" in docs

print("PASS: 当前基线仍存在本文记录的 evaluation、非严格 RM 回填、fan-out 与 postprocess 签名边界")
'@ | python -
```

预期输出以 `PASS:` 开头。若断言失败，说明 upstream 已变化，应重新阅读调用点，而不是机械更新行号。随后把以下五项写成你自己插件的最小单元测试：

1. 自定义 RM 输入两个样本、只返回一个 reward；断言插件自己的长度检查会失败，而不是依赖 Slime 的非严格 `zip`。
2. 自定义 generate 只声明 `**kwargs`；断言 `evaluation` 不会被传入，再改为显式参数验证差异。
3. 两个原始 sample 各 fan-out 两个片段；画出 filter 前的 `list[list[Sample]]` 与训练转换前拍平后的四个 `Sample`。
4. 在 `group_rm=True` 下让 custom generate 返回 list；定位默认代码会在哪次 `.reward` 赋值上把 list 当成 sample。
5. 对照 `docs/en/get_started/customization.md` 与 actor 调用点，验证 `rollout_data_postprocess` 必须接收 `rollout_id`。

预期不是“所有组合都通过”，而是把当前基线的支持边界稳定复现出来。若测试暴露的是框架缺口，应在插件文档中声明限制，或切换到自定义完整 rollout。静态检查只能证明相关调用形态仍存在；真实插件的返回对象和副作用仍需运行 contract tests 与小规模 rollout。

## 5. 小规模运行观察点

真正接入自定义 hook 后，先看这些信号：

| 观察点 | 正常信号 | 异常指向 |
|--------|----------|----------|
| 启动期 | path 能 import，actor/rollout manager 初始化完成 | path、PYTHONPATH、包安装问题 |
| generate | 每个 `Sample` 有 tokens、response、status | custom generate 返回对象不完整 |
| fan-out | 兄弟样本共享非空 `rollout_id`，所用 RM/filter 支持嵌套输入 | group RM、abort、filter 把 list 当 Sample |
| RM | 插件主动校验 reward 数量与样本数量对齐 | 非严格 zip 静默留下或丢弃 reward |
| filter | 样本保留但 loss mask 可被屏蔽 | 直接删除样本破坏 group 结构 |
| actor postprocess | batch 字段长度一致 | rollout_data 原地修改破坏 shape |

## 6. 衔接阅读

继续看实例落地：[[Slime-插件与示例]]。
回看默认 rollout：[[Slime-SGLang-Rollout]]。
回看 agent 轨迹：[[Slime-Agent轨迹]]。
