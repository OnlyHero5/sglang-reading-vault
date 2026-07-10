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
updated: 2026-07-10
---
# 自定义扩展 · 学习检查

这一页用于验收两件事：读者是否真正理解 hook 边界，以及本专题文档是否已经摆脱旧结构并保留源码证据。

## 1. 读者自测

- [ ] 能说明为什么 agentic workflow 通常先选 `custom_generate + custom_rm`，而不是直接替换 `rollout_function`。
- [ ] 能说出 `load_function` 做什么、不做什么。
- [ ] 能区分 `Sample`、`list[Sample]`、`list[list[Sample]]`、`train_data`、`rollout_data`。
- [ ] 能解释 fan-out 时为什么兄弟样本要共享 `rollout_id`。
- [ ] 能说明 `rollout_sample_filter` 为什么设置 `remove_sample`，而不是删除样本。
- [ ] 能说出 batch RM 的返回长度约束。
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

这些命令验证的是 import path、签名、返回结构和副作用。训练效果、reward 质量和分布式一致性还需要小规模任务回放。

## 3. 文档专题验收

在 vault 根目录运行：

```powershell
$path='slime_reading/高级特性/自定义扩展'
$oldMarkers = @('Explain','Code','Comment') | ForEach-Object { $_ + '：' }
$oldMarkers += @('Explain','Code','Comment') | ForEach-Object { $_ + ':' }
rg -n ($oldMarkers -join '|') $path
$ellipsisPattern = ('\.' * 3) + '|' + ([char]0x2026 + [char]0x2026)
rg -n $ellipsisPattern $path
node maintenance/audit_source_evidence.mjs --note 'slime_reading/高级特性/自定义扩展/Slime-自定义扩展-源码走读.md'
node maintenance/audit_wikilinks.mjs
git diff --check -- $path 'maintenance/知识库重构计划.md'
```

期望结果：

- 旧三段式标记命中为 0。
- 旧式省略占位写法不再出现。
- [[Slime-自定义扩展-源码走读]] 应覆盖 `customization.md`、`misc.py`、`rollout.py`、`sglang_rollout.py`、`rm_hub`、`actor.py`、`parsing.py`、`harness/common.py` 的关键边界。
- wikilink 审计 broken target 为 0。
- diff check 无 whitespace error。

## 4. 小规模运行观察点

真正接入自定义 hook 后，先看这些信号：

| 观察点 | 正常信号 | 异常指向 |
|--------|----------|----------|
| 启动期 | path 能 import，actor/rollout manager 初始化完成 | path、PYTHONPATH、包安装问题 |
| generate | 每个 `Sample` 有 tokens、response、status | custom generate 返回对象不完整 |
| fan-out | 兄弟样本共享 `rollout_id` | group reward 或 advantage 错位 |
| RM | reward 数量与样本数量对齐 | batch RM 返回长度错误 |
| filter | 样本保留但 loss mask 可被屏蔽 | 直接删除样本破坏 group 结构 |
| actor postprocess | batch 字段长度一致 | rollout_data 原地修改破坏 shape |

## 5. 衔接阅读

继续看实例落地：[[Slime-插件与示例]]。
回看默认 rollout：[[Slime-SGLang-Rollout]]。
回看 agent 轨迹：[[Slime-Agent轨迹]]。
