---
title: "插件与示例 · 学习检查"
type: exercise
framework: slime
topic: "插件与示例"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 插件与示例 · 学习检查

这一页验收读者是否能把 example 的边界迁移到自己的项目，并检查本专题文档是否清理干净。

## 1. 读者自测

- [ ] 能说明 Search-R1 为什么是 `custom_generate + custom_rm`。
- [ ] 能指出 multi_agent 当前使用 `custom_generate`，不是 `rollout_function`。
- [ ] 能解释 Search-R1 中 model token 与 observation token 的 loss mask 分界。
- [ ] 能说明 rollout_buffer 的 `/start_rollout`、`/buffer/write`、`/get_rollout_data` 各自职责。
- [ ] 能说出 generator 的必需符号：`TASK_TYPE` 和 `run_rollout`。
- [ ] 能解释 rollout_buffer wrapper 如何把 OpenAI messages 转成 `Sample`。
- [ ] 能区分 runnable example 和 GLM5 这种模型结构插件。

## 2. 文档专题验收

在 vault 根目录运行：

```powershell
$path='slime_reading/扩展与生态/插件与示例'
$oldMarkers = @('Explain','Code','Comment') | ForEach-Object { $_ + '：' }
$oldMarkers += @('Explain','Code','Comment') | ForEach-Object { $_ + ':' }
rg -n ($oldMarkers -join '|') $path
$ellipsisPattern = ('\.' * 3) + '|' + ([char]0x2026 + [char]0x2026)
rg -n $ellipsisPattern $path
node maintenance/audit_source_evidence.mjs --note 'slime_reading/扩展与生态/插件与示例/Slime-插件与示例-源码走读.md'
node maintenance/audit_wikilinks.mjs
git diff --check -- $path 'maintenance/知识库重构计划.md'
```

期望：

- 旧结构标记为 0。
- 旧式省略占位写法不再出现。
- [[Slime-插件与示例-源码走读]] 覆盖 examples README、Search-R1、multi_agent、rollout_buffer、GLM5 的关键接入点。
- wikilink broken target 为 0。
- diff check 无 whitespace error。

## 3. example 级 smoke

Search-R1 最小检查：

```powershell
python tests/plugin_contracts/test_plugin_generate_contracts.py --custom-generate-function-path examples.search_r1.generate_with_search.generate -q
```

multi_agent 最小检查：

```powershell
python tests/plugin_contracts/test_plugin_generate_contracts.py --custom-generate-function-path examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents -q
```

rollout_buffer wrapper 最小检查：

```powershell
python tests/plugin_contracts/test_plugin_rollout_contracts.py --rollout-function-path slime_plugins.rollout_buffer.rollout_buffer_example.generate_rollout -q
```

这些命令只检查接口形状。Search-R1 的检索服务、multi_agent 的 agent system、rollout_buffer 的 HTTP 服务仍需要按各自 README 做小规模运行验证。

## 4. 收官衔接

完成本专题后，扩展生态主线已经接到 customization。继续阅读 [[Slime-总结复盘]]，把训练闭环、rollout、权重同步和扩展点统一复盘。
