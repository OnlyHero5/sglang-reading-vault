---
title: "阅读方法 · 学习检查"
type: exercise
framework: slime
topic: "阅读方法"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# 阅读方法 · 学习检查

## 1. 读者自测

- [ ] 能用一句话说明 Slime 的两大核心能力。
- [ ] 能画出 Training / Rollout / Data Buffer 三角，并标出数据方向。
- [ ] 能口述 `generate → train → update_weights` 各步输入输出。
- [ ] 能解释 Slime native 透传相对多 backend 抽象层的意义。
- [ ] 能说明 Data Buffer 为什么不是一个必须独立存在的 daemon。
- [ ] 能指出 `train.py` 为什么被暴露成可读主循环。
- [ ] 能说出什么时候看 rollout-only debug，什么时候看 train-only debug。

## 2. 快速自测题

| 问题 | 期望答案 |
|------|----------|
| Slime 三角第三角 Data Buffer 在代码里主要对应谁 | RolloutManager、DataSource、Sample group、train_data 交付 |
| 为什么博文强调不 wrap trainer class | 方便移动同步点、实验同步/异步和 colocate/decoupled |
| SGLang 参数如何传入 | CLI 加 `--sglang-` 前缀，由独立 parser 解析并合并到 args |
| 为什么 custom workflow 不 fork training kernel | 它们通过 `Sample` / Data Buffer 契约回到同一训练闭环 |

## 3. 文档专题验收

在 vault 根目录运行：

```powershell
$path='slime_reading/阅读方法'
$oldMarkers = @('Explain','Code','Comment') | ForEach-Object { $_ + '：' }
$oldMarkers += @('Explain','Code','Comment') | ForEach-Object { $_ + ':' }
rg -n ($oldMarkers -join '|') $path
$ellipsisPattern = ('\.' * 3) + '|' + ([char]0x2026 + [char]0x2026)
rg -n $ellipsisPattern $path
node maintenance/audit_source_evidence.mjs --note 'slime_reading/阅读方法/Slime-阅读方法-源码走读.md'
node maintenance/audit_wikilinks.mjs
git diff --check -- $path 'maintenance/知识库重构计划.md'
```

期望：

- 旧结构标记为 0。
- 旧式省略占位写法不再出现。
- [[Slime-阅读方法-源码走读]] 覆盖 README、愿景博文、参数透传、setup 和 requirements。
- wikilink broken target 为 0。
- diff check 无 whitespace error。

## 4. 下一步

继续读 [[Slime-训练主循环]]，把方法论里的闭环落实到入口脚本和 Ray actor 调用。
