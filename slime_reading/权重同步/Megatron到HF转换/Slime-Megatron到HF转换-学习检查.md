---
title: "Megatron到HF转换 · 学习检查"
type: exercise
framework: slime
topic: "Megatron到HF转换"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Megatron到HF转换 · 学习检查

## 读者能做什么

- [ ] 能画出 `--load -> load_checkpoint -> Megatron/HF branch -> model` 的加载路径。
- [ ] 能解释 `bridge` 和 `raw` 在加载与保存上的不对称能力。
- [ ] 能沿 `Actor.save_model -> save_hf_model_to_path -> HfWeightIteratorDirect -> convert_to_hf -> writer -> index` 复述一次导出。
- [ ] 能说明 `--hf-checkpoint` 是 raw 保存的资产模板，不是当前训练权重来源。
- [ ] 能指出 converter 改动会同时影响 `--save-hf`、[[Slime-磁盘权重同步]] 和 [[Slime-分布式权重同步]]。
- [ ] 能说出 3 个失败模式及入口：raw 加载 HF 被拒、duplicate HF tensor、QKV shape 不匹配。

## 可执行验证

raw saver 单测：

```powershell
python -m pytest tests/utils/test_hf_checkpoint_saver.py
```

专题源码引用审计：

```powershell
node maintenance/audit_source_evidence.mjs --note 'slime_reading/权重同步/Megatron到HF转换/Slime-Megatron到HF转换-源码走读.md'
```

专题旧结构残留：

```powershell
$old = 'Explain','Code','Comment' | ForEach-Object { [regex]::Escape($_) + '[:：]' }
rg -n ($old -join '|') 'slime_reading/权重同步/Megatron到HF转换'
$dots = ([string][char]46) * 3
$cn = ([string][char]0x2026) * 2
rg -n ([regex]::Escape($dots) + '|' + [regex]::Escape($cn)) 'slime_reading/权重同步/Megatron到HF转换'
```

全 vault 链接审计：

```powershell
node maintenance/audit_wikilinks.mjs
```

## 预期现象

- raw saver 测试通过，尤其是拒绝覆盖模板目录、资产复制跳过旧权重、multi-writer finalize 和 pending chunk flush。
- 专题旧结构搜索没有命中。
- [[Slime-Megatron到HF转换-源码走读]] 应覆盖 `checkpoint.py`、`actor.py`、`hf_checkpoint_saver.py`、`hf_weight_iterator_direct.py`、`megatron_to_hf/__init__.py`、`qwen2.py` 的关键边界。
- 全 vault wikilink 审计 broken target 为 0。

## 权重同步收口

完成本专题后，权重同步三段闭环应能连起来：

| 专题 | 读者应掌握 |
|------|------------|
| [[Slime-分布式权重同步]] | 在线 NCCL 同步如何把 HF tensor 流推给 rollout engine |
| [[Slime-磁盘权重同步]] | 磁盘版本目录如何发布、reload 和清理 |
| [[Slime-Megatron到HF转换]] | Megatron 权重如何导出为标准 HF checkpoint，并服务保存与 disk sync |
