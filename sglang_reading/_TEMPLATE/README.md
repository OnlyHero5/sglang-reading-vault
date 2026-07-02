---
type: template
title: "批次文档模板"
tags:
  - sglang/template
updated: 2026-07-02
---

# 批次文档模板

> 复制本结构到各批次文件夹。**读者只读 sglang_reading，不读 sglang**——所有源码必须内嵌在文档中。

## 文件说明（Obsidian 唯一命名）

| 文件 | 用途 | doc_type tag |
| --------------------- | ------------------- | ------------------------ |
| `{模块名}-MOC.md` | 批次概述、目标、源码范围、验收标准 | `sglang/doc/moc` |
| `{模块名}-核心概念.md` | 术语、设计动机、架构位置 | `sglang/doc/concept` |
| `{模块名}-源码走读.md` | 按调用顺序的代码精读（**主文档**） | `sglang/doc/walkthrough` |
| `{模块名}-数据流与交互.md` | 数据结构、消息流、模块边界 | `sglang/doc/dataflow` |
| `{模块名}-关键问题.md` | FAQ、易错点、对比分析 | `sglang/doc/faq` |
| `{模块名}-checkpoint.md` | 验收勾选清单 | `sglang/doc/checkpoint` |

**示例（Scheduler）：** `07-Scheduler-MOC.md`、`07-Scheduler-核心概念.md` …

> ⚠️ 禁止 `README.md`、`01-核心概念.md` 等泛化名 — Obsidian 图谱会重复节点。

**全批合计：≥ 15 段内嵌代码，≥ 200 行。**

### frontmatter 模板

```yaml
---
type: batch-doc
module: 07-Scheduler
batch: "07"
doc_type: concept
title: "Scheduler · Scheduler · 核心概念"
tags:
 - sglang/batch/07
 - sglang/module/scheduler
 - sglang/doc/concept
updated: 2026-07-02
---
```

模块间链接用双链：`[[06-TokenizerManager-03-数据流与交互]]`，不用 `./03-数据流与交互.md`。

---

## 写作格式：ETC 三段式（强制）

```markdown
### 3.1 HTTP 默认启动路径

**Explain：** 当未开启 gRPC、Ray、Encoder 模式时，`run_server` 走 HTTP 分支，
调用 `srt.entrypoints.http_server.launch_server`——这是绝大多数部署的入口。

**Code：**

```python
# 来源：python/sglang/launch_server.py L47-L51
 else:
 # Default mode: HTTP mode.
 from sglang.srt.entrypoints.http_server import launch_server

 launch_server(server_args)
```

**Comment：**
- `else` 为默认分支：普通 OpenAI 兼容 HTTP 服务
- 延迟 import 避免未使用 HTTP 时加载 FastAPI 等依赖
- 下一批（03）将展开 `launch_server` 内部如何挂载路由与 Engine
```

---

## 源码走读结构模板

```markdown
# 批次 XX：源码走读

## 走读顺序

1. `文件A.py` — 入口
2. `文件B.py` — 核心类

---

## 1. 文件A.py

### 1.1 函数 foo()

（ETC 三段式）
```

---

## 数据流文档结构模板

```markdown
# 批次 XX：数据流与交互

## 1. 架构位置

（Mermaid 节点换行用 `<br/>`，不用 `\n`）

## 2. 输入 / 输出

## 3. 上下游连接

## 4. 典型数据流（逐步 + 代码）
```

---

## checkpoint 模板

```markdown
# 批次 XX 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明本模块职责
- [ ] 能画出本模块在全局架构中的位置

## 维护者检查

- [ ] frontmatter tags 完整
- [ ] 已更新 [[progress]]
```

---

## 禁止事项

1. ❌ 泛化文件名（`README.md`、`01-核心概念.md`）
2. ❌ Mermaid 标签内使用 `\n`（用 `<br/>`）
3. ❌ 只写「详见 `xxx.py` 第 N 行」而不贴代码
4. ❌ 代码块内伪造 Obsidian 双链
