# 源码阅读 Vault — AI Agent Orientation

> 本文件面向 AI 代理（Claude、Codex、Cursor 等）。首次进入此 vault 时请先阅读本文件。
>
> 本 vault 与 **OPD**（`F:\OPD`）、**ESC_papers**（`F:\ESC_papers`）采用同一套 Obsidian 管理习惯，但内容形态是**源码阅读笔记**，不是论文库。

---

## 1. Vault 概览

这是一个 **SGLang 源码阅读知识库**。核心可读内容在 `sglang_reading/`：32 批次、自包含中文讲解 + 内嵌源码。`sglang/` 是 upstream 源码对照目录，**读者日常不必打开**。

| 指标 | 数值 |
|------|------|
| 阅读批次 | 32 / 32 已完成 |
| 标准文档 | 每批 6 篇（MOC + 01–04 + checkpoint），**唯一文件名** |
| 内嵌代码基线 | sglang commit `70df09b` |

---

## 2. 目录结构

| 目录 | 用途 | AI 可写？ |
|------|------|-----------|
| `sglang_reading/` | **主内容**：批次阅读笔记、索引、进度 | ✅ 可写 |
| `sglang/` | upstream 源码（对照用） | ⚠️ 只读，除非用户要求改 upstream |
| `90_meta/` | 维护脚本、Obsidian 规范 | ✅ 可写 |
| `.obsidian/` | Obsidian 配置 | ❌ **永远不要修改** |

### sglang_reading 内部结构

| 路径 | 说明 |
|------|------|
| `00-方法论/` | 批次 01：阅读方法论 + 五层架构 |
| `01-启动与入口/` | 批次 02–05 |
| `02-请求调度/` | 批次 06–10 |
| `03-模型执行/` | 批次 11–14 |
| `04-内存与Attention/` | 批次 15–19 |
| `05-高级特性/` | 批次 20–23、31–32 |
| `06-扩展组件/` | 批次 24–29 |
| `07-总结与索引/` | 批次 30：onboarding + 全链路追踪 |
| `_TEMPLATE/` | 批次文档模板 | ⚠️ 修改需用户确认 |
| `PLAN.md` | 总计划与写作规范 |
| `progress.md` | 阅读进度 |

---

## 3. 新会话启动协议

1. **读本文件**（`AGENTS.md`）
2. **读 [[index]]** — vault 导航
3. **读 [[SGLang源码阅读指南]]** — 阅读体系入口
4. **读 [[90_meta/obsidian-syntax-rules]]** — 命名、双链、frontmatter
5. **读 [[obsidian-graph-presets]]** — 图谱过滤（必读）
6. 按任务深入具体批次或 [[04-导读路径]]

---

## 4. 边界规则

### ❌ 绝对禁止

- 修改 `.obsidian/` 下任何文件
- 删除 `sglang/` 源码或 `sglang_reading/` 已发布笔记（除非用户明确要求）
- 批量重命名批次目录或文件（除非用户明确要求）
- 编造源码行为、行号、函数签名

### ⚠️ 谨慎操作

- 修改 `_TEMPLATE/`、`PLAN.md` 结构
- 将相对 Markdown 链接批量改为双链（需保持链接可解析）
- 在 Mermaid 代码块外做 `\n` → `<br/>` 替换（会破坏 Python 源码示例）

### ✅ 自由操作

- 编辑 `sglang_reading/` 下笔记内容
- 添加/修正 Obsidian 双链 `[[]]`
- 更新 `progress.md`、索引页、维护日志
- 运行 `90_meta/fix_mermaid_newlines.py` 修复 Mermaid 换行

---

## 5. 语言与写作规范

- **正文使用中文**
- 英文仅限：源码标识符、路径、CLI 命令、技术术语首次中英对照
- 每篇标准文档须含 **Explain → Code → Comment** 结构（见 `PLAN.md` §六）
- 架构图：**Mermaid**（流程/分层）或 **ASCII 盒状图**（复杂训练/数据流）；二者 Obsidian 均支持

---

## 6. Obsidian 适配要点（与 OPD / ESC 对齐）

| 规则 | 说明 |
|------|------|
| **唯一文件名** | `{模块名}-{文档类型}.md`，禁止泛化 `README` / `01-核心概念` |
| frontmatter tags | `sglang/batch/NN` + `sglang/doc/类型`，用于 Graph 过滤 |
| Mermaid 换行 | `<br/>`，禁止 `\n` |
| 双链 | `[[07-Scheduler-01-核心概念]]`，禁止 `./01-核心概念.md` |
| 图谱 | 默认过滤与颜色已写入 `.obsidian/graph.json`；见 [[91_dashboard/graph-hub]] |

完整语法见 [[90_meta/obsidian-syntax-rules]]。

---

## 7. 维护检查清单

- [ ] 文件名含模块前缀（无重复 `README` / `01-核心概念`）
- [ ] frontmatter `tags` 含 batch + doc_type
- [ ] Mermaid 块内无 `\n`；双链无 `./0X-*.md` 旧路径
- [ ] 代码块内无被误拆的 `[[...]]`
- [ ] 每篇文档有且仅有一个 H1
- [ ] `_archive/` 内容未链入主阅读路径

---

*最后更新: 2026-07-02 — 图谱唯一命名 + tag 过滤*
