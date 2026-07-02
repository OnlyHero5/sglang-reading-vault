# sglang-reading-vault

> **SGLang 源码阅读知识库** — 32 个专题、自包含中文讲解 + 内嵌源码，基于 [SGLang](https://github.com/sgl-project/sglang) commit `70df09b`。

[![Progress](https://img.shields.io/badge/阅读进度-32%2F32-brightgreen)](#阅读进度)
[![Obsidian](https://img.shields.io/badge/Obsidian-Vault-7c3aed)](#快速开始)

---

## 这是什么

本仓库是一个 **Obsidian 知识库（Vault）**，系统性地拆解 [SGLang](https://github.com/sgl-project/sglang) 推理框架源码。与论文库、OPD 笔记库采用同一套 Obsidian 管理习惯，但内容形态是**源码阅读笔记**。

核心设计原则：

| 原则 | 说明 |
|------|------|
| **自包含** | 每篇文档内嵌足够源码片段 + 中文讲解，读者无需打开 upstream |
| **可追溯** | 代码块标注 sglang 路径与行号，便于与 upstream 对照 |
| **批次闭环** | 每模块 6 篇标准文档（MOC + 01–04 + checkpoint） |
| **图谱导航** | frontmatter tags + Obsidian Graph 预设，按模块/批次过滤 |

---

## 目录结构

```
sglang-reading-vault/
├── README.md                 # 本文件
├── AGENTS.md                 # AI 代理入门指南
├── index.md                  # Vault 导航首页
├── sglang_reading/           # ★ 主内容：32 批次阅读笔记
│   ├── SGLang源码阅读指南.md  # 阅读体系总入口
│   ├── PLAN.md               # 写作规范与批次计划
│   ├── progress.md           # 阅读进度（32/32 已完成）
│   ├── 00-方法论/             # 批次 01
│   ├── 01-启动与入口/         # 批次 02–05
│   ├── 02-请求调度/           # 批次 06–10
│   ├── 03-模型执行/           # 批次 11–14
│   ├── 04-内存与Attention/    # 批次 15–19
│   ├── 05-高级特性/           # 批次 20–23、31–32
│   ├── 06-扩展组件/           # 批次 24–29
│   └── 07-总结与索引/         # 批次 30 + 全链路追踪
├── 91_dashboard/             # Dataview 可视化仪表盘
├── 90_meta/                  # 维护脚本与 Obsidian 规范
└── .obsidian/                # Obsidian 配置（含 Graph 预设）
```

> `sglang/` upstream 源码**不在本仓库中**（见下方「可选：对照源码」）。日常阅读只需 `sglang_reading/`。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone git@github.com:OnlyHero5/sglang-reading-vault.git
cd sglang-reading-vault
```

### 2. 用 Obsidian 打开

1. 启动 [Obsidian](https://obsidian.md/)
2. **打开文件夹作为仓库** → 选择克隆下来的根目录
3. 从 [[index]] 或 `sglang_reading/SGLang源码阅读指南.md` 开始阅读

### 3. 推荐阅读路径

| 读者类型 | 起点 |
|----------|------|
| 零基础 | `sglang_reading/07-总结与索引/00-零基础先修.md` |
| 有 LLM serving 经验 | `sglang_reading/SGLang源码阅读指南.md` → `04-导读路径.md` |
| 想跟一条请求走完 | `sglang_reading/07-总结与索引/全链路请求追踪.md` |
| 生产排障 | `sglang_reading/07-总结与索引/09-生产排障速查.md` |

---

## 阅读进度

```
[████████████████████████████████] 32/32 (100%)
```

| 阶段 | 批次 | 主题 | 状态 |
|------|------|------|------|
| I 地基 | 01–05 | 启动与入口 | ✅ |
| II 调度 | 06–10 | 请求调度 | ✅ |
| III 执行 | 11–14 | 模型执行 | ✅ |
| IV 内存 | 15–19 | 内存与 Attention | ✅ |
| V 高级 | 20–23 | 高级特性 | ✅ |
| V+ 运维 | 31–32 | 可观测性 / 热更新 | ✅ |
| VI 扩展 | 24–29 | 扩展组件 | ✅ |
| VII 收官 | 30 | 全链路复盘 | ✅ |

详细进度见 [`sglang_reading/progress.md`](sglang_reading/progress.md)。

---

## 可选：对照 upstream 源码

笔记内嵌的代码基线为 SGLang commit **`70df09b`**。若需本地对照 upstream：

```bash
git clone https://github.com/sgl-project/sglang.git sglang
cd sglang
git checkout 70df09b
```

将 `sglang/` 放在 vault 根目录即可与笔记中的路径引用对齐。

---

## 维护与规范

| 文档 | 说明 |
|------|------|
| [`AGENTS.md`](AGENTS.md) | AI 代理（Cursor / Claude 等）首次进入时的操作指南 |
| [`90_meta/obsidian-syntax-rules.md`](90_meta/obsidian-syntax-rules.md) | 命名、双链、frontmatter 规范 |
| [`sglang_reading/PLAN.md`](sglang_reading/PLAN.md) | 批次计划与 Explain → Code → Comment 写作规范 |
| [`91_dashboard/home.md`](91_dashboard/home.md) | Dataview 可视化入口 |

---

## 相关链接

- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [SGLang 文档](https://docs.sglang.ai/)

---

## License

阅读笔记内容为个人学习整理。SGLang 源码版权归 [sgl-project](https://github.com/sgl-project) 所有，遵循其 upstream 许可证。
