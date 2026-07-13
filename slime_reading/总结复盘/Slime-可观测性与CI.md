---
title: "Slime 可观测性与 CI"
type: reference
framework: slime
topic: "总结复盘"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/reference
  - source-reading
updated: 2026-07-13
---
# Slime 可观测性与 CI

> **读者任务：** 为“慢、错、挂、漂移”选择正确证据。Trace、profile、日志、CPU tests 和 GPU e2e 观察的层级不同，不能互相替代。

## 1. 先看证据矩阵

| 你想证明什么 | 首选证据 | 它不能证明什么 |
|--------------|----------|----------------|
| 一条 Sample 在生成、RM、filter 中花了多久 | Sample trace + timeline viewer | CUDA kernel 细节、跨进程自动因果 |
| actor forward/backward/optimizer 慢在哪里 | PyTorch profiler | 外部服务等待与单样本语义 |
| 显存增长或 OOM 前分配历史 | torch memory snapshot / memray | 权重或 loss 数值正确 |
| hook 的 import、签名和返回 shape | CPU contract test | 真实 GPU collective 与模型数值 |
| Megatron + SGLang 闭环能跑通 | label-gated GPU e2e | 任意硬件/workload 的性能结论 |
| engine 崩溃后能否在指定边界恢复 | fault-tolerance e2e + version/log | 任意位置、任意外部副作用的事务恢复 |

## 2. Trace：跟踪 Sample，不是全局分布式追踪系统

trace carrier 保存在 `Sample.trace`，包含 version、trace id、sample/group id、attempt 和 event 列表。`trace_span` 写 start/end，`trace_event` 写瞬时事件，`trace_function` 可包同步或异步函数；SGLang metadata 还能被转换为 token、latency、throughput 和 PD prefill/decode 子 span。来源：`slime/utils/trace_utils.py` L16-L61、L244-L520。

默认 rollout 已在这些边界埋点：

```text
generate_and_rm_group
└─ generate_and_rm
   ├─ sglang_generate
   └─ reward_model
```

来源：`slime/rollout/sglang_rollout.py` L193-L334。

### Trace 的四个重要边界

- **best-effort：** 多数 trace helper 捕获自身异常并只写 debug log，避免观测代码打断训练；因此“没有 trace”不等于业务步骤没执行。
- **进程内上下文：** 父 span 依赖 `contextvars`。跨 Ray actor、HTTP 或其他进程时，必须显式 `export_trace` / `import_trace` 或把 carrier 放回 Sample；当前调用点不会替所有自定义服务自动传播。
- **墙钟时间：** event 使用 `time.time()`。跨机器时间线依赖时钟同步，不能把微小先后差当作严格 happens-before。
- **内存与文件成本：** event 直接追加到 Sample carrier；大批量、细粒度埋点会增加 rollout dump 和序列化成本。

`trace_next_attempt` 会增加 attempt 并写 `attempt_start`，但它不会自动回滚之前事件。重试分析要按 attempt 分段，而不是只看同一个 trace id。

### 保存与查看

```bash
python train.py ... \
  --save-debug-rollout-data /path/to/rollout_{rollout_id}.pt

python tools/trace_timeline_viewer.py /path/to/rollout_0.pt --no-serve
```

预期生成 `.trace_timeline_cache.json` 与 `.trace_timeline_viewer.html`。有 PD metadata 时出现合成的 `[P]`、`[D]` lane；没有时只显示基础 span。来源：`docs/en/developer_guide/trace.md` L9-L47。

## 3. Profile：训练进程里的算子与显存证据

`TrainProfiler` 的 API 暴露 `train_overall`、`train_actor` 和 `train_log_probs` 三种 target。overall profiler 在 actor 初始化结束时 start，并在 actor train 结束后按 rollout 调用 `step`。虽然 helper 定义了两个 iterator wrapper，当前基线的 actor 调用点只使用 `on_init_end()` 与 `step()`，没有调用 `iterate_train_actor()` / `iterate_train_log_probs()`；因此后两个 target 不能仅凭参数名就视为已接线。来源：`slime/utils/profile_utils.py` L13-L73、`slime/backends/megatron_utils/actor.py` L180-L187、L525-L535。

| 参数/组件 | 作用 | 使用提醒 |
|-----------|------|----------|
| `--use-pytorch-profiler` | 采集 shape、stack、memory、FLOPs，写 TensorBoard trace | schedule 的 active 必须为正；需要初始化 distributed rank |
| `--profile-target ...` | 选择 overall/actor/log_probs | 当前只有 overall 可见实际调用链；另外两个先核对调用点 |
| `--record-memory-history` | 启动 torch 或 memray recorder | 只有 target 包含 `train_overall` 才创建 |
| `--memory-snapshot-num-steps N` | 第 `N` 个 rollout step 停 memory recorder | memray 强制要求；torch 不填时只靠 OOM observer/进程结束 |

当前 helper 会 start 和 step profiler，但没有在该文件中显式调用 PyTorch profiler 的 `stop()`；长跑或异常退出时，应实际确认 trace handler 是否已经 flush，而不是只看到参数启用就认定产物完整。torch memory recorder stop 时会 dump snapshot 并关闭 history，但已挂载的 OOM observer 也不是通用资源生命周期管理器。来源：`slime/utils/profile_utils.py` L25-L56、L103-L143。

## 4. Debug mode：切断哪一段，要说准确

| 参数 | 实际切断点 | 适合场景 |
|------|------------|----------|
| `--debug-rollout-only` | rollout dump/log 后、Sample→train data 转换前返回；训练 actor 也直接返回 | 只验生成、RM、trace 和 Sample |
| `--debug-train-only` | pre-parse 时跳过 SGLang args/server；eval 返回；训练从 debug 数据路径进入 | 隔离 loss、micro-batch、actor/critic |
| `--save-debug-rollout-data` | 保存原始 rollout samples 和 trace | 离线 viewer、之后重放 |
| `--load-debug-rollout-data` | 强制 debug train-only，并复用 dump | 切断在线生成随机性 |
| `--load-forge-rollout-data` | 重放 dump，但保留 SGLang、weight update 与 colocate 生命周期 | 测长上下文显存等系统行为 |
| `--check-weight-update-equal` | 同步主入口初始权重推送后请求 compare | 检查初始化同步；不是每轮自动证明 |

来源：`slime/utils/arguments.py` L1234-L1310、L1531-L1590；`slime/ray/rollout.py` L543-L577；`train.py` L23-L32。

## 5. Fault tolerance：恢复点有限，先画失败状态

RolloutManager 只在启用 health monitor 时管理 engine 健康；CI fault injection 在指定 rollout 后模拟 engine crash，并等待 health monitor 检出。训练 actor 到 `update_weights` 时，rank 0 调用 `recover_updatable_engines`，随后以 gloo barrier 对齐，再重取 engine 布局和必要时重连。来源：`slime/ray/rollout.py` L462-L496、L543-L554；`slime/backends/megatron_utils/actor.py` L580-L625。

这证明的是“rollout engine 在权重更新边界可恢复”，不代表：

- DataSource、外部 RM、rollout_buffer 或用户 hook 自动事务回滚；
- 已经发出的请求不会重复；
- 半次磁盘更新会被统一 rollback；
- 恢复后的样本天然带正确 weight version。

故障测试至少记录四项：崩溃发生点、inflight 样本去向、恢复后的 engine/version、是否出现重复或丢失。

## 6. CI：测试层级和触发条件

官方 CI 分为 always-on CPU correctness 和 label-gated GPU e2e。CPU job 使用 GitHub-hosted runner，不拿 GPU；GPU job 在 self-hosted Docker 中安装 editable package，通过 `gpu_lock_exec.py` 申请 GPU，再执行目标测试。changed-test job 从顶层 `NUM_GPUS` 构造矩阵，缺失时默认 8，因此 CPU test 应显式写 `NUM_GPUS = 0`。来源：`docs/en/developer_guide/ci.md` L3-L56。

| 改动 | 最低测试层 | 何时升级 |
|------|------------|----------|
| argument、reward、Sample、hook contract | CPU unit/contract | 涉及真实 Megatron/SGLang 调用时升级 GPU |
| Agent adapter | 独立 CPU adapter job | 涉及真实 provider/模型服务时补集成环境 |
| engine topology、PD、async rollout | GPU SGLang/Megatron e2e | 涉及性能时再固定硬件与 workload |
| loss、parallel consistency | CPU shape/invariance + GPU precision | 新 kernel/collective 必须真 GPU |
| checkpoint/weight sync | helper unit + GPU checkpoint/e2e | 跨节点、故障恢复需要目标拓扑 |

CI 事实源是 `.github/workflows/pr-test.yml.j2`；修改后再运行生成脚本，不直接手改生成的 workflow。来源：`docs/en/developer_guide/ci.md` L138-L155。

## 7. Megatron teacher server 的观测位置

`megatron_server.py` 是专用 teacher logprob HTTP 服务，不是默认 RL 主循环。它有 request queue、DP worker、RPS/TPS stats、`/get_loads` 和磁盘热更新；详见 [[Slime-补充主题]]。服务的 `/healthz` 只证明 HTTP 进程响应，数值正确仍需 request/result 测试。

## 8. 当前环境可执行的 CPU 验证

从 `slime/` 目录运行：

```powershell
python -m pytest tests/utils/test_trace_utils.py `
  tests/utils/test_megatron_server_arguments.py -q
```

预期：trace metadata/无 PD lane 和 teacher-only 参数契约通过。它不启动 Ray teacher server，也不覆盖 GPU profiler、engine crash recovery 或 PPO e2e。

真实闭环参考 `tests/test_qwen3_4B_ppo.py`：它声明 `NUM_GPUS = 8`，会下载 Qwen3-4B 与数据、转换 checkpoint，并启动 colocated actor/critic + SGLang。没有满足资源前提时应完整阅读而不是冒充运行。

## 导航

- [[Slime-复杂度热点]]
- [[Slime-补充主题]]
- [[Slime-综合学习检查]]
