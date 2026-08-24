# Mini Agent v0.1 开发 PRD

> 语言：本文为中文产品文档；英文用户入口与当前实现边界分别见
> [README.md](../README.md) 和 [ARCHITECTURE.md](ARCHITECTURE.md)。
>
> 状态：Draft v3，面向 v0.1 RC 的目标与验收
>
> 更新时间：2026-08-24
>
> 产品定位：本地、单主 Agent、可观察、支持长程任务续接的实验型 Coding Agent

本文是产品目标、验收标准和明确标记的未来方向，不是实现清单。以代码、
`.mini-agent/config.example.toml`、测试和 CI 为可执行事实；当前实现的模块边界、
持久化不变量和信任边界以 [ARCHITECTURE.md](ARCHITECTURE.md) 为准。本文中标记为
“未来”的能力尚未纳入 v0.1 RC，不能据此宣称已经可用。

## 1. 背景

Coding Agent 的模型本身是无状态的。每次模型调用能否延续上一轮工作，取决于 runtime 如何保存历史、组织工具结果、重建输入并恢复任务状态。

短任务可以直接携带完整历史；长任务会出现四类问题：

- 工具输出、代码和日志最终填满上下文窗口；
- 即使尚未填满，长输入也会稀释早期约束和当前意图；
- 临近窗口上限时才做一次摘要，提取质量差且摘要自身可能溢出；
- 如果只保留摘要而无法回查原始轨迹，遗漏的信息无法恢复。

Mini Agent 不复制 Claude Code、Codex、Cursor、OpenCode 或 MiMo Code 的完整产品。它实现一个足够小、容易阅读和修改的本地 Agent，用于：

- 理解 Coding Agent 的基本 runtime；
- 实验 checkpoint、rebuild、retrieval 和 fallback 策略；
- 保存可复现、可审计的完整执行轨迹；
- 为后续 Agent 数据分析、评测和训练提供实验平台。

## 2. 产品目标

v0.1 交付一个本地 CLI Coding Agent。用户在指定 workspace 中启动后，可以持续对话，让 Agent 搜索、读取和修改文件并执行命令。单个物理上下文窗口始终有界，但同一逻辑 session 可以经过多个 cycle 延续。

核心目标：

1. **能工作**：完成基本代码理解、修改、命令执行和验证。
2. **能续接**：提前生成增量 checkpoint，在临界点 rebuild 新窗口。
3. **能回查**：摘要遗漏后，Agent 可以检索原始 transcript 和大型 artifact。
4. **可观察**：完整保存消息、工具调用、审批、token 估算、checkpoint 和 rebuild。
5. **可恢复**：程序崩溃、模型 overflow 或 checkpoint 失败后，不破坏 session，也不盲目重放副作用。
6. **容易实验**：模型、预算、checkpoint、rebuild 和 fallback 策略可替换；通用离线
   replay 平台属于未来方向。

## 3. v0.1 范围

### 3.1 v0.1 目标范围

- 单个前台主 Agent；
- 一个受限的内部 checkpoint writer；
- OpenAI-compatible provider adapter；
- 本地 CLI 与 session resume；
- 文件读取、搜索、patch 和命令执行；
- append-only transcript 与大型输出 artifact；
- history/artifact 检索工具；
- checkpoint、rebuild、工具结果外部化和 emergency fallback；
- goal、acceptance criteria、任务状态和完成证据；
- 权限审批、workspace 边界和凭证脱敏；
- 长上下文 fixture、故障注入和确定性回归评测；通用策略 replay 标记为未来。

内部 checkpoint writer 不是通用 subagent。它不能调用 workspace 工具或修改代码，只能读取固定范围的 session 事件，并向 checkpoint store 提交结构化状态。

### 3.2 非目标

v0.1 暂不支持：

- 用户可调用的 subagent、agent team 或大规模并行编排；
- Max Mode、多候选采样与独立 judge 选优；
- 独立 Goal judge；
- Dynamic Workflow；
- GUI、IDE 插件或 Web Dashboard；
- MCP、Skills 或插件市场；
- 云端 session 同步；
- 向量数据库或全仓库语义索引；
- 项目/全局记忆的自动提炼；
- Dream、Distill 或自动生成 skill；
- 自动 commit、push 或 pull request；
- 浏览器自动化和外部 SaaS 集成；
- RL 训练或在线数据回流；
- 递归分块摘要；
- 完整容器或操作系统级沙箱。

这些能力只有在单 Agent loop、event store 和 context lifecycle 稳定后再评估。

## 4. 核心术语

| 术语 | 定义 |
|---|---|
| Logical session | 用户感知的一次连续任务，可跨越多个上下文窗口 |
| Cycle | 从一次 rebuild 后开始，到下一次 rebuild 前结束的一段物理窗口 |
| Durable transcript | 完整、脱敏、只追加、可恢复的原始事件记录 |
| Active context | 下一次真正发送给模型的派生工作视图 |
| Checkpoint | 从固定事件范围增量提取的结构化 session 状态 |
| Rebuild | 开启新 cycle，并从 checkpoint、最近原文和稳定指令重建 active context |
| Prune | 未来可选的独立 active-context 压缩策略；当前 RC 不实现 |
| Artifact | 外部化保存的大型、已脱敏工具输出 |
| Watermark | 某次 checkpoint 已完整处理到的最后一个 event ID |

## 5. 目标用户与典型场景

### 5.1 目标用户

- 研究 Agent harness、上下文管理和长程任务的开发者；
- 希望使用本地、透明 Coding Agent 的个人用户；
- 希望获取结构化 Agent 轨迹用于分析或训练的研究人员。

### 5.2 核心场景

#### 场景 A：完成小型代码修改

Agent 搜索代码、读取文件、应用 patch、运行测试，并用测试结果作为完成证据。

#### 场景 B：执行跨 cycle 的调试任务

Agent 在 context 使用率约 20%、45%、70% 时增量写 checkpoint；接近工作上限时 rebuild。新窗口恢复目标、任务状态、失败路径、文件和下一步后继续执行。

#### 场景 C：回查压缩前细节

Agent 发现 checkpoint 中缺少某条错误或参数，通过 history search 定位原始事件，再读取对应范围，而不是猜测。

#### 场景 D：恢复中断任务

用户退出或进程崩溃后恢复 session。runtime 识别未完成工具调用，不自动重放非幂等操作，并依据 workspace 实际状态继续。

#### 场景 E：比较上下文策略（未来）

未来的离线 replay 平台可对同一固定轨迹运行 `full-history`、`rolling-summary`、
`checkpoint-rebuild` 和 `deterministic` 策略，比较任务成功率、信息保留、token、延迟和成本。
当前 RC 仅提供 versioned fixture 的 `evaluate` 与确定性 `benchmark`，不提供策略 replay CLI。

## 6. 产品原则

### 6.1 原始事实与工作视图分离

- Durable transcript 是事实来源，只追加，不被 compaction 覆盖。
- Active context 是可丢弃、可重新构建的派生视图。
- Checkpoint 必须记录 source event 范围，不能成为无来源的自由文本。

### 6.2 提前增量提取，不在临界点孤注一掷

- 在上下文能力尚未明显退化时生成 checkpoint；
- 后续 checkpoint 只处理上一 watermark 之后的新事件；
- 临界点主要执行 rebuild，而不是首次总结全部历史。

### 6.3 先减量，再提取，最后重建

默认顺序：

1. 工具产生大输出时立即外部化；
2. 固定里程碑增量更新 checkpoint；
3. 接近上限时 rebuild，并从完整 transcript 选择可放入预算的原子消息组；
4. provider overflow 时走 deterministic recovery。

### 6.4 最近原文与用户意图优先

用户最近请求、当前 goal、pending tool chain、精确错误和未验证状态优先保留。旧 assistant reasoning 和大型工具正文优先外部化或压缩。

### 6.5 结构化状态不等于完整真相

Checkpoint 帮助继续行动，history 负责兜底。Agent 不确定时必须检索原始记录，不能把摘要补写成事实。

### 6.6 自动行为可检查

工具执行、审批、token 估算、checkpoint 输入范围、状态变更、rebuild 注入和 fallback 都写入事件记录。

### 6.7 副作用最多执行一次

程序崩溃后，任何状态不明的写文件或命令都不得自动重放。恢复流程先检查现实状态，再决定继续、重试或请求用户确认。

### 6.8 安全默认，授权范围精确

普通 workspace 读取工具自动执行；越界路径直接拒绝，不进入审批。`apply_patch` 可获得同一
规范化相对文件路径的 session 授权；`exec_command` 每次执行均需显式审批，不存在命令前缀、
通配符或其他持久命令授权。

## 7. 用户体验

### 7.1 CLI 入口

```bash
mini-agent
mini-agent chat --workspace /path/to/repo
mini-agent resume <session-id>
mini-agent sessions
mini-agent inspect <session-id>
mini-agent init-config --workspace /path/to/repo
mini-agent evaluate <session-id> --fixture path/to/fixture.json
mini-agent benchmark
```

### 7.2 会话内命令

```text
/system                   查看发送给 provider 的 system prompt
/context                  查看 active context、cycle 和 token 预算
/checkpoint               立即生成增量 checkpoint，但不 rebuild
/compact [focus]          生成 checkpoint 并立即 rebuild
/goal                     查看 goal；其余 goal 子命令按 CLI 提示输入
/resume [session-id]      切换到同 workspace、同 model 的可恢复 session
/exit                     保存一致状态后退出
```

`history_search`、`history_read` 和 `artifact_read` 是提供给 Agent 的工具，不是用户 slash
命令。`/help`、`/status`、`/clear` 和 `mini-agent replay` 均不是当前 RC 接口。

### 7.3 审批交互

```text
Agent 准备执行：pytest tests/auth -q

[1] 允许一次
[2] 拒绝并反馈原因
```

`exec_command` 不提供 session 授权。仅 `apply_patch` 可以对同一规范化相对文件路径授予
session scope；不得为广泛、递归、删除、权限修改或网络行为建立宽泛持久授权；workspace
外路径直接拒绝。

## 8. 总体架构

```text
                     ┌──────────────────────────┐
User ──> Main Agent ─┤ Context Builder / Budget ├──> Model Provider
             │       └──────────────────────────┘          │
             │                                             │ tool calls
             ▼                                             ▼
       Append-only Event Store <──── Tool Runtime / Permissions
             │
             ├──> Artifact Store
             ├──> History Search
             └──> Checkpoint Writer ──> Checkpoint Store
                                          │
                                          └──> Rebuild next cycle
```

核心状态机：

```text
NORMAL
  ├─ checkpoint milestone ─> CHECKPOINT_WRITING ─> CHECKPOINT_COMMITTED
  ├─ rebuild threshold ────> CHECKPOINT_WRITING ─> REBUILD ─> NEW_CYCLE
  ├─ provider overflow ────> DETERMINISTIC_RECOVERY ─> REBUILD
  └─ call cap reached ─────> DURABLE FAILURE ─> RETURN CONTROL
```

## 9. 功能需求

### 9.1 Main Agent Loop

每个模型步骤按以下顺序执行：

1. 接收用户输入，创建或更新 goal，并追加事件；
2. 恢复或构建 active context；
3. 估算完整请求 token，包括 system、tools、messages 和输出预留；
4. 处理到期 checkpoint 或 rebuild，并在 rebuild 时选择有界的消息组；
5. 记录 `model_request_started`；
6. 调用模型并展示返回文本；v0.1 RC 使用完整响应接口，不承诺流式传输；
7. 持久化完整 assistant message 或失败状态；
8. 解析 canonical tool calls；
9. 校验权限，持久化执行生命周期并运行工具；
10. 判断继续、完成、阻塞、等待用户或预算耗尽；
11. 在安全点协调 checkpoint writer；当前 RC 等待写入终态后再继续。

每个用户 turn 有可配置的硬 Provider 调用上限 `max_model_calls_per_turn`，默认 50。
主模型、自动 checkpoint model、重试和 overflow recovery 共用该预算；自动 checkpoint
必须为后续主调用保留一次额度，不足时回退到确定性 extractor。达到上限时抛出明确错误，
已持久化事件和 active Goal 保持可恢复，且不得把“达到上限”解释为任务已完成。用户可在
下一 turn 继续。交互式“请求扩展 soft budget”、费用硬上限和 wall-clock 硬上限不属于当前
v0.1 RC 实现。

Logical session 不设固定 cycle 数量上限。

### 9.2 Model Provider

v0.1 实现一个 OpenAI-compatible adapter，同时以以下窄接口与具体 SDK 解耦：

```text
ModelProvider
- complete(request) -> ModelResponse

CapabilityProvider（可选）
- capabilities: ProviderCapabilities

ProviderCapabilities
- context_window
- max_input_tokens (optional)
- max_output_tokens
- native_tool_calling
- usage_reporting
- reasoning_support
```

Provider message 和 tool call 在进入 event store 前转换成内部 canonical schema。第一版使用 provider 原生 function calling；JSON、XML 或受限命令行工具语法作为后续可替换 adapter，不写死在 Agent loop 中。

当前 RC 请求为完整响应，不承诺 streaming、provider 内部错误分类接口或按 model 查询
capabilities 的方法；未声明 capability 时，runtime 使用配置中的保守限制。

支持配置：

- base URL；
- model ID；
- context/window override；
- maximum output tokens；
- temperature；
- timeout 和 retry；
- 独立 checkpoint model，可选。

`reasoning_support` 是为 capability 识别保留的字段；当前 adapter 明确报告不支持，
v0.1 RC 不发送 provider-specific reasoning 参数。等第二个实际 Provider/model 需要并完成
兼容性测试后，再把 reasoning 请求配置作为未来能力加入，而不是提前固化未验证的协议。

凭证只能从配置指定的环境变量读取；操作系统或外部凭证系统如需参与，应在进程外把值注入该环境变量。凭证不得进入配置、日志、transcript、artifact、异常文本或测试 fixture。

### 9.3 Workspace 工具

#### `read_file`

- 输入相对 workspace 的路径、可选起止行；
- 输出带行号文本；
- 大文件必须分段读取；
- canonicalize 后验证路径边界；
- 默认不跟随指向 workspace 外的 symlink。

#### `search`

- 输入 query、可选 path 和 glob；
- 优先使用 `rg`；
- 限制匹配数量和直接返回字符数；
- 完整超长结果写入 artifact。

#### `apply_patch`

- 输入结构化 patch；
- 修改前验证所有目标位于 workspace；
- 展示修改范围并进入审批策略；
- 保存 patch、修改文件与结果；
- 不接受绝对路径或越界 symlink。

#### `exec_command`

- 输入为 `argv: list[str]`，不经过 shell 展开；不支持 shell 字符串模式；
- cwd 必须位于 workspace；
- 捕获 stdout、stderr、exit code 和执行耗时；
- 只支持单次、带上限的同步执行；无交互式轮询或用户取消接口，超时由 runtime 终止进程组；
- 超长输出写入 artifact；
- 每次执行均须审批；不支持 session 命令前缀授权。删除、覆盖、安装、网络和权限修改
  不会因曾经批准过类似命令而自动放行。

### 9.4 恢复与检索工具

#### `history_search`

- 输入 query、limit、可选 event type 与 before/after event ID；
- v0.1 对脱敏后的 JSONL 做线性或 `rg` 搜索；
- 返回 event ID、时间、角色、摘要片段，不返回无界正文。

#### `history_read`

- 输入 from/to event ID 或以某 event 为中心的窗口；
- 只读取当前 session；
- 结果受 token/字符预算限制；
- 大范围读取必须分段。

#### `artifact_read`

- 输入 artifact ID、offset 和 limit；
- 只能读取当前 session 已登记 artifact；
- 返回内容仍需经过输出预算与脱敏检查。

这些工具默认只读且可自动执行。它们是压缩遗漏后的恢复通道，属于 v0.1 核心能力。

### 9.5 Goal 与完成状态

每个活跃任务包含：

```text
Goal
- objective
- acceptance_criteria[]
- status: active | completed | blocked | cancelled
- completion_evidence[]
- blocked_reason (optional)
```

用户可用 `/goal` 明确设置；未设置时，runtime 从首条用户请求生成可编辑的 objective，不能擅自增加高风险停止条件。

主 Agent 请求结束任务时必须声明状态：

- `completed`：逐条给出 acceptance criteria 的证据；
- `blocked`：说明阻塞、已尝试路径和需要的外部条件；
- `active`：任务仍可继续，不能输出完成措辞；
- `cancelled`：仅由用户取消或明确策略触发。

v0.1 Completion Gate 做结构和机械证据检查，例如测试命令是否实际执行、exit code 是否存在、目标文件是否修改。缺少证据时向 Agent 返回 gap 并继续。独立 judge model 后置。

## 10. Session 与持久化

### 10.1 默认存储位置

当前 RC 的运行数据默认放在操作系统用户数据目录，而不是目标代码仓库：

```text
<user-data-dir>/mini-agent/
└── sessions/
    └── <session-id>/
        ├── metadata.json
        ├── events.jsonl
        ├── checkpoints/
        │   ├── index.json
        │   └── checkpoint-<checkpoint-id>.json
        └── artifacts/
            ├── index.json
            └── <artifact-id>.txt
```

实现使用 `platformdirs` 解析路径。具体位置由操作系统和 `platformdirs` 决定，不将某个
绝对目录作为跨平台接口承诺。

默认情况下，目标仓库内只会出现可选的 `.mini-agent/config.toml` 和用户明确选择的项目记忆文件。运行 transcript、artifact 和 checkpoint 默认不污染 workspace，也不进入代码搜索结果。用户显式设置的 `runtime.data_dir` 可以位于别处，也可以是相对配置文件解析的路径；无论位置如何，该 runtime 数据根目录都会作为 workspace 文件工具的排除根，普通 read/search/apply_patch 不能通过它们的路径参数访问该目录。`exec_command` 获批后拥有当前用户进程的操作系统权限，不受这项文件工具排除规则约束。

### 10.2 Event Store

`events.jsonl` 是 append-only source of truth：

- 每条事件一行，带 schema version；
- 同一 session 只有一个 append writer；
- 关键副作用边界写入后 flush；
- event ID 单调递增且稳定；
- 不修改历史事件，用补偿事件表达状态变化；
- 尾部损坏时保留原文件并生成 recovery report，不能静默丢弃有副作用的事件。

### 10.3 Session 恢复

恢复时：

1. 校验 metadata、event 顺序，以及已登记 checkpoint 和 artifact 的不可变内容；
2. 找到最后一个 committed checkpoint；
3. 扫描未结束的 model/tool/checkpoint 生命周期；
4. 将未结束的只读工具标记为 interrupted，可安全重试；
5. 将可能产生副作用的工具标记为 unknown；
6. 检查 workspace 实际状态并向用户/Agent提供恢复说明；
7. 从 checkpoint、rebuild 事件和 transcript 重建内存中的 active context；
8. 开启新的 resume turn，绝不自动重放 unknown 操作。

当前 cycle、projection version、已跨过的 milestone、可见 checkpoint 和 pending tool 等
active-state 事实由 session 事件与已登记的不可变记录派生到内存中；v0.1 RC 不持久化、
也不承诺一个名为 `active-state.json` 的缓存 schema。

## 11. Artifact 管理

工具结果超过阈值时：

- 完整的脱敏版本写入 `artifacts/`；
- active context 只保留来源、大小、head、tail、内容哈希和 artifact ID；
- transcript 记录 artifact 元数据，不复制完整正文；
- Agent 通过 `artifact_read` 分段回查；
- artifact 写入失败时工具结果不能伪装为已完整保存；
- artifact retention 与 session 一致，v0.1 默认不自动删除。

不得持久化未脱敏的原始输出。内容哈希基于脱敏后的持久化内容计算。

## 12. Context Lifecycle

### 12.1 Working Context Budget

首先解析当前 provider route 的物理限制：

```text
working_context_limit = min(
  provider_context_window,
  configured_max_context_if_present
)

desired_output_tokens = min(
  configured_max_output_tokens,
  provider_max_output_tokens
)

output_reserve = max(
  configured_reserve_tokens,
  desired_output_tokens
)

request_input_limit = min(
  provider_max_input_if_present,
  working_context_limit - output_reserve
)
```

规则：

- `configured_max_context` 只能降低工作窗口，不能超过 provider capability；
- `output_reserve` 至少覆盖计划输出上限；若剩余输入空间低于系统最小要求，降低本轮输出上限或拒绝无效配置，不能产生负预算；
- token 估算覆盖 system、tools、messages、附件和 protocol overhead；
- 当前 RC 使用保守、与 tokenizer 无关的估算，不依赖 provider usage；provider usage 的采集、估算误差记录和滚动校准属于未来能力；
- 如果 capability 未知，使用保守默认，并允许 provider overflow 进入恢复流程。

### 12.2 默认生命周期参数

| 参数 | 默认值 |
|---|---:|
| Checkpoint milestones | 20%、45%、70% |
| Rebuild threshold | 85% |
| Output reserve | 16,384 tokens，按模型 clamp |
| Rebuild seed budget | `min(65,536, request_input_limit × 50%)` |
| Checkpoint model output | 最多 4,096 tokens |
| Checkpoint durable JSON | 最多 65,536 bytes |
| Checkpoint 内单个 tool result | 最多 2,000 chars |
| 单次工具直接返回 | 最多 12,000 chars |

milestone 以当前 cycle 的 active request utilization 计算，每个 milestone 每个 cycle 最多触发一次。Rebuild 后，低于新窗口初始占用率的 milestone 直接标记为已跨过，避免立即重复触发。

所有比例和绝对值可按模型覆盖。

### 12.3 Checkpoint Writer

Checkpoint writer 是独立的内部模型调用：

- runtime 在 milestone 捕获不可变 event watermark；
- writer 输入为上一 checkpoint + watermark 之后的新事件；
- 手动 `/compact [focus]` 的 focus 作为本次提取重点，但不能覆盖 goal、安全规则或事实状态；
- 当前 RC 在安全点等待 writer 完成，不与 writer 共享 active context；后台并发写入留作后续优化；
- writer 无 tools，不可修改 workspace；
- 同一 session 只有一个 writer 可以提交；
- 多个等待中的 milestone 合并到最新 watermark，避免重复工作；
- 输出通过 schema、大小和敏感信息检查后，先写临时文件，再原子提交；
- writer 失败时保留上一 committed checkpoint，并记录失败事件。

### 12.4 Checkpoint Schema

```text
Checkpoint
- schema_version
- checkpoint_id
- session_id
- cycle_id
- version
- source_from_event_id
- source_through_event_id
- current_intent
- acceptance_criteria[]
- constraints_and_preferences[]
- task_tree[]
- completed[]
- active_work[]
- blocked[]
- next_actions[]
- relevant_files[]
- cross_task_findings[]
- errors_and_fixes[]
- runtime_state[]
- key_decisions[]
- artifact_references[]
- miscellaneous_notes[]
- writer_model
- created_at
```

语义约束：

- 不得把计划、猜测或失败尝试写成 completed；
- 保留精确路径、symbol、命令、数字、错误和验证状态；
- 新决定推翻旧决定时，更新当前状态并保留简短 decision trace；
- 不确定事实标记为 `unknown` 或 `needs_verification`；
- 不复制凭证或疑似敏感值；
- 每条高价值状态尽可能携带 source event IDs。

Markdown 只作为人类查看的派生渲染；JSON checkpoint 是 canonical state。

### 12.5 工具结果外部化与 rebuild 选择

当前 RC 不实现独立、平时运行的 active-context prune。工具结果超过阈值时立即外部化，
transcript 仅保留 artifact 元数据；原文可由 `artifact_read` 回查。

rebuild 时，runtime 从完整 transcript 按原子 tool turn 选择能放入预算的最近消息组：

- tool call 与对应 result 始终成组处理；
- 最近真实用户请求必选，过大时作有界截断并提供 history 回查提示；
- 省略的旧组不改写 durable transcript；
- 有组被省略时记录 `context_pruned`，供恢复和评测观察。

独立的、每次请求前执行的 prune 策略属于未来方向；引入前必须定义其 projection、恢复和
事件语义，且不能破坏原子 tool turn。

### 12.6 Rebuild

达到 rebuild threshold 或用户执行 `/compact` 时：

1. 停止发起新的模型请求；
2. 对当前 durable watermark 同步生成并提交下一个 checkpoint；
3. checkpoint 成功后，生成新的 active context projection；
4. 记录完整注入清单、截断情况与 token 估算；
5. `cycle_id += 1`，继续当前 logical session。

checkpoint 写入、提交或 activation 出现失败/不确定性时，本次 rebuild 显式中止，不切换
projection 或 cycle；最近已提交 checkpoint 仍是可见状态。若提交结果可能已落盘但本进程
未能确认 activation，当前进程停止进一步 mutation，必须 resume 后重新验证 durable records。
v0.1 RC 不用“deterministic delta”掩盖这类失败。

### 12.7 Rebuild 注入顺序

当前 RC 将 system/safety instructions 和 tool schemas 单独计入请求预算，并以可验证的固定顺序重建：已提交 checkpoint、最新用户消息，以及按原子 tool turn（tool call 与对应 result 成组）倒序选择的最近事实。每一项在加入前按完整请求估算，超过预算时不拆开 tool call/result pair，而是停止选择更旧的原子组；最新用户消息过大时保留有界文本并给出 history 回查提示。

下列内容不可被完全删除：

- safety/permission instructions；
- 当前 goal；
- 最新真实用户请求；
- 未结束 tool call/result pair；
- checkpoint watermark 与 history retrieval 指引。

当前实现不会为 goal、task tree、workspace instructions、history/artifact index 等段落分配固定百分比注入预算；这些内容由 checkpoint 或稳定 system context 承载。未来若引入分段配额或跨段借用策略，必须先以完整请求预算、原子 tool turn 和截断可观察性为验收前提，不能改变上述当前语义。

### 12.8 History Retrieval

Rebuild 后 system context 明确告诉 Agent：

- checkpoint 是有损工作状态；
- 缺少精确细节时使用 `history_search`；
- 大输出通过 `artifact_read` 回查；
- 不得凭摘要猜测未记录事实。

Checkpoint 中的 source event ID 和 artifact ID 是检索入口。

### 12.9 Emergency Recovery

以下情况进入 deterministic recovery：

- provider 返回 context overflow；
- checkpoint request 自身 overflow；
- writer 输出为空、schema 非法、未缩小或连续失败；
- 单条请求/payload 过大；
- rebuild seed 超过预算且无法正常裁剪。

Deterministic extractor 不调用模型或 workspace 工具。它从上一 committed checkpoint 与本次
事件构造有界状态：最新 user message 更新当前 intent 和 active work；Goal 事件更新 intent
与 acceptance criteria；goal status、tool terminal 状态、assistant message 和 artifact 事件
分别补入运行状态、错误、notes 或 artifact references。每个保留列表只取最近有限项，单项
文本也有上限。

持久化前按 UTF-8 byte budget 拟合：先丢弃低优先级列表中最旧的项，再缩短最大的文本，
必要时才继续丢弃其余列表项；缩短不会切断多字节字符。结果必须通过 typed schema 且不超过
配置的 durable byte 上限，否则本次 checkpoint 明确失败。它不是完整语义摘要，也不承诺
保留固定字段清单；需要精确细节时必须使用 history/artifact retrieval。

## 13. 数据模型

### 13.1 Event

```text
Event
- schema_version
- id
- session_id
- cycle_id
- turn_id
- timestamp
- type
- payload
- parent_event_id (optional)
- correlation_id (optional)
```

首批事件类型：

```text
session_started / session_resumed / session_stopped
goal_created / goal_updated / goal_status_changed
user_message
model_request_started / assistant_message / model_request_failed
approval_requested / approval_resolved
tool_requested / tool_started / tool_completed / tool_failed / tool_interrupted
artifact_created
context_estimated / context_pruned
checkpoint_started / checkpoint_committed / checkpoint_failed
rebuild_started / rebuild_completed / rebuild_failed
```

所有 tool lifecycle 事件共享稳定 `correlation_id`。恢复时不得根据缺失的 completed event 推断工具“没有执行”。
恢复结果通过 `session_resumed` 以及对中断 lifecycle 补写的稳定终态表达；当前 RC 不定义
独立的 recovery start/completed 事件。

### 13.2 派生活跃状态

v0.1 RC 的 active state 不是独立持久化模型。session metadata、append-only events、已登记
checkpoint/artifact 与最近的 rebuild 事件共同构成事实来源；runtime 在启动、resume 和每次
请求准备时派生当前 cycle、goal、checkpoint/watermark、pending tool、projection version、
预算估算和已跨过 milestone。该内存视图可丢弃并重建，不承诺额外文件、原子更新协议或
稳定 JSON schema。

### 13.3 Artifact

```text
Artifact
- artifact_id
- session_id
- source_event_id
- media_type
- redacted
- chars
- content_hash
- relative_storage_path
- created_at
```

所有数据结构带 schema version，后续通过显式 migration 升级。

## 14. 权限与安全

### 14.1 Workspace 边界

- read/search/apply_patch 的路径参数 canonicalize 后验证位于 workspace；
- 默认不跟随越界 symlink；
- 命令 cwd 位于 workspace；
- workspace 文件工具拒绝 workspace 外的路径；
- session store 对普通 workspace 文件工具不可见，需通过专用 history/artifact API 读取；
  获批的 `exec_command` 仍可使用当前用户的操作系统权限访问其它路径。

v0.1 的路径边界防止普通绝对路径、逃出 workspace 的父目录遍历、运行时目录访问和静态 symlink
逃逸；不把 workspace 视为可被恶意并发进程改写的安全边界。严格防御
rename/symlink/hard-link race 需要后续 dirfd 或操作系统隔离后端。`exec_command`
是逐次审批的本地执行能力，不是操作系统沙箱。

### 14.2 操作分级

| 操作 | 默认行为 |
|---|---|
| workspace 读取、搜索 | 自动允许 |
| history/artifact 读取 | 自动允许 |
| 修改普通文件 | 询问；仅 `apply_patch` 可授予同一规范化相对文件路径的 session scope |
| 测试、lint、只读 Git | 通过 `exec_command` 询问；每次调用逐次确认 |
| 安装依赖、网络访问 | 询问 |
| shell 模式 | v0.1 不支持 |
| 删除、覆盖、Git reset、权限修改 | 强制询问 |
| workspace 外文件路径访问 | 直接拒绝 |

### 14.3 凭证安全

- 不得在任何输出或持久化内容中泄露或硬编码 API key、Token、密码、私钥、Cookie、连接字符串或其他凭证；
- 配置只保存环境变量名，不保存凭证值；
- tool output、assistant content、checkpoint、artifact 和日志在持久化前执行脱敏；
- 请求 header 和完整环境变量永不写日志；
- history search 只检索脱敏后的持久化数据；
- 测试只能使用明确标记且无法用于真实服务的假凭证。

脱敏命中必须记录类型和数量，但不能记录原值。

## 15. 配置

当前 RC 的配置查找顺序为：显式 `--config`、workspace 的
`.mini-agent/config.toml`、platformdirs 用户配置目录中的 `config.toml`。CLI 参数优先级最高。

完整、可执行且随代码测试的字段清单位于
`.mini-agent/config.example.toml`，可通过 `mini-agent init-config --workspace ...`
从源码或 wheel 安装。PRD 不复制一份容易漂移的参数表。当前配置分为 `model`、`context`、
`runtime` 和 `system_prompt`；权限策略由 v0.1 runtime 固定执行，不存在旧设计中的
`permissions.auto_approve_*` 配置。`checkpoint_max_tokens` 控制 checkpoint provider 输出，
`checkpoint_max_bytes` 独立限制持久化 JSON；`max_model_calls_per_turn` 是硬调用上限。

示例中的模型和 URL 是占位符。真实凭证由配置指定的环境变量提供；外部凭证系统应在进程外注入该变量。

配置加载时必须校验：

- milestone 单调递增且小于 rebuild ratio；
- rebuild ratio 小于 1；
- max context 不超过 provider capability；
- reserve 和 seed budget 对小窗口模型完成 clamp；
- checkpoint token 上限与持久化 byte 上限分别校验；
- 未知字段默认报错，防止拼写错误被静默忽略。

## 16. 可观察性

当前 RC 的 `/context` 输出紧凑的运行快照：projection strategy/version、保守估算的总 token 与输入上限、利用率、消息数、checkpoint ID 和 watermark。它不是完整遥测面板。

当前每次模型请求的持久化记录只覆盖运行恢复和审计所需的生命周期事实；不以 provider usage、cache token、费用或完整延迟分类作为稳定接口。provider usage 采集、估算误差校准，以及按 system/tools/messages/checkpoint/recent tail 分项展示、writer/retrieval 计数和费用/延迟明细，均为未来可观察性能力；实现时不得记录 header、凭证或未脱敏环境变量。

## 17. 错误与崩溃处理

必须区分：

- context/token overflow；
- HTTP payload too large；
- rate limit；
- provider timeout；
- checkpoint validation failure；
- tool timeout/cancel；
- tool permission denied；
- malformed tool call；
- event store/session corruption；
- user cancellation。

原则：

- 只有分类为 retryable 的错误自动重试，默认最多两次；
- overflow 进入 recovery，不盲目重试原请求；
- malformed tool call 返回结构化错误给模型；
- `tool_started` 必须在副作用发生前持久化；
- `tool_completed` 只有拿到真实结果后写入；
- started 但未结束的副作用操作在恢复时标记 unknown；
- unknown 操作不得自动 replay；
- checkpoint 写入失败不推进 watermark；
- rebuild 失败不切换 active projection；
- event store 无法持久化时停止产生新副作用；
- 用户中断保留已完成工具结果和一致 pending state。

## 18. 测试与评测

### 18.1 单元测试

- path canonicalization 和 symlink 边界；
- event append、flush、尾部损坏检测；
- session event、checkpoint/artifact index 的校验，以及 active context 重建；
- tool lifecycle 与 unknown 恢复；
- canonical provider/tool schema；
- request token budget 和 estimator 校准；
- checkpoint 增量范围与 single-writer；
- checkpoint schema 和 watermark；
- rebuild 选择时的 tool pair 完整性；
- rebuild 注入优先级与预算；
- deterministic recovery；
- history/artifact 分段读取；
- goal completion evidence；
- 凭证脱敏。

### 18.2 集成测试

- 读文件 → 修改 → 测试 → completion evidence 完整 loop；
- 至少三个 cycle 的连续 rebuild；
- checkpoint writer 失败后使用上一版本继续；
- provider overflow 后 deterministic recovery；
- 进程在 tool_started 后崩溃，恢复时不自动重放；
- 大型 shell output 外部化并由 Agent 回查；
- checkpoint 遗漏细节后通过 history 找回；
- 拒绝危险操作后 Agent 仍可继续；
- session 退出和恢复。

未来：checkpoint writer 若改为与主 Agent 并行，必须新增 watermark 一致性、取消和恢复语义的集成测试。当前 RC 在安全点同步等待 writer 终态后再继续，不以并发写入作为策略或验收条件。

### 18.3 长上下文固定轨迹

每条 fixture 预先标注 information atoms：

- critical constraints；
- exact paths/symbols/commands/errors；
- completed/active/blocked 状态；
- superseded decisions；
- artifact/history retrieval targets；
- acceptance criteria 和 completion evidence。

至少覆盖：

1. 首轮硬约束经过三个 cycle 后仍被遵守；
2. 中途推翻旧方案，checkpoint 不保留过时决策；
3. 单个大工具输出不挤占 active context；
4. tool call/result 不在 rebuild 选择时拆开；
5. checkpoint 缺少精确细节时 Agent 主动检索 history；
6. writer、provider 和 payload 分别 overflow；
7. 崩溃恢复后非幂等命令不重复执行；
8. Agent 不因已有进展而提前宣告完成。

### 18.4 指标

| 指标 | 含义 |
|---|---|
| Critical constraint exact recall | 关键硬约束逐项保留率 |
| State precision/recall | completed、active、blocked 是否准确 |
| Stale fact rate | 已失效状态仍出现在 checkpoint 的比例 |
| Exact detail recall | 路径、symbol、错误和数字保留率 |
| Retrieval success | 缺失信息能否从 history/artifact 找回 |
| False completion rate | 未满足标准却宣告完成的比例 |
| End-task success | 最终任务是否完成 |
| Recovery success | overflow/crash 后是否继续且无重复副作用 |
| Cost/latency | 每 cycle token、费用、checkpoint 延迟和总耗时 |

摘要文本相似度不是主要指标。最终任务成功与状态正确性优先于摘要长度。

### 18.5 v0.1 RC 工程验收标准

- 在 deterministic fixtures 中完成至少三个 cycle；
- 标注的 critical constraints exact recall 为 100%；
- 状态字段 micro-F1 不低于 0.95；
- 固定 fixture 中 false completion 为 0；
- history/artifact 指定目标 retrieval success 为 100%；
- provider overflow、writer failure 和 tool crash 不损坏 session；
- unknown 副作用操作不被自动重放；
- workspace 外文件路径访问被阻止；
- transcript、checkpoint、artifact 和日志通过凭证扫描；
- 核心测试全部通过；

当前 `mini-agent benchmark` 的三个任务是隔离的 synthetic deterministic fixtures，
用于回归 runtime 的 read → patch → exec → completion gate → rebuild 路径。报告中的
`estimated_request_tokens` 是保守估算，latency 是本地全流程 wall-clock；provider usage、
费用和人工评价为 `unavailable`。它们不应被描述为真实模型质量证据。

### 18.6 v0.1 稳定发布验收标准

- 在至少三个匿名化的真实小型代码任务上报告成功率；
- 记录逐 cycle 的 provider token、provider/checkpoint latency 和总耗时；
- 使用预先定义的 rubric 完成人工评价；
- 报告携带候选版本/源码 provenance，但不得包含组织身份、内部路由或凭证。

真实模型任务受随机性影响，不以单次成功作为唯一发布门槛；deterministic replay 是回归测试基线。
以上真实任务证据当前尚未完成，因此工作树只能称为 v0.1 RC 候选，不能声称已满足稳定发布验收。

## 19. 当前工程结构与未来演进

下面是当前关键源码与测试分类的概览，不是要求目录必须与下图逐项一一对应的未来蓝图；完整文件归属以 [ARCHITECTURE.md](ARCHITECTURE.md) 和实际工作树为准。

```text
mini-agent/
├── pyproject.toml
├── README.md
├── .mini-agent/
│   └── config.example.toml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── INDEX.md
│   └── PRD.md
├── src/mini_agent/
│   ├── cli.py
│   ├── agent.py
│   ├── config.py
│   ├── context.py
│   ├── checkpoint.py
│   ├── checkpoint_model.py
│   ├── checkpoint_writer.py
│   ├── event_store.py
│   ├── session.py
│   ├── session_recovery.py
│   ├── history.py
│   ├── artifacts.py
│   ├── permissions.py
│   ├── redaction.py
│   ├── host_path_redaction.py
│   ├── providers/
│   │   └── openai_compatible.py
│   └── tools/
│       └── registry.py
└── tests/
    ├── providers/
    ├── release/
    ├── support/
    ├── tools/
    ├── test_agent.py
    ├── test_context.py
    ├── test_checkpoint.py
    ├── test_session.py
    └── test_session_durability.py
```

未来若模块数量继续增长，可在不改变依赖方向的前提下拆分为更细的领域目录；这种重组不是 v0.1 RC 的验收项。

## 20. 开发里程碑

### M1：Crash-safe Runtime

- Python + `uv` + Typer 工程；
- Provider interface 与 fake provider；
- canonical messages/tool calls；
- append-only event store；
- session create/list/resume；
- tool lifecycle 与 interrupted/unknown 恢复。

完成标准：不使用 workspace 工具也能连续对话、退出、模拟崩溃并恢复一致事件状态。

### M2：安全工具与可检索存储

- read/search/apply_patch/exec；
- workspace boundary；
- approval flow；`apply_patch` 的精确文件路径 session scope；
- artifact externalization；
- history_search/history_read/artifact_read；
- 凭证脱敏。

完成标准：能安全完成小型代码修改；大型输出可回查；非幂等工具崩溃后不自动重放。

### M3：Checkpoint 与 Rebuild

- 完整请求 token 预算；
- `/context`；
- early checkpoint milestones；
- single checkpoint writer；
- 原子轮次投影与可观察的 `context_pruned`；
- budgeted rebuild；
- `/checkpoint` 和 `/compact`；
- deterministic emergency recovery。

完成标准：固定长轨迹完成至少三个 cycle，遗漏细节可通过 history 找回。

当前实现说明：v0.1 默认使用受限、确定性的 checkpoint extractor；配置
`context.checkpoint_model` 后会优先进行一次无 tools、有限输入输出、强 schema 校验的
内部模型调用。该调用失败时回退到确定性 extractor，上一 committed checkpoint 不受影响。

### M4：Goal、评测与稳定性

- goal/acceptance criteria/status；
- Completion Gate；
- information-atom fixtures；
- provider/writer/tool 故障注入；
- retention、recovery、token、费用与延迟报告；
- 使用文档与示例。

完成标准：RC 满足 18.5；稳定发布还必须满足 18.6 并产出可复现的真实模型评测报告。

当前实现说明：Goal 作为 append-only events 持久化，首条用户请求在缺省时生成
objective；`/goal` 可设置 objective、typed acceptance criteria、blocked/active/cancelled
状态和 completion evidence。Completion Gate 只接受已存在 event ID，并机械检查
assistant response、tool completion、成功 command、文件修改或 artifact 创建。模型文本不能
直接改变 Goal 状态。

`mini-agent evaluate` 对 versioned information-atom fixture 计算 retention、state
precision/recall/F1、stale fact、retrieval、false completion、recovery 和 estimated token；
`mini-agent benchmark` 在三个隔离的临时代码 workspace 中复现 read → patch → exec →
Completion Gate → 三次 rebuild。无 provider usage、价格或人工输入时，报告必须写
`unavailable`，不得推测。该 deterministic benchmark 是回归基线，不替代真实模型质量或人工
评价。

## 21. 已确定决策与待确认项

### 21.1 PRD 已确定

- 技术栈：Python 3.12–3.14、`asyncio`、Pydantic、Typer、`pytest`、`uv`；
- 单前台 Agent，允许一个受限的内部 checkpoint writer；
- runtime 数据默认保存在用户数据目录，不写入目标仓库；
- durable transcript 使用 JSONL；
- checkpoint 使用 versioned JSON，Markdown 仅用于展示；
- history v0.1 使用本地线性/`rg` 搜索，不引入 SQLite；
- provider 和 tool protocol 使用 canonical internal schema；
- v0.1 不实现项目/全局记忆自动提炼。

### 21.2 已选 v0.1 保守策略与待确认项

1. 首个真实 provider/model；
2. 写文件：`apply_patch` 可授予同一规范化相对文件路径的 session scope；每次完整脱敏 patch 参数仍持久化供审计。路径不同或非法形状不复用授权；`replacement_content` 中的凭证形态与脱敏占位符在审批前直接拒绝，`expected_content` 仍可匹配并删除已有敏感内容；
3. 命令：v0.1 不设“安全命令前缀” allowlist，`exec_command` 一律逐次确认；凭证形态或已脱敏的 argv/cwd 在审批前直接拒绝；
4. 三个真实验收任务使用哪个目标仓库；
5. 是否设置默认费用上限，还是第一版只记录不拦截。

## 22. 后续候选方向

v0.1 稳定后再评估：

- 独立 Goal judge；
- checkpoint 专用模型路由；
- SQLite FTS 与更强 history retrieval；
- 项目记忆、全局记忆和显式 promotion；
- checkpoint 与 Git diff/checkpoint 关联；
- branch/fork session；
- MCP/Skills 按需发现；
- Max Mode；
- deterministic workflow runtime；
- Dream/Distill；
- context 策略离线 replay 平台；
- 把 self-summarization 纳入训练轨迹；
- 多 Agent 的隔离上下文与结果汇总。

这些方向不提前进入 v0.1 核心实现。

## 23. 设计依据

- [MiMo Code：将编程 Agent 扩展到长程任务](https://mimo.xiaomi.com/zh/blog/mimo-code-long-horizon)
- [MiMo Code 官方仓库](https://github.com/XiaomiMiMo/MiMo-Code)
- [Claude Code：Context window](https://code.claude.com/docs/en/context-window)
- [Codex：Compaction implementation](https://github.com/openai/codex/tree/main/codex-rs/core/src)
- [Pi：Compaction](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md)
- [Cline：Auto Compact](https://docs.cline.bot/features/auto-compact)
- [Cursor：Dynamic context discovery](https://cursor.com/blog/dynamic-context-discovery)
- [Cursor：Training Composer for longer horizons](https://cursor.com/blog/self-summarization)
- [OpenCode：Session compaction](https://github.com/anomalyco/opencode/blob/dev/packages/opencode/src/session/compaction.ts)
- [Kilo Code：Context Condensing](https://kilo.ai/docs/customize/context/context-condensing)
