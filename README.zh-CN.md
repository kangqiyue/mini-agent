# Mini Agent

[English](README.md) | 简体中文

Mini Agent 是一个小型、可观测的本地编程 Agent，用于探索长程任务的运行时设计。实现强调显式状态、仅追加事件，以及简单且类型明确的模块。

文档职责与阅读顺序见 [docs/INDEX.md](docs/INDEX.md)。当前实现的模块边界与不变量见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，产品目标与验收标准见
[docs/PRD.md](docs/PRD.md)。安全边界与漏洞报告方式见
[SECURITY.md](SECURITY.md)，开发约定见 [CONTRIBUTING.md](CONTRIBUTING.md)，
发布前检查见 [docs/RELEASING.md](docs/RELEASING.md)。本项目采用
[Apache License 2.0](LICENSE) 许可证。

## 当前里程碑

当前 v0.1 release candidate 已在有界、审批驱动的编程运行时之上实现 M4
目标与评测范围，但尚不是稳定版 v0.1：

- 统一且有明确类型的消息模型；
- 统一的 Assistant 工具调用和 `role="tool"` 工具结果；
- OpenAI 兼容的 Provider 边界；
- 仅追加的 JSONL 事件，以及截断尾部恢复；
- 单写入者的会话创建、恢复、检查和停止；
- 对中断或结果未知的工具调用进行恢复，且不自动重放；
- 有界的工作区文件读取和搜索；
- 有界目录列表，以及一次只修改或新建一个文件的 `apply_patch`；替换现有文件时校验预期内容；
- 使用显式 `argv`、不进行 shell 展开的 `exec_command`；
- 在任何写入或命令开始前持久化审批事件；
- 脱敏后的 Artifact，以及有界的会话历史检索；
- 保守的完整请求预算，并根据 Provider 能力限制预算；
- 单写入者、提交到 Transcript 的增量 Checkpoint；
- 跨逻辑周期的自动和手动上下文重建；
- 确定性的上下文溢出恢复，以及可选的模型驱动 Checkpoint 提取；
- 一个持久化且可编辑的 Goal，其验收标准始终保留在模型上下文中；
- 基于持久化 Assistant、工具和 Artifact 证据的机械式完成门禁；
- 带版本的信息原子 Fixture，以及确定性的会话评测；
- 包含三个任务的读取、修改、执行 Benchmark，并明确标记暂不可用的用量、成本和人工评测字段。

## 实现逻辑概览

仅追加的持久化事件流是系统的事实来源。每一轮中，Agent 先持久化用户消息，再根据经过验证的会话状态构建有界模型请求，随后持久化模型响应，并让其中的工具调用依次经过预检查和审批门禁。任何有副作用的操作都会被持久化的 `tool_started` 和终态事件包围。

```text
用户输入
  → 追加事件
  → 构建有界上下文
  → 调用模型
  → 追加 Assistant 和工具调用事件
  → 预检查与审批
  → 追加 tool_started
  → 执行工具
  → 追加 completed / failed / unknown 结果
  → 继续模型循环
```

上下文压缩不会重写或删除原始 Transcript。Checkpoint 只保存增量工作状态；重建请求时，系统会在预算内选择系统指令、权威 Goal、最新 Checkpoint，以及最近的完整工具调用轮次。被移出活跃上下文的精确细节，仍可通过有界的历史和 Artifact 检索找回。

恢复会话时，系统完全根据事件重建状态。未完成的读取会记为 `interrupted`；未完成的写入或命令会记为 `unknown`，且绝不自动重放。只有每项验收标准都引用了匹配的持久化证据，Goal 才能完成。

## 安装

Mini Agent 当前支持 Python 3.12–3.14 的 macOS 和 Linux，暂不支持
Windows。

内置 `search` 工具需要系统 `PATH` 中存在
[ripgrep](https://github.com/BurntSushi/ripgrep) 的 `rg` 可执行文件；Python wheel
无法安装这个系统二进制。请先通过操作系统包管理器安装 ripgrep，并确认：

```bash
rg --version
```

缺少它时，Mini Agent 仍可安装，但 `search` 会明确返回 `missing_dependency` 工具错误。

从源码检出目录安装命令：

```bash
git clone https://github.com/kangqiyue/mini-agent.git
cd mini-agent
uv tool install .
mini-agent --help
```

以上方式使用 [uv](https://docs.astral.sh/uv/)。也可以先激活 Python 3.12–3.14
虚拟环境，再执行 `python -m pip install .`。参与开发时请使用下方“开发”中的
锁定两阶段流程：先安装固定的构建后端，再以无构建隔离方式安装项目。
该工具链固定 `hatchling==1.32.0` 及其 editable 安装所需的
`editables==0.6`。
仓库开发与发布检查要求使用 `uv==0.11.28`，避免锁文件和构建行为随工具版本漂移。

项目目前尚未发布 v0.1 包。若要在不依赖源码目录的情况下测试候选 wheel，可直接安装构建产物：

```bash
python -m pip install /path/to/mini_agent-0.1.0rc1-py3-none-any.whl
mini-agent --help
```

只有在正式发布到公共包索引后，才应使用对应的包名安装命令；当前文档不假设它已经上线。

## 运行

```bash
mini-agent                                      # 在当前目录启动对话
mini-agent init-config --workspace /path/to/project
mini-agent chat --workspace /path/to/project
mini-agent run "Review this project" --workspace /path/to/project --json
mini-agent sessions --workspace /path/to/project
mini-agent resume                                # 恢复当前工作区最近活动的会话
mini-agent resume SESSION_ID --workspace /path/to/project
mini-agent inspect SESSION_ID --workspace /path/to/project
mini-agent evaluate SESSION_ID --fixture fixture.json --output report.json
mini-agent benchmark --output benchmark.json
```

`mini-agent run` 执行一个非交互回合，默认拒绝文件写入和命令执行；添加 `--auto-approve` 后逐次批准这些操作。JSON 结果包含回合结束原因、工具调用数量、当前 Goal 状态，以及本回合主模型响应上报的 Token 用量。缺失或不完整的用量字段保持 `null`。回合结束并不代表任务通过验收；清理或会话收尾失败时返回非零退出码。较长的工具输出与交互模式一样保存为可检索的 Artifact。

`apply_patch` 的 `expected_content` 省略或为 `null` 时表示新建文件：父目录必须已存在，目标文件必须不存在。新建操作需要单次批准，不会授权日后替换同一路径的文件。

不带子命令运行 `mini-agent`，等价于 `mini-agent chat --workspace .`。为了兼容脚本，以及需要选择其他工作区的场景，显式的 `chat` 子命令仍然保留。

启动聊天时不会立即创建持久化 Session。只有第一条普通请求到来时，Mini Agent 才会创建 Session 并记录工作；因此直接 `/exit`、`Ctrl-C` 或关闭未输入的界面不会留下空会话。`/system`、`/context`、`/goal` 和 `/resume` 可以在这个未保存阶段使用。旧版本留下的纯生命周期空会话会从 `sessions` 和无参数 `resume` 列表中隐藏，但不会被自动删除。

`mini-agent resume` 不要求复制 Session ID：在交互式终端中，它会列出当前工作区和模型下的历史会话，按最后活动时间排序，并显示最近一条用户消息摘要。直接回车恢复第 1 条（最近会话），也可以输入序号或 Session ID。恢复成功后会在新输入框前回放最近 30 条 User/Assistant 消息；工具调用和底层事件不占用这 30 条额度。非交互脚本中仍会自动选择最近会话。传入 `SESSION_ID` 时直接精确恢复，不显示选择列表，但仍会回放最近对话。

### 对话内 `/` 命令

这些命令由 Mini Agent 在本地处理，不会作为普通消息发送给模型。

| 命令 | 作用 |
|---|---|
| `/output brief` | 默认简略模式：显示工具名、简短参数、完成状态和错误。 |
| `/output detailed` | 详细模式：同时显示有长度限制的工具参数和结果。 |
| `/history` | 按当前显示模式回放最近对话与工具结果；先切到 detailed 可以查看已保存输出。 |
| `/permissions` | 查看审批规则与当前会话的文件授权数量。 |
| `/system` | 查看下一次模型请求实际使用的完整 System Prompt；已知工作区/用户目录前缀会被归一化，自动采集的分支名只保留分类。 |
| `/context` | 查看当前上下文投影的策略版本、估算 Token、利用率、消息数、Checkpoint 和事件水位。 |
| `/checkpoint` | 把当前增量工作状态整理成一个持久化 Checkpoint。 |
| `/compact [关注点]` | 先创建 Checkpoint，再开启新的有界上下文周期；可选的关注点用于指导本次压缩。 |
| `/goal` | 查看当前持久化 Goal、状态和验收标准。 |
| `/goal set <目标>` | 创建或修改当前目标。 |
| `/goal criteria <JSON 数组>` | 设置结构化验收标准。 |
| `/goal complete <JSON 数组>` | 提交持久化证据；只有全部验收标准通过才会完成 Goal。 |
| `/goal block <原因>` | 把当前 Goal 标记为阻塞。 |
| `/goal resume` | 重新激活被阻塞的 Goal。 |
| `/goal cancel` | 取消当前 Goal。 |
| `/resume` | 显示当前工作区和模型下的历史会话；选择后关闭当前会话，并从目标 Transcript 重建运行时状态。直接回车选择最近会话。 |
| `/resume SESSION_ID` | 安全切换到同一工作区、同一模型的指定会话。 |
| `/exit` | 结束交互并持久化会话停止状态。 |

显示模式在本进程内切换 `/resume` 时保留，不改变模型上下文或 Transcript。
较长的结果会显示 Artifact 引用。审批会先解释具体命令或文件、适用时的工作目录及授权范围，
并暂停工作动画：`o` 单次允许，`d` 拒绝；只有明确提供时，`s` 才能授权本会话内的同一文件。
空输入或无效选项都拒绝。默认对外部命令逐次审批，包括可能调用配置钩子或过滤器的 Git 命令。

需要跳过工具审批时，显式选择自动批准：

```bash
mini-agent --auto-approve
mini-agent chat --auto-approve
mini-agent resume --auto-approve
```

状态栏会显示 `approvals AUTO`，命令和文件写入直接执行。该选择仅在本进程内生效，
包括 `/resume` 切换；退出后重新启动，需要再次添加参数。工作区路径检查、凭据脱敏、
审批记录和恢复规则继续生效，配置来源的信任检查也保持独立。
适合可信的隔离工作区；获批命令仍使用当前操作系统用户的权限。

已有 macOS Colima/LiteLLM 网关时，可使用
[手动启动脚本](scripts/local_gateway_startup/README.md)启动现有服务并等待健康检查。

当前没有单独的 `/model` 命令。模型名称会显示在启动面板中；对话过程中执行 `/system`，在 `Runtime context` 的 `Model` 字段也可以查看。最常用的几个命令是：

```text
/system
/context
/checkpoint
/compact 只保留当前实现、未解决问题和验证状态
/goal
/resume
/exit
```

Checkpoint 和上下文重建只改变下一次请求使用的派生视图，不会重写仅追加的 Transcript。更早的精确细节仍可通过历史和 Artifact 检索工具找回。

Goal 命令刻意保持显式：

```text
/goal set Ship and verify the change
/goal criteria [{"criterion_id":"tests","description":"Tests pass","evidence_kind":"command_succeeded"}]
/goal complete [{"criterion_id":"tests","event_ids":[42],"note":"pytest exited zero"}]
/goal block Waiting for an external service
/goal resume
/goal cancel
```

如果当前没有 Goal，第一条普通用户请求会创建一个可编辑的目标。没有验收标准的 Goal 不能完成。将 Goal 标记为 `completed` 时，每项标准都必须提供一个证据，引用已经存在的持久化事件，并满足所要求的机械证据类型：`assistant_response`、`tool_completed`、`command_succeeded`、`file_modified` 或 `artifact_created`。运行时会把这份权威 Goal 状态注入每次模型请求；模型仅在文本中声称“已完成”无法绕过门禁。

`mini-agent evaluate` 是只读的：它加载带版本的信息原子 Fixture，根据经过验证的持久化状态，计算信息保留率、状态精确率/召回率/F1、陈旧事实、检索、错误完成、恢复、周期数量和估算请求 Token。`mini-agent benchmark` 会在三个相互隔离的合成临时编程工作区中，运行实际 runtime 的读取 → 修改 → 执行 → 完成门禁 → 三次上下文重建流程。它不会调用外部模型。除非提供真实数据，否则 Provider 用量、成本和人工评测都会明确标记为 `unavailable`；工具不会虚构这些指标。

读取操作自动执行。每条命令，以及尚未获得精确路径授权的合规文件修改，都会在持久化 `tool_started` 前要求用户在终端明确决定。对于合法且不含敏感信息的 `apply_patch`，可以授权在当前会话剩余时间内修改一个确定的相对文件路径；此后每次修改的完整脱敏参数仍会持久化，便于审计。包含凭证特征或已脱敏标记的替换内容会在审批前被拒绝，且不会写入文件。

`exec_command` 每次调用都只获得一次性审批，只接受 `argv` 数组，不接受 shell 字符串。包含凭证特征或已脱敏标记的命令参数会在审批前被拒绝。

`exec_command` 不是操作系统级沙箱。它从当前用户的 `PATH` 解析命令名称，再以最小环境运行解析后的可执行文件；超时时会终止该命令原有的进程组，但刻意脱离到新会话的程序仍可能在清理后继续运行。请在批准前检查每一条命令。

工作区路径检查会拒绝绝对路径、逃出工作区的父目录遍历、被排除的运行时数据和符号链接逃逸。文件工具与命令工作目录会从已验证的工作区根开始，通过保持打开且禁止跟随符号链接的目录描述符逐级访问；写操作还会在目录身份影响安全性时重新验证已绑定的父目录。这可以关闭常见的路径替换竞争，但它不是操作系统沙箱：以同一用户身份运行的其他进程仍可修改工作区内容，获批命令也保留该用户的操作系统权限。仅应在会话期间文件系统命名空间可信的工作区中使用 Mini Agent。

## 开发

```bash
uv sync --extra dev --locked --no-install-project
uv sync --extra dev --locked --no-build-isolation
uv run pytest
uv run ruff check .
uv run pyright
uv build --no-build-isolation
uv run mini-agent benchmark --output reports/v0.1-benchmark.json
```

以上命令也是常规的本地发布检查。维护者还需按照
[docs/RELEASING.md](docs/RELEASING.md) 完成 clean-root 历史扫描、私有
denylist 检查和全新环境中的 wheel 安装验证。

## 配置

把安装包内的完整示例初始化到 Agent 实际工作的目标 Workspace，再替换 Provider 配置：

```bash
mini-agent init-config --workspace /path/to/project
# 编辑 /path/to/project/.mini-agent/config.toml
export MINI_AGENT_API_KEY="<provider-key>"
mini-agent chat --workspace /path/to/project
```

`init-config` 可从已安装 wheel 或源码目录运行，会创建私有权限文件，并拒绝覆盖已有配置。
真实的 `.mini-agent/config.toml` 已被 Git 忽略。示例覆盖全部支持字段，明确标注实际默认值，
并以禁用的注释形式展示默认未设置的可选项；测试也会校验示例与代码没有漂移。
源码用户也可直接查看或复制
[.mini-agent/config.example.toml](.mini-agent/config.example.toml)。配置查找顺序是：显式 `--config`、
`<workspace>/.mini-agent/config.toml`、操作系统对应的用户配置目录。
`runtime.data_dir` 的相对路径以最终选中的配置文件所在目录为基准解析。文档中的可选
`data_dir = "data"` 因此会解析为 `.mini-agent/data/`，该目录已被本仓库 Git 忽略；如果改用
其他位于仓库内的数据目录，必须在该项目的 Git ignore 规则中自行加入它。

使用本地 LiteLLM 网关时，先查询网关当前注册的模型，再填入它返回的准确模型 ID。启用可选的
本地集成测试前，必须显式设置 `MINI_AGENT_LITELLM_MODEL`；该测试刻意不提供默认模型：

```toml
[model]
provider = "openai-compatible"
model = "<gateway-model-id>"
base_url = "http://127.0.0.1:4000/v1"
api_key_env = "LITELLM_API_KEY"
```

本地 loopback 网关允许使用 HTTP；远程 Provider 必须使用 HTTPS。配置文件只保存凭证对应的
环境变量名，不保存凭证值。全部支持字段、实际默认值和禁用的可选示例见
[.mini-agent/config.example.toml](.mini-agent/config.example.toml)。

`SystemPromptAssembler` 会为每次模型请求生成一条权威的首条 System Message。为了提高 Provider 前缀缓存的命中机会，内容按“稳定前缀 + 动态后缀”排列。主模型和 checkpoint 模型请求离开进程前，会把已知 workspace/home 前缀归一化为 `<workspace-root>`/`<home>`；该边界覆盖 System Prompt、对话/工具视图、history、checkpoint 输入和自定义工具 schema。自动采集的 Git 分支信息在写入 System Prompt 前只保留分类；这不是分支名 DLP，对话、项目内容或已审批工具输出里的分支文本仍可能发给 Provider。当前 OpenAI-compatible 适配器依赖 Provider 的自动前缀缓存，尚未发送专有的 `cache_control`，也尚未采集 `cached_tokens`。

工作区专属指令通过 `system_prompt.instructions_file` 配置；该文件必须位于工作区内，并受严格的读取长度限制。完整的有界文件内容会在使用前分类：PEM 私钥会被拒绝，公开证书仍允许使用。该文件内容、对话、选中的文件内容和工具结果会按任务需要发送给已配置 Provider，因此未经批准不要在其中放入公司或组织敏感信息。路径归一化不是通用 DLP：其它绝对路径、组织名称、域名和项目内容仍可能被发送。终端继续使用 Rich；每次输入前将模型单独显示一行，其余状态块按终端宽度换行，显示 Git、估算上下文占用、Checkpoint 版本、Goal 状态和输出模式，例如：

```text
provider/model-name
git main │ ctx ~31.2k/83.6k 37% │ checkpoint v2+ │ goal active
output brief │ approvals ask
```

自动 Git telemetry 不读取工作区 clean/dirty 状态，因此该状态行不会用 `*` 表示未提交
修改。其中 `~` 表示当前 Token 是本地保守估算值，Checkpoint 后的 `+` 表示最新
Transcript 中还有尚未纳入该 Checkpoint 的事件；`/context` 仍用于查看详细上下文状态。

## 许可证

版权所有 © 2026 kangqiyue。Mini Agent 采用
[Apache License 2.0](LICENSE) 许可证；完整条款见 [LICENSE](LICENSE)。
