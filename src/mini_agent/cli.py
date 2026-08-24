"""Command-line entry points for session creation and inspection."""

from __future__ import annotations

import asyncio
import json
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Never, cast

import typer
from pydantic import ValidationError

from mini_agent.agent import MiniAgent
from mini_agent.artifacts import ArtifactStore, ArtifactStoreCorruptionError
from mini_agent.benchmark import run_reproducible_benchmark, write_benchmark_report
from mini_agent.checkpoint import CheckpointStore, CheckpointStoreCorruptionError
from mini_agent.checkpoint_writer import CheckpointRecoveryRequiredError
from mini_agent.config import ConfigNotFoundError, MiniAgentConfig, load_config
from mini_agent.config_template import (
    ConfigAlreadyExistsError,
    ConfigInitializationError,
    initialize_workspace_config,
)
from mini_agent.context import (
    ContextConfigurationError,
    Utf8TokenEstimator,
    initial_goal_objective,
    resolve_request_budget,
    validate_prospective_request_floor,
)
from mini_agent.evaluation import (
    evaluate_session,
    load_evaluation_fixture,
    write_evaluation_report,
)
from mini_agent.event_store import EventStoreCorruptionError, EventStoreWriterBusyError
from mini_agent.events import ApprovalDecision, ArtifactCreatedData
from mini_agent.goal import (
    AcceptanceCriterion,
    CompletionEvidence,
    GoalCompletionRejected,
    GoalState,
    GoalStatus,
)
from mini_agent.history import SessionHistory
from mini_agent.host_path_redaction import (
    normalize_conversation_message_host_paths,
    normalize_host_path_text,
    normalize_model_request_host_paths,
)
from mini_agent.messages import ConversationMessage, MessageRole, ModelRequest
from mini_agent.permissions import ApprovalPrompt, ApprovalRequest, PermissionController
from mini_agent.provider import ProviderError, resolve_provider_capabilities
from mini_agent.providers import OpenAICompatibleProvider
from mini_agent.session import (
    AgentSession,
    SessionMetadata,
    SessionNotFoundError,
    SessionSummary,
    discover_sessions,
)
from mini_agent.storage_paths import find_config_path, resolve_data_dir
from mini_agent.system_prompt import SystemPromptAssembler, SystemPromptError
from mini_agent.terminal_safety import safe_terminal_text
from mini_agent.terminal_ui import TerminalChatUI
from mini_agent.tools.apply_patch import ApplyPatchTool
from mini_agent.tools.artifact_read import ARTIFACT_READ_DEFINITION, ArtifactReadTool
from mini_agent.tools.base import ToolDefinition
from mini_agent.tools.exec_command import ExecCommandTool
from mini_agent.tools.history_read import HistoryReadTool
from mini_agent.tools.history_search import HistorySearchTool
from mini_agent.tools.read_file import ReadFileTool
from mini_agent.tools.registry import ToolRegistry
from mini_agent.tools.search import SearchTool
from mini_agent.workspace import Workspace

app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="A small, observable local coding agent. Run without a command to chat here.",
)

ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", help="Path to config.toml."),
]
WorkspaceOption = Annotated[
    Path | None,
    typer.Option("--workspace", help="Workspace used for project config and session metadata."),
]


@dataclass(frozen=True)
class InteractiveResult:
    stop_reason: str
    resume_session_id: str | None = None
    initial_input: str | None = None


@dataclass(frozen=True)
class _PendingChatSetup:
    tools: tuple[ToolDefinition, ...]
    prompt_assembler: SystemPromptAssembler


_SESSION_UNREADABLE_ERRORS = (
    ArtifactStoreCorruptionError,
    CheckpointStoreCorruptionError,
    EventStoreCorruptionError,
    SessionNotFoundError,
    OSError,
    UnicodeError,
    ValidationError,
    ValueError,
)


def _raise_unreadable_session_error() -> Never:
    """Raise one stable public error for untrusted session storage."""

    raise typer.BadParameter(
        "Requested session was not found or is not readable.",
        param_hint="session_id",
    ) from None


def _raise_cli_error(message: str) -> Never:
    """Report one stable CLI failure without formatter-dependent exception types."""

    typer.echo(message, err=True)
    raise typer.Exit(code=1) from None


def _raise_session_busy_error() -> Never:
    """Raise one stable public error for an already-owned session writer."""

    _raise_cli_error("Requested session is currently in use by another process.")


@app.callback()
def main(context: typer.Context) -> None:
    """Start chat in the current directory when no subcommand is given."""
    if context.invoked_subcommand is None:
        chat()


@app.command()
def init_config(workspace: WorkspaceOption = None) -> None:
    """Create a complete workspace config without overwriting an existing file."""

    try:
        initialize_workspace_config(_resolve_workspace(workspace))
    except ConfigAlreadyExistsError:
        raise typer.BadParameter(
            "Configuration already exists and was not overwritten.",
            param_hint="--workspace",
        ) from None
    except ConfigInitializationError:
        raise typer.BadParameter(
            "Configuration could not be initialized safely.",
            param_hint="--workspace",
        ) from None
    typer.echo("Created .mini-agent/config.toml in the selected workspace.")


@app.command()
def chat(
    workspace: WorkspaceOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Start chat and persist a session only when work begins."""
    resolved_workspace = _resolve_workspace(workspace)
    config = _load_runtime_config(resolved_workspace, config_path)
    pending_setup = _prepare_pending_chat(config, resolved_workspace)
    try:
        pending_result = _run_pending_chat(
            config,
            resolved_workspace,
            setup=pending_setup,
        )
    except SystemPromptError:
        _raise_cli_error("Configured system prompt could not be assembled safely.")
    if pending_result.stop_reason in {"end_of_input", "user_interrupt", "user_exit"}:
        return
    if pending_result.stop_reason == "resume_requested":
        if pending_result.resume_session_id is None:
            raise AssertionError("Resume request must select a session")
        try:
            session = AgentSession.resume(
                data_dir=resolve_data_dir(config),
                session_id=pending_result.resume_session_id,
            )
        except EventStoreWriterBusyError:
            _raise_session_busy_error()
        except _SESSION_UNREADABLE_ERRORS:
            _raise_unreadable_session_error()
        stop_reason = _run_session_chain(
            config,
            session,
            registrations=session.artifact_registrations,
        )
        if stop_reason in {
            "turn_failure",
            "checkpoint_recovery_required",
            "session_finalization_failed",
        }:
            raise typer.Exit(code=1)
        return
    if pending_result.stop_reason != "start_session" or pending_result.initial_input is None:
        raise AssertionError("Pending chat must stop with input, resume, or exit")
    try:
        _validate_pending_request_floor(
            config,
            resolved_workspace,
            pending_setup,
            initial_input=pending_result.initial_input,
        )
    except SystemPromptError:
        _raise_cli_error("Configured system prompt could not be assembled safely.")
    except ContextConfigurationError:
        typer.echo("Configuration cannot fit the required model request.", err=True)
        raise typer.Exit(code=1) from None

    try:
        session = AgentSession.create(
            data_dir=resolve_data_dir(config),
            workspace=resolved_workspace,
            model=config.model.model,
        )
    except (EventStoreCorruptionError, OSError, ValueError):
        _raise_cli_error("Session storage could not be initialized safely.")
    stop_reason = _run_session_chain(
        config,
        session,
        registrations=(),
        initial_input=pending_result.initial_input,
        show_welcome=False,
    )
    if stop_reason in {
        "turn_failure",
        "checkpoint_recovery_required",
        "session_finalization_failed",
    }:
        raise typer.Exit(code=1)


@app.command()
def resume(
    session_id: Annotated[
        str | None,
        typer.Argument(help="Session id; defaults to the latest session in this workspace."),
    ] = None,
    workspace: WorkspaceOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Resume a session, defaulting to the latest one in this workspace."""
    resolved_workspace = _resolve_workspace(workspace)
    config = _load_runtime_config(resolved_workspace, config_path)
    data_dir = resolve_data_dir(config)
    if session_id is None:
        matching_sessions = _matching_sessions(
            data_dir=data_dir,
            workspace=resolved_workspace,
            model=config.model.model,
        )
        if not matching_sessions:
            raise typer.BadParameter(
                "No resumable session was found for this workspace and model.",
                param_hint="session_id",
            )
        selected_session_id = (
            TerminalChatUI().select_session(matching_sessions)
            if _is_interactive_terminal()
            else matching_sessions[0].session_id
        )
        if selected_session_id is None:
            return
    else:
        selected_session_id = session_id
    try:
        metadata = AgentSession.peek_metadata(data_dir=data_dir, session_id=selected_session_id)
    except _SESSION_UNREADABLE_ERRORS:
        _raise_unreadable_session_error()
    metadata_issue = _resume_metadata_issue(
        metadata=metadata,
        workspace=str(resolved_workspace),
        model=config.model.model,
    )
    if metadata_issue == "workspace":
        raise typer.BadParameter(
            "Requested session belongs to a different workspace.",
            param_hint="--workspace",
        )
    if metadata_issue == "model":
        raise typer.BadParameter(
            "Configured model differs from the model recorded by this session",
            param_hint="--config",
        )

    try:
        session = AgentSession.resume(data_dir=data_dir, session_id=selected_session_id)
    except EventStoreWriterBusyError:
        _raise_session_busy_error()
    except _SESSION_UNREADABLE_ERRORS:
        _raise_unreadable_session_error()
    stop_reason = _run_session_chain(
        config,
        session,
        registrations=session.artifact_registrations,
    )
    if stop_reason in {
        "turn_failure",
        "checkpoint_recovery_required",
        "session_finalization_failed",
    }:
        raise typer.Exit(code=1)


@app.command("sessions")
def show_sessions(
    workspace: WorkspaceOption = None,
    config_path: ConfigOption = None,
) -> None:
    """List locally persisted sessions."""
    config = _load_runtime_config(_resolve_workspace(workspace), config_path)
    summaries = tuple(
        summary
        for summary in _discover_readable_sessions(resolve_data_dir(config))
        if summary.has_work
    )
    if not summaries:
        typer.echo("No sessions found.")
        return

    for summary in summaries:
        typer.echo(
            safe_terminal_text(
                f"{summary.session_id}  {summary.created_at.isoformat()}  "
                f"active={summary.last_event_at.isoformat()}  "
                f"events={summary.event_count}  last={summary.last_event_kind}  "
                f"model={summary.model}"
            )
        )


@app.command()
def inspect(
    session_id: Annotated[str, typer.Argument(help="Session id to inspect.")],
    workspace: WorkspaceOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Show event metadata without printing message bodies."""
    config = _load_runtime_config(_resolve_workspace(workspace), config_path)
    try:
        session = AgentSession.load(data_dir=resolve_data_dir(config), session_id=session_id)
    except _SESSION_UNREADABLE_ERRORS:
        _raise_unreadable_session_error()
    try:
        typer.echo(safe_terminal_text(f"session={session.metadata.session_id}"))
        typer.echo(safe_terminal_text(f"workspace={session.metadata.workspace}"))
        typer.echo(safe_terminal_text(f"model={session.metadata.model}"))
        for event in session.events:
            turn = event.turn_id or "-"
            typer.echo(
                safe_terminal_text(
                    f"{event.id:04d}  {event.timestamp.isoformat()}  "
                    f"{event.data.kind}  turn={turn}"
                )
            )
    finally:
        session.close()


@app.command()
def evaluate(
    session_id: Annotated[str, typer.Argument(help="Session id to evaluate.")],
    fixture: Annotated[Path, typer.Option("--fixture", help="Information-atom fixture JSON.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional report JSON path."),
    ] = None,
    workspace: WorkspaceOption = None,
    config_path: ConfigOption = None,
) -> None:
    """Evaluate durable retention/recovery without calling a model or tools."""

    config = _load_runtime_config(_resolve_workspace(workspace), config_path)
    try:
        session = AgentSession.load(data_dir=resolve_data_dir(config), session_id=session_id)
    except _SESSION_UNREADABLE_ERRORS:
        _raise_unreadable_session_error()
    try:
        try:
            checkpoints = CheckpointStore.open(
                session.paths.root,
                registrations=session.checkpoint_registrations,
                workspace_root=Path(session.metadata.workspace),
                writable=False,
            )
        except _SESSION_UNREADABLE_ERRORS:
            _raise_unreadable_session_error()
        try:
            report = evaluate_session(
                session,
                checkpoints,
                load_evaluation_fixture(fixture),
            )
        except (OSError, UnicodeError, ValueError):
            raise typer.BadParameter(
                "Evaluation fixture could not be loaded or does not match the session.",
                param_hint="--fixture",
            ) from None
        if output is not None:
            try:
                write_evaluation_report(report, output)
            except OSError:
                _raise_cli_error("Evaluation report could not be written safely.")
        typer.echo(report.model_dump_json(indent=2))
    finally:
        session.close()


@app.command()
def benchmark(
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional aggregate report JSON path."),
    ] = None,
) -> None:
    """Run the deterministic three-task v0.1 coding benchmark."""

    try:
        report = asyncio.run(run_reproducible_benchmark())
    except (OSError, UnicodeError, ValueError):
        _raise_cli_error("Benchmark could not be completed safely.")
    if output is not None:
        try:
            write_benchmark_report(report, output)
        except (OSError, UnicodeError, ValueError):
            _raise_cli_error("Benchmark report could not be written safely.")
    typer.echo(report.model_dump_json(indent=2))


def _load_runtime_config(workspace: Path, config_path: Path | None) -> MiniAgentConfig:
    """Load config without allowing parser diagnostics to expose its contents."""

    try:
        resolved_config_path = find_config_path(workspace, config_path)
        return load_config(resolved_config_path)
    except (
        ConfigNotFoundError,
        OSError,
        RuntimeError,
        UnicodeError,
        tomllib.TOMLDecodeError,
        ValidationError,
        ValueError,
    ):
        raise typer.BadParameter(
            "Configuration could not be loaded safely.",
            param_hint="--config",
        ) from None


def _matching_sessions(
    *, data_dir: Path, workspace: Path, model: str
) -> tuple[SessionSummary, ...]:
    canonical_workspace = str(workspace)
    return tuple(
        summary
        for summary in _discover_readable_sessions(data_dir)
        if summary.workspace == canonical_workspace
        and summary.model == model
        and summary.has_work
        and summary.goal_status not in {GoalStatus.COMPLETED, GoalStatus.CANCELLED}
    )


def _resume_metadata_issue(
    *,
    metadata: SessionMetadata,
    workspace: str,
    model: str,
) -> Literal["workspace", "model"] | None:
    """Return the stable compatibility reason without exposing metadata values."""

    if metadata.workspace != workspace:
        return "workspace"
    if metadata.model != model:
        return "model"
    return None


def _discover_readable_sessions(data_dir: Path) -> tuple[SessionSummary, ...]:
    try:
        discovery = discover_sessions(data_dir)
    except OSError:
        _raise_cli_error("Session discovery could not be completed safely.")
    if discovery.unreadable_session_ids:
        session_ids = ", ".join(discovery.unreadable_session_ids)
        typer.echo(
            f"Warning: skipped unreadable sessions: {safe_terminal_text(session_ids)}",
            err=True,
        )
    return discovery.summaries


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _resolve_workspace(workspace: Path | None) -> Path:
    selected_workspace = workspace or Path.cwd()
    try:
        resolved_workspace = selected_workspace.expanduser().resolve(strict=True)
        if not resolved_workspace.is_dir():
            raise ValueError
    except (OSError, RuntimeError, ValueError):
        raise typer.BadParameter(
            "Workspace must be an existing directory",
            param_hint="--workspace",
        ) from None
    return resolved_workspace


def _run_pending_chat(
    config: MiniAgentConfig,
    workspace_path: Path,
    *,
    setup: _PendingChatSetup | None = None,
) -> InteractiveResult:
    """Handle local commands before any durable session exists."""

    ui = TerminalChatUI()
    pending_setup = setup or _prepare_pending_chat(config, workspace_path)
    prompt_assembler = pending_setup.prompt_assembler
    runtime = prompt_assembler.runtime_context()
    ui.show_welcome(session_id=None, runtime=runtime)

    while True:
        ui.show_pending_status(runtime)
        try:
            user_input = ui.read_input()
        except EOFError:
            return InteractiveResult(stop_reason="end_of_input")
        except KeyboardInterrupt:
            typer.echo()
            return InteractiveResult(stop_reason="user_interrupt")

        command = user_input.strip()
        if not command:
            continue
        if command == "/exit":
            return InteractiveResult(stop_reason="user_exit")
        if command == "/system":
            runtime = prompt_assembler.runtime_context()
            ui.show_system(prompt_assembler.assemble(None, runtime=runtime).content or "")
            continue
        if command == "/context":
            typer.echo("Context is empty; the session will be created by the first request.")
            continue
        if command == "/goal":
            typer.echo("No goal yet; the first user request will create one.")
            continue
        if command == "/resume" or command.startswith("/resume "):
            candidate_id = _select_pending_resume_target(
                config=config,
                workspace=workspace_path,
                requested_id=command[len("/resume") :].strip(),
                ui=ui,
            )
            if candidate_id is None:
                continue
            return InteractiveResult(
                stop_reason="resume_requested",
                resume_session_id=candidate_id,
            )
        return InteractiveResult(stop_reason="start_session", initial_input=user_input)


def _prepare_pending_chat(
    config: MiniAgentConfig,
    workspace_path: Path,
) -> _PendingChatSetup:
    """Build the one prompt/tool definition shared by pending UI and preflight."""

    tools = _pending_tool_definitions(config, workspace_path)
    return _PendingChatSetup(
        tools=tools,
        prompt_assembler=SystemPromptAssembler(
            config=config.system_prompt,
            model=config.model.model,
            workspace=workspace_path,
            tools=tools,
            excluded_roots=(resolve_data_dir(config),),
        ),
    )


def _validate_pending_request_floor(
    config: MiniAgentConfig,
    workspace_path: Path,
    setup: _PendingChatSetup,
    *,
    initial_input: str,
) -> None:
    """Reject an unsendable first request before any session is created."""

    provider = OpenAICompatibleProvider(config.model)
    budget = resolve_request_budget(
        model=config.model,
        context=config.context,
        capabilities=resolve_provider_capabilities(provider, config.model),
    )
    runtime = setup.prompt_assembler.runtime_context()
    prospective_goal = GoalState(
        goal_id="0" * 32,
        objective=initial_goal_objective(
            normalize_host_path_text(initial_input.strip(), workspace_root=workspace_path)
        ),
        acceptance_criteria=(),
    )
    system_message = normalize_conversation_message_host_paths(
        setup.prompt_assembler.assemble(prospective_goal, runtime=runtime),
        workspace_root=workspace_path,
    )
    request = normalize_model_request_host_paths(
        ModelRequest(
            model=config.model.model,
            messages=(
                system_message,
                ConversationMessage(role=MessageRole.USER, content=initial_input.strip()),
            ),
            tools=setup.tools,
            max_output_tokens=budget.desired_output_tokens,
        ),
        workspace_root=workspace_path,
    )
    validate_prospective_request_floor(
        request,
        budget=budget,
        estimator=Utf8TokenEstimator(),
        maximum_checkpoint_bytes=config.context.checkpoint_max_bytes,
    )


def _pending_tool_definitions(
    config: MiniAgentConfig,
    workspace_path: Path,
) -> tuple[ToolDefinition, ...]:
    workspace = Workspace(
        workspace_path,
        excluded_roots=(resolve_data_dir(config),),
    )
    empty_history = SessionHistory(())
    return (
        ReadFileTool(workspace).definition,
        SearchTool(workspace).definition,
        HistorySearchTool(lambda: empty_history).definition,
        HistoryReadTool(lambda: empty_history).definition,
        ARTIFACT_READ_DEFINITION,
        ApplyPatchTool(workspace).definition,
        ExecCommandTool(workspace).definition,
    )


def _select_pending_resume_target(
    *,
    config: MiniAgentConfig,
    workspace: Path,
    requested_id: str,
    ui: TerminalChatUI,
) -> str | None:
    if requested_id:
        if _can_resume_without_current_session(
            config=config,
            workspace=workspace,
            candidate_id=requested_id,
        ):
            return requested_id
        return None

    candidates = _matching_sessions(
        data_dir=resolve_data_dir(config),
        workspace=workspace,
        model=config.model.model,
    )
    if not candidates:
        typer.echo("No resumable session was found for this workspace and model.")
        return None
    if not _is_interactive_terminal():
        return candidates[0].session_id
    return ui.select_session(candidates)


def _can_resume_without_current_session(
    *,
    config: MiniAgentConfig,
    workspace: Path,
    candidate_id: str,
) -> bool:
    try:
        metadata = AgentSession.peek_metadata(
            data_dir=resolve_data_dir(config),
            session_id=candidate_id,
        )
        validated_session = AgentSession.load(
            data_dir=resolve_data_dir(config),
            session_id=candidate_id,
        )
        validated_session.close()
    except _SESSION_UNREADABLE_ERRORS:
        typer.echo("Requested session was not found or is not readable.", err=True)
        return False
    metadata_issue = _resume_metadata_issue(
        metadata=metadata,
        workspace=str(workspace),
        model=config.model.model,
    )
    if metadata_issue == "workspace":
        typer.echo("Requested session belongs to a different workspace.", err=True)
        return False
    if metadata_issue == "model":
        typer.echo("Requested session uses a different configured model.", err=True)
        return False
    return True


async def _run_interactive(
    config: MiniAgentConfig,
    session: AgentSession,
    *,
    registrations: tuple[ArtifactCreatedData, ...],
    initial_input: str | None = None,
    show_welcome: bool = True,
) -> InteractiveResult:
    provider = OpenAICompatibleProvider(config.model)
    workspace = build_workspace_for_session(config, session)
    artifacts = ArtifactStore.open(
        session.paths.root,
        registrations=registrations,
        workspace_root=Path(session.metadata.workspace),
    )

    def current_history() -> SessionHistory:
        return SessionHistory(
            session.events,
            workspace_root=Path(session.metadata.workspace),
        )

    tools = ToolRegistry(
        (
            ReadFileTool(workspace),
            SearchTool(workspace),
            HistorySearchTool(current_history),
            HistoryReadTool(current_history),
            ArtifactReadTool(artifacts),
            ApplyPatchTool(workspace),
            ExecCommandTool(workspace),
        )
    )
    permissions = PermissionController(
        prompt=TerminalApprovalPrompt(),
        session_grant_fingerprints=session.session_grant_fingerprints,
    )
    agent = MiniAgent(
        config=config,
        provider=provider,
        session=session,
        tools=tools,
        artifacts=artifacts,
        permissions=permissions,
    )
    ui = TerminalChatUI()
    if show_welcome:
        ui.show_welcome(
            session_id=session.metadata.session_id,
            runtime=agent.runtime_context(),
        )
        ui.show_recent_conversation(session.conversation_messages(), limit=30)
    stop_reason: str | None = None
    resume_session_id: str | None = None
    pending_input = initial_input
    has_provider_cleanup_failure = False

    try:
        while stop_reason is None:
            if pending_input is not None:
                user_input = pending_input
                pending_input = None
            else:
                ui.show_status(agent.status())
                try:
                    user_input = ui.read_input()
                except EOFError:
                    stop_reason = "end_of_input"
                    continue
                except KeyboardInterrupt:
                    typer.echo()
                    stop_reason = "user_interrupt"
                    continue

            if user_input.strip() == "/exit":
                stop_reason = "user_exit"
                continue
            if user_input.strip() == "/resume" or user_input.strip().startswith("/resume "):
                requested_id = user_input.strip()[len("/resume") :].strip()
                if requested_id:
                    candidate_id = requested_id
                elif _is_interactive_terminal():
                    candidates = _matching_sessions(
                        data_dir=resolve_data_dir(config),
                        workspace=Path(session.metadata.workspace),
                        model=config.model.model,
                    )
                    candidate_id = ui.select_session(
                        candidates,
                        current_session_id=session.metadata.session_id,
                    )
                    if candidate_id is None:
                        continue
                else:
                    candidate_id = session.metadata.session_id
                if not _can_switch_session(
                    config=config,
                    current_session=session,
                    candidate_id=candidate_id,
                ):
                    continue
                resume_session_id = candidate_id
                stop_reason = "resume_requested"
                continue
            if user_input.strip() == "/context":
                show_context(agent)
                continue
            if user_input.strip() == "/system":
                ui.show_system(agent.system_context().content or "")
                continue
            if user_input.strip() == "/checkpoint":
                try:
                    await agent.checkpoint()
                except CheckpointRecoveryRequiredError:
                    typer.echo(
                        "Checkpoint state requires resume; this session will stop.",
                        err=True,
                    )
                    stop_reason = "checkpoint_recovery_required"
                    break
                except Exception:
                    typer.echo("Checkpoint failed; the previous committed state remains active.")
                else:
                    typer.echo("Checkpoint committed.")
                continue
            if user_input.strip().startswith("/goal"):
                handle_goal_command(
                    session,
                    user_input.strip(),
                    validate_goal=agent.validate_goal_request_floor,
                )
                continue
            if user_input.strip().startswith("/compact"):
                focus = user_input.strip()[len("/compact") :].strip() or None
                try:
                    projection = await agent.compact(focus=focus)
                except CheckpointRecoveryRequiredError:
                    typer.echo(
                        "Checkpoint state requires resume; this session will stop.",
                        err=True,
                    )
                    stop_reason = "checkpoint_recovery_required"
                    break
                except Exception:
                    typer.echo("Context rebuild failed; the previous projection remains active.")
                else:
                    typer.echo(
                        f"Context rebuilt: cycle={session.current_cycle_id} "
                        f"messages={len(projection.messages)} "
                        f"tokens~={projection.estimate.total_tokens}"
                    )
                continue
            if not user_input.strip():
                continue

            try:
                with ui.thinking():
                    response = await agent.run_turn(user_input)
            except ProviderError as error:
                typer.echo(
                    safe_terminal_text(f"Provider error [{error.code}]: {error}"),
                    err=True,
                )
                continue
            except Exception:
                typer.echo(
                    "Agent turn failed; resume the session before continuing.",
                    err=True,
                )
                stop_reason = "turn_failure"
                continue
            if response.content is not None:
                ui.show_assistant(response.content)
    finally:
        try:
            await provider.aclose()
        except Exception:
            typer.echo(
                "Agent cleanup failed; resume the session before continuing.",
                err=True,
            )
            has_provider_cleanup_failure = True

    try:
        session.stop(stop_reason)
    except (EventStoreCorruptionError, OSError, RuntimeError, UnicodeError, ValueError):
        typer.echo(
            "Session state could not be finalized; resume the session before continuing.",
            err=True,
        )
        return InteractiveResult(stop_reason="session_finalization_failed")
    if has_provider_cleanup_failure:
        # The terminal event may be durable even though the provider's transport
        # teardown was not confirmed.  Do not turn that ambiguous cleanup into a
        # successful CLI exit.
        return InteractiveResult(stop_reason="session_finalization_failed")
    return InteractiveResult(
        stop_reason=stop_reason,
        resume_session_id=resume_session_id,
    )


def show_context(agent: MiniAgent) -> None:
    try:
        projection = agent.context_status()
    except Exception:
        typer.echo("Context estimate is unavailable until the active request can be rebuilt.")
        return
    checkpoint = projection.checkpoint_id or "none"
    typer.echo(
        " ".join(
            (
                f"strategy={projection.strategy}",
                f"version={projection.version}",
                f"tokens~={projection.estimate.total_tokens}/{projection.input_limit}",
                f"utilization={projection.utilization_ratio:.1%}",
                f"messages={len(projection.messages)}",
                f"checkpoint={checkpoint}",
                f"watermark={projection.checkpoint_watermark}",
            )
        )
    )


def handle_goal_command(
    session: AgentSession,
    command: str,
    *,
    validate_goal: Callable[[GoalState], None] | None = None,
) -> None:
    """Handle typed JSON goal mutations without sending them to the model."""

    action, _, payload = command.partition(" ")
    del action
    payload = payload.strip()
    if not payload:
        show_goal(session)
        return
    operation, _, argument = payload.partition(" ")
    argument = argument.strip()
    try:
        if operation == "set":
            if not argument:
                raise ValueError("Objective is required")
            goal = session.goal
            if goal is None:
                _validate_prospective_goal(
                    GoalState(
                        goal_id="0" * 32,
                        objective=argument,
                        acceptance_criteria=(),
                    ),
                    validate_goal,
                )
                session.create_goal(objective=argument)
            else:
                _validate_prospective_goal(
                    goal.model_copy(update={"objective": argument}),
                    validate_goal,
                )
                session.update_goal(
                    objective=argument,
                    acceptance_criteria=goal.acceptance_criteria,
                )
        elif operation == "criteria":
            criteria = _parse_goal_models(argument, AcceptanceCriterion)
            goal = session.goal
            if goal is None:
                raise ValueError("Set an objective before criteria")
            _validate_prospective_goal(
                goal.model_copy(update={"acceptance_criteria": criteria}),
                validate_goal,
            )
            session.update_goal(
                objective=goal.objective,
                acceptance_criteria=criteria,
            )
        elif operation == "complete":
            evidence = _parse_goal_models(argument, CompletionEvidence)
            session.complete_goal(evidence)
        elif operation == "block":
            if not argument:
                raise ValueError("Blocked reason is required")
            session.block_goal(argument)
        elif operation == "resume":
            session.activate_goal()
        elif operation == "cancel":
            session.cancel_goal()
        else:
            raise ValueError("Unknown goal operation")
    except ContextConfigurationError:
        typer.echo("Goal update cannot fit the required model request.")
        return
    except GoalCompletionRejected as error:
        typer.echo("Completion gate rejected the claim:")
        for gap in error.result.gaps:
            criterion = f" [{gap.criterion_id}]" if gap.criterion_id else ""
            typer.echo(f"- {gap.code}{criterion}: {gap.message}")
        return
    except (json.JSONDecodeError, ValueError):
        typer.echo(
            "Goal command invalid. Use /goal, /goal set TEXT, "
            "/goal criteria JSON, /goal complete JSON, /goal block REASON, "
            "/goal resume, or /goal cancel."
        )
        return
    show_goal(session)


def _validate_prospective_goal(
    goal: GoalState,
    validate_goal: Callable[[GoalState], None] | None,
) -> None:
    if validate_goal is not None:
        validate_goal(goal)


def show_goal(session: AgentSession) -> None:
    goal = session.goal
    if goal is None:
        typer.echo("No goal yet; the first user request will create one.")
        return
    typer.echo(safe_terminal_text(f"goal={goal.goal_id} status={goal.status.value}"))
    typer.echo(safe_terminal_text(f"objective={goal.objective}"))
    if not goal.acceptance_criteria:
        typer.echo("acceptance criteria: none (completion is disabled)")
        return
    for criterion in goal.acceptance_criteria:
        typer.echo(
            safe_terminal_text(
                f"- {criterion.criterion_id} [{criterion.evidence_kind.value}] "
                f"{criterion.description}"
            )
        )


def _parse_goal_models[ModelT: AcceptanceCriterion | CompletionEvidence](
    value: str,
    model_type: type[ModelT],
) -> tuple[ModelT, ...]:
    if not value:
        raise ValueError("JSON array is required")
    raw_items = json.loads(value)
    if not isinstance(raw_items, list):
        raise ValueError("Goal payload must be a JSON array")
    return tuple(
        model_type.model_validate(item)
        for item in cast(list[object], raw_items)
    )


def _run_interactive_safely(
    config: MiniAgentConfig,
    session: AgentSession,
    *,
    registrations: tuple[ArtifactCreatedData, ...],
    initial_input: str | None = None,
    show_welcome: bool = True,
) -> InteractiveResult:
    """Run and close one interactive session without printing unexpected exceptions."""

    try:
        result = asyncio.run(
            _run_interactive(
                config,
                session,
                registrations=registrations,
                initial_input=initial_input,
                show_welcome=show_welcome,
            )
        )
    except Exception:
        typer.echo("Agent session failed; resume the session before continuing.", err=True)
        result = InteractiveResult(stop_reason="turn_failure")
    try:
        session.close()
    except Exception:
        typer.echo(
            "Session cleanup failed; resume the session before continuing.",
            err=True,
        )
        # A failed close may still leave the writer lock held.  In particular,
        # never open a requested /resume target after this result.
        return InteractiveResult(stop_reason="session_finalization_failed")
    return result


def _run_session_chain(
    config: MiniAgentConfig,
    session: AgentSession,
    *,
    registrations: tuple[ArtifactCreatedData, ...],
    initial_input: str | None = None,
    show_welcome: bool = True,
) -> str:
    """Run slash-command session switches without holding two writer locks."""

    current_session = session
    current_registrations = registrations
    current_initial_input = initial_input
    current_show_welcome = show_welcome
    while True:
        if current_initial_input is None and current_show_welcome:
            result = _run_interactive_safely(
                config,
                current_session,
                registrations=current_registrations,
            )
        else:
            result = _run_interactive_safely(
                config,
                current_session,
                registrations=current_registrations,
                initial_input=current_initial_input,
                show_welcome=current_show_welcome,
            )
        current_initial_input = None
        current_show_welcome = True
        if result.stop_reason != "resume_requested":
            return result.stop_reason
        if result.resume_session_id is None:
            raise AssertionError("Resume request must select a session")
        try:
            current_session = AgentSession.resume(
                data_dir=resolve_data_dir(config),
                session_id=result.resume_session_id,
            )
        except EventStoreWriterBusyError:
            typer.echo("Requested session is currently in use by another process.", err=True)
            return "turn_failure"
        except _SESSION_UNREADABLE_ERRORS:
            typer.echo("Requested session could not be resumed safely.", err=True)
            return "turn_failure"
        current_registrations = current_session.artifact_registrations


def _can_switch_session(
    *,
    config: MiniAgentConfig,
    current_session: AgentSession,
    candidate_id: str,
) -> bool:
    """Validate a slash-command target before releasing the current writer."""

    try:
        metadata = AgentSession.peek_metadata(
            data_dir=resolve_data_dir(config),
            session_id=candidate_id,
        )
        validated_session = AgentSession.load(
            data_dir=resolve_data_dir(config),
            session_id=candidate_id,
        )
        validated_session.close()
    except _SESSION_UNREADABLE_ERRORS:
        typer.echo("Requested session was not found or is not readable.", err=True)
        return False
    metadata_issue = _resume_metadata_issue(
        metadata=metadata,
        workspace=current_session.metadata.workspace,
        model=config.model.model,
    )
    if metadata_issue == "workspace":
        typer.echo("Requested session belongs to a different workspace.", err=True)
        return False
    if metadata_issue == "model":
        typer.echo("Requested session uses a different configured model.", err=True)
        return False
    return True


def build_workspace_for_session(config: MiniAgentConfig, session: AgentSession) -> Workspace:
    """Create a tool workspace that cannot inspect any persisted agent state."""

    return Workspace(
        Path(session.metadata.workspace),
        excluded_roots=(resolve_data_dir(config),),
    )


class TerminalApprovalPrompt(ApprovalPrompt):
    """Ask for an explicit terminal decision using only redacted arguments."""

    def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        typer.echo()
        typer.echo(f"Approval required: {safe_terminal_text(request.tool_name)}")
        typer.echo(safe_terminal_text(request.redacted_arguments))
        choices = "[o]nce / [d]eny"
        if request.can_allow_session:
            choices = "[o]nce / exact [s]ame file path this session / [d]eny"

        try:
            answer = input(f"allow {choices}? ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return ApprovalDecision.DENY

        if answer in {"o", "once"}:
            return ApprovalDecision.ALLOW_ONCE
        if request.can_allow_session and answer in {"s", "session"}:
            return ApprovalDecision.ALLOW_SESSION
        return ApprovalDecision.DENY


if __name__ == "__main__":
    app()
