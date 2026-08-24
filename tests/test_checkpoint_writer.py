"""Incremental single-writer checkpoint coordination tests."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mini_agent.checkpoint import Checkpoint, CheckpointItem, CheckpointRecord, CheckpointStore
from mini_agent.checkpoint_writer import (
    CheckpointCoordinator,
    CheckpointRecoveryRequiredError,
    DeterministicCheckpointExtractor,
    ModelCallBudget,
    _fit_checkpoint,  # pyright: ignore[reportPrivateUsage]
)
from mini_agent.events import (
    CheckpointCommittedData,
    CheckpointFailedData,
    CheckpointStartedData,
    StoredEvent,
    UserMessageData,
)
from mini_agent.goal import AcceptanceCriterion, EvidenceKind
from mini_agent.session import AgentSession


class RecordingExtractor:
    def __init__(
        self,
        *,
        pause_first: bool = False,
        should_fail: bool = False,
        first_failure: BaseException | None = None,
    ) -> None:
        self.calls: list[tuple[int, int]] = []
        self.pause_first = pause_first
        self.should_fail = should_fail
        self.first_failure = first_failure
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def extract(
        self,
        *,
        previous: Checkpoint | None,
        events: tuple[StoredEvent, ...],
        source_from_event_id: int,
        source_through_event_id: int,
        checkpoint_id: str,
        version: int,
        focus: str | None,
        model_call_budget: ModelCallBudget | None = None,
        reserved_model_call_count: int = 0,
    ) -> Checkpoint:
        self.calls.append((source_from_event_id, source_through_event_id))
        if self.pause_first and len(self.calls) == 1:
            self.first_started.set()
            await self.release_first.wait()
        if self.first_failure is not None:
            failure = self.first_failure
            self.first_failure = None
            raise failure
        if self.should_fail:
            raise RuntimeError("synthetic writer failure")
        return Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=events[0].session_id,
            cycle_id=events[-1].cycle_id,
            version=version,
            source_from_event_id=source_from_event_id,
            source_through_event_id=source_through_event_id,
            current_intent=focus or "Continue the active task",
            writer_model="checkpoint-model",
            created_at=datetime.now(UTC),
        )


def _session(tmp_path: Path) -> AgentSession:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    session.append_user_message("first", turn_id=session.new_turn_id())
    session.append_user_message("second", turn_id=session.new_turn_id())
    return session


@pytest.mark.asyncio
async def test_checkpoint_writer_commits_incremental_watermarks(tmp_path: Path) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    extractor = RecordingExtractor()
    coordinator = CheckpointCoordinator(session=session, store=store, extractor=extractor)

    first = await coordinator.checkpoint(through_event_id=2, focus="first focus")
    second = await coordinator.checkpoint(through_event_id=3, focus="second focus")

    assert first.version == 1
    assert second.version == 2
    assert extractor.calls == [(1, 2), (3, 3)]
    assert [type(event.data) for event in session.events[-4:]] == [
        CheckpointStartedData,
        CheckpointCommittedData,
        CheckpointStartedData,
        CheckpointCommittedData,
    ]
    session.close()
    loaded = AgentSession.load(data_dir=tmp_path / "data", session_id=session.metadata.session_id)
    reopened = CheckpointStore.open(
        loaded.paths.root,
        registrations=loaded.checkpoint_registrations,
        workspace_root=Path(loaded.metadata.workspace),
    )
    assert reopened.current == second
    loaded.close()


@pytest.mark.asyncio
async def test_checkpoint_writer_coalesces_a_new_watermark_while_busy(tmp_path: Path) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    extractor = RecordingExtractor(pause_first=True)
    coordinator = CheckpointCoordinator(session=session, store=store, extractor=extractor)

    first_task = asyncio.create_task(coordinator.checkpoint(through_event_id=2))
    await extractor.first_started.wait()
    second_task = asyncio.create_task(coordinator.checkpoint(through_event_id=3))
    await asyncio.sleep(0)
    extractor.release_first.set()

    first, second = await asyncio.gather(first_task, second_task)

    assert first.source_through_event_id == 3
    assert second == first
    assert extractor.calls == [(1, 2), (3, 3)]
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_failure_does_not_advance_visible_watermark(tmp_path: Path) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(should_fail=True),
    )

    with pytest.raises(RuntimeError, match="synthetic writer failure"):
        await coordinator.checkpoint(through_event_id=3)

    assert store.current is None
    assert isinstance(session.events[-2].data, CheckpointStartedData)
    assert isinstance(session.events[-1].data, CheckpointFailedData)
    session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [asyncio.CancelledError, KeyboardInterrupt])
async def test_checkpoint_base_exception_after_start_records_terminal_before_retry(
    tmp_path: Path,
    failure_type: type[BaseException],
) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(first_failure=failure_type()),
    )

    with pytest.raises(failure_type):
        await coordinator.checkpoint(through_event_id=3)

    assert coordinator.last_checkpoint_attempt_started is True
    assert [type(event.data) for event in session.events[-2:]] == [
        CheckpointStartedData,
        CheckpointFailedData,
    ]

    checkpoint = await coordinator.checkpoint(through_event_id=3)

    assert checkpoint.version == 1
    assert len(
        [event for event in session.events if isinstance(event.data, CheckpointFailedData)]
    ) == 1
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_terminal_append_failure_poisoned_until_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(first_failure=asyncio.CancelledError()),
    )

    def fail_terminal_append(**_kwargs: object) -> object:
        raise OSError("synthetic terminal append failure")

    monkeypatch.setattr(session, "append_checkpoint_failed", fail_terminal_append)

    with pytest.raises(asyncio.CancelledError):
        await coordinator.checkpoint(through_event_id=3)
    with pytest.raises(CheckpointRecoveryRequiredError):
        await coordinator.checkpoint(through_event_id=3)

    assert isinstance(session.events[-1].data, CheckpointStartedData)
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    assert isinstance(resumed.events[-2].data, CheckpointFailedData)
    assert resumed.events[-2].data.error_code == "checkpoint_interrupted"
    assert resumed.events[-1].data.kind == "session_resumed"
    resumed.close()


@pytest.mark.asyncio
async def test_checkpoint_start_append_interruption_poisoned_until_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(),
    )
    original_append_start = session.append_checkpoint_started

    def append_start_then_raise(**kwargs: object) -> object:
        original_append_start(**kwargs)  # type: ignore[arg-type]
        raise OSError("synthetic post-write start failure")

    monkeypatch.setattr(session, "append_checkpoint_started", append_start_then_raise)

    with pytest.raises(OSError, match="synthetic post-write start failure"):
        await coordinator.checkpoint(through_event_id=3)
    with pytest.raises(CheckpointRecoveryRequiredError):
        await coordinator.checkpoint(through_event_id=3)

    assert isinstance(session.events[-1].data, CheckpointStartedData)
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    assert isinstance(resumed.events[-2].data, CheckpointFailedData)
    assert resumed.events[-1].data.kind == "session_resumed"
    resumed.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_type", [asyncio.CancelledError, KeyboardInterrupt])
async def test_checkpoint_post_commit_base_exception_is_preserved_and_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(),
    )

    def fail_activation(_registration: CheckpointCommittedData) -> Checkpoint:
        raise failure_type()

    monkeypatch.setattr(store, "activate", fail_activation)

    with pytest.raises(failure_type):
        await coordinator.checkpoint(through_event_id=3)
    with pytest.raises(CheckpointRecoveryRequiredError):
        await coordinator.checkpoint(through_event_id=3)

    assert [type(event.data) for event in session.events[-2:]] == [
        CheckpointStartedData,
        CheckpointCommittedData,
    ]
    session.close()


@pytest.mark.asyncio
async def test_checkpoint_activation_failure_requires_resume_without_duplicate_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(),
    )

    def fail_activation(_registration: CheckpointCommittedData) -> Checkpoint:
        raise OSError("synthetic activation failure")

    monkeypatch.setattr(store, "activate", fail_activation)

    with pytest.raises(CheckpointRecoveryRequiredError, match="requires resuming"):
        await coordinator.checkpoint(through_event_id=3)
    with pytest.raises(CheckpointRecoveryRequiredError, match="requires resuming"):
        await coordinator.checkpoint(through_event_id=3)

    assert [type(event.data) for event in session.events[-2:]] == [
        CheckpointStartedData,
        CheckpointCommittedData,
    ]
    committed_events = [
        event
        for event in session.events
        if isinstance(event.data, CheckpointCommittedData)
    ]
    assert len(committed_events) == 1
    assert store.current is None
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    reopened = CheckpointStore.open(
        resumed.paths.root,
        registrations=resumed.checkpoint_registrations,
        workspace_root=Path(resumed.metadata.workspace),
    )
    assert reopened.current is not None
    assert reopened.current.version == 1
    resumed.append_user_message("third", turn_id=resumed.new_turn_id())
    second = await CheckpointCoordinator(
        session=resumed,
        store=reopened,
        extractor=RecordingExtractor(),
    ).checkpoint(through_event_id=resumed.events[-1].id)
    assert second.version == 2
    resumed.close()


@pytest.mark.asyncio
async def test_checkpoint_commit_write_then_raise_recovers_on_restart_and_advances_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(),
    )
    original_commit = coordinator._commit  # pyright: ignore[reportPrivateUsage]

    def commit_then_raise(record: CheckpointRecord) -> CheckpointCommittedData:
        original_commit(record)
        raise OSError("synthetic post-write commit failure")

    monkeypatch.setattr(coordinator, "_commit", commit_then_raise)

    with pytest.raises(CheckpointRecoveryRequiredError):
        await coordinator.checkpoint(through_event_id=3)

    session_id = session.metadata.session_id
    session.close()
    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    reopened = CheckpointStore.open(
        resumed.paths.root,
        registrations=resumed.checkpoint_registrations,
        workspace_root=Path(resumed.metadata.workspace),
    )
    assert reopened.current is not None
    assert reopened.current.version == 1

    resumed.append_user_message("third", turn_id=resumed.new_turn_id())
    next_coordinator = CheckpointCoordinator(
        session=resumed,
        store=reopened,
        extractor=RecordingExtractor(),
    )
    second = await next_coordinator.checkpoint(through_event_id=resumed.events[-1].id)
    assert second.version == 2
    resumed.close()


@pytest.mark.asyncio
async def test_resume_discards_staged_orphan_before_new_same_version_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after stage but before commit cannot reserve checkpoint version one."""

    session = _session(tmp_path)
    store = CheckpointStore.open(
        session.paths.root,
        registrations=(),
        workspace_root=Path(session.metadata.workspace),
    )
    coordinator = CheckpointCoordinator(
        session=session,
        store=store,
        extractor=RecordingExtractor(),
    )

    def fail_before_commit(_record: CheckpointRecord) -> CheckpointCommittedData:
        raise OSError("synthetic pre-commit crash")

    monkeypatch.setattr(coordinator, "_commit", fail_before_commit)
    with pytest.raises(CheckpointRecoveryRequiredError):
        await coordinator.checkpoint(through_event_id=3)

    durable_transcript_before_resume = session.paths.events.read_bytes()
    checkpoint_root = session.paths.root / "checkpoints"
    orphan_index_before_resume = checkpoint_root.joinpath("index.json").read_bytes()
    assert json.loads(orphan_index_before_resume)["checkpoints"]
    assert not session.checkpoint_registrations
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)
    assert resumed.paths.events.read_bytes().startswith(durable_transcript_before_resume)
    assert not resumed.checkpoint_registrations
    assert isinstance(resumed.events[-2].data, CheckpointFailedData)
    assert resumed.events[-2].data.error_code == "checkpoint_interrupted"
    assert json.loads(checkpoint_root.joinpath("index.json").read_text())["checkpoints"] == []
    assert not tuple(checkpoint_root.glob("checkpoint-*.json"))

    third_event = resumed.append_user_message("third", turn_id=resumed.new_turn_id())
    reopened = CheckpointStore.open(
        resumed.paths.root,
        registrations=resumed.checkpoint_registrations,
        workspace_root=Path(resumed.metadata.workspace),
    )
    replacement = await CheckpointCoordinator(
        session=resumed,
        store=reopened,
        extractor=RecordingExtractor(),
    ).checkpoint(through_event_id=resumed.events[-1].id)

    assert replacement.version == 1
    assert replacement.source_through_event_id == third_event.id
    assert [
        event.data.content
        for event in resumed.events
        if isinstance(event.data, UserMessageData)
    ] == ["first", "second", "third"]
    resumed.close()


def test_resume_closes_an_interrupted_checkpoint_writer(tmp_path: Path) -> None:
    session = _session(tmp_path)
    checkpoint_id = "d" * 32
    session.append_checkpoint_started(
        checkpoint_id=checkpoint_id,
        source_from_event_id=1,
        source_through_event_id=3,
    )
    session_id = session.metadata.session_id
    session.close()

    resumed = AgentSession.resume(data_dir=tmp_path / "data", session_id=session_id)

    failures = [
        event.data
        for event in resumed.events
        if isinstance(event.data, CheckpointFailedData)
    ]
    assert len(failures) == 1
    assert failures[0].checkpoint_id == checkpoint_id
    assert failures[0].error_code == "checkpoint_interrupted"
    assert any(isinstance(event.data, UserMessageData) for event in resumed.events)
    resumed.close()


@pytest.mark.asyncio
async def test_deterministic_extractor_fits_maximum_legal_goal_and_user_text(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    criteria = tuple(
        AcceptanceCriterion(
            criterion_id=f"criterion-{index}",
            description=f"criterion-{index}-" + "x" * (512 - len(f"criterion-{index}-")),
            evidence_kind=EvidenceKind.ASSISTANT_RESPONSE,
        )
        for index in range(16)
    )
    objective = "objective-" + "o" * (4_096 - len("objective-"))
    goal = session.create_goal(objective=objective, acceptance_criteria=criteria)
    goal_event_id = session.events[-1].id
    user_event = session.append_user_message(
        "user-" + "u" * (4_096 - len("user-")),
        turn_id=session.new_turn_id(),
    )
    extractor = DeterministicCheckpointExtractor(maximum_bytes=4_096)
    first = await extractor.extract(
        previous=None,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=session.events[-1].id,
        checkpoint_id="e" * 32,
        version=1,
        focus=None,
    )
    second = await extractor.extract(
        previous=None,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=session.events[-1].id,
        checkpoint_id="e" * 32,
        version=1,
        focus=None,
    )

    assert first == second
    assert len(first.model_dump_json().encode("utf-8")) <= 4_096
    assert first.acceptance_criteria
    assert all(item.text.startswith("criterion-") for item in first.acceptance_criteria)
    assert all(item.source_event_ids == (goal_event_id,) for item in first.acceptance_criteria)
    assert first.active_work
    assert first.active_work[0].source_event_ids == (user_event.id,)
    assert goal.goal_id
    session.close()


@pytest.mark.asyncio
async def test_deterministic_extractor_counts_multibyte_text_by_utf8_bytes(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    criteria = tuple(
        AcceptanceCriterion(
            criterion_id=f"multi-{index}",
            description="标准" * 256,
            evidence_kind=EvidenceKind.ASSISTANT_RESPONSE,
        )
        for index in range(16)
    )
    session.create_goal(
        objective="目标" * 2_048,
        acceptance_criteria=criteria,
    )
    session.append_user_message("输入" * 2_048, turn_id=session.new_turn_id())

    checkpoint = await DeterministicCheckpointExtractor(maximum_bytes=4_096).extract(
        previous=None,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=session.events[-1].id,
        checkpoint_id="f" * 32,
        version=1,
        focus=None,
    )

    assert len(checkpoint.model_dump_json().encode("utf-8")) <= 4_096
    assert checkpoint.current_intent.startswith("目")
    assert checkpoint.acceptance_criteria
    assert all(
        item.text.startswith("標") or item.text.startswith("标")
        for item in checkpoint.acceptance_criteria
    )
    session.close()


def test_fit_checkpoint_handles_multifield_model_like_previous_state_at_minimum_budget() -> None:
    item = CheckpointItem(text="多字段" * 512, source_event_ids=(1,))
    fields = {
        field: (item,) * 5
        for field in (
            "acceptance_criteria",
            "constraints_and_preferences",
            "task_tree",
            "completed",
            "active_work",
            "blocked",
            "next_actions",
            "relevant_files",
            "cross_task_findings",
            "errors_and_fixes",
            "runtime_state",
            "key_decisions",
            "artifact_references",
            "miscellaneous_notes",
        )
    }
    checkpoint = Checkpoint(
        checkpoint_id="a" * 32,
        session_id="b" * 32,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=1,
        current_intent="意图" * 2_048,
        writer_model="模型" * 1_000,
        created_at=datetime.now(UTC),
    ).model_copy(update=fields)

    fitted = _fit_checkpoint(checkpoint, 4_096)

    assert len(fitted.model_dump_json().encode("utf-8")) <= 4_096
    assert fitted.current_intent
    assert fitted.writer_model


@pytest.mark.asyncio
async def test_deterministic_extractor_fits_multifield_previous_state_at_minimum_budget(
    tmp_path: Path,
) -> None:
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    item = CheckpointItem(text="历史状态" * 512, source_event_ids=(1,))
    previous = Checkpoint(
        checkpoint_id="c" * 32,
        session_id=session.metadata.session_id,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=1,
        current_intent="旧目标" * 2_048,
        acceptance_criteria=(item,) * 5,
        completed=(item,) * 5,
        active_work=(item,) * 5,
        relevant_files=(item,) * 5,
        errors_and_fixes=(item,) * 5,
        artifact_references=(item,) * 5,
        runtime_state=(item,) * 5,
        miscellaneous_notes=(item,) * 5,
        writer_model="model-like" * 512,
        created_at=datetime.now(UTC),
    )
    session.append_user_message("新输入" * 2_048, turn_id=session.new_turn_id())

    checkpoint = await DeterministicCheckpointExtractor(maximum_bytes=4_096).extract(
        previous=previous,
        events=session.events,
        source_from_event_id=2,
        source_through_event_id=session.events[-1].id,
        checkpoint_id="d" * 32,
        version=2,
        focus=None,
    )

    assert len(checkpoint.model_dump_json().encode("utf-8")) <= 4_096
    assert checkpoint.current_intent
    assert checkpoint.writer_model == "deterministic-v1"
    session.close()


@pytest.mark.asyncio
async def test_deterministic_extractor_carries_forward_model_owned_fields(
    tmp_path: Path,
) -> None:
    # The deterministic fallback cannot derive constraints, blockers, the task
    # tree, next actions, cross-task findings, or key decisions from raw events,
    # so it must carry them forward from the previous checkpoint instead of
    # silently dropping them.
    session = AgentSession.create(data_dir=tmp_path / "data", workspace=tmp_path, model="m")
    item = CheckpointItem(text="历史状态", source_event_ids=(1,))
    previous = Checkpoint(
        checkpoint_id="c" * 32,
        session_id=session.metadata.session_id,
        cycle_id=1,
        version=1,
        source_from_event_id=1,
        source_through_event_id=1,
        current_intent="旧目标",
        constraints_and_preferences=(item,),
        task_tree=(item,),
        blocked=(item,),
        next_actions=(item,),
        cross_task_findings=(item,),
        key_decisions=(item,),
        writer_model="model-like",
        created_at=datetime.now(UTC),
    )
    session.append_user_message("继续", turn_id=session.new_turn_id())

    checkpoint = await DeterministicCheckpointExtractor(maximum_bytes=65_536).extract(
        previous=previous,
        events=session.events,
        source_from_event_id=1,
        source_through_event_id=session.events[-1].id,
        checkpoint_id="d" * 32,
        version=2,
        focus=None,
    )

    assert checkpoint.constraints_and_preferences == (item,)
    assert checkpoint.task_tree == (item,)
    assert checkpoint.blocked == (item,)
    assert checkpoint.next_actions == (item,)
    assert checkpoint.cross_task_findings == (item,)
    assert checkpoint.key_decisions == (item,)
    session.close()
