from pathlib import Path

import pytest

from mini_agent.benchmark import (
    run_reproducible_benchmark,
    run_single_benchmark_task,
    write_benchmark_report,
)


@pytest.mark.asyncio
async def test_three_synthetic_coding_tasks_produce_a_reproducible_report(
    tmp_path: Path,
) -> None:
    report = await run_reproducible_benchmark()

    assert report.task_count == 3
    assert report.passed_task_count == 3
    assert report.success_rate == 1.0
    assert all(task.success for task in report.tasks)
    assert all(task.evaluation.cycle_count >= 4 for task in report.tasks)
    assert all(task.evaluation.retrieval_success == 1.0 for task in report.tasks)
    assert all(task.evaluation.false_completion_rate == 0.0 for task in report.tasks)
    assert all(task.evaluation.passed_acceptance_thresholds for task in report.tasks)

    output = tmp_path / "benchmark.json"
    write_benchmark_report(report, output)
    assert output.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.asyncio
async def test_tool_failure_is_reported_and_does_not_produce_a_false_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_patch(self: object, arguments_json: str) -> object:
        del self, arguments_json
        from mini_agent.tools.base import ToolError

        raise ToolError("synthetic_patch_failure", "injected")

    monkeypatch.setattr("mini_agent.tools.apply_patch.ApplyPatchTool.execute", fail_patch)

    with pytest.raises(RuntimeError, match="did not complete patch and command"):
        await run_single_benchmark_task("fix_increment")
