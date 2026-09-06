# Code review artifacts

This directory owns the review notes, verification logs, and local test scratch for this review. Source evidence lives in `src/mini_agent`, `tests`, and the Git diff.

- `README.md`: artifact map.
- `cache/original/`: initial source snapshots used to distinguish existing work.
- `cache/dist/`, `cache/wheel_install/`, `cache/smoke_workspace/`: built packages,
  installed-wheel smoke evidence, and generated example configuration (untracked).
- `cache/benchmark.json`: deterministic installed-wheel benchmark results.
- Other `cache/` subdirectories: task-local test and tool caches (untracked).
- `*.log`: verification output (untracked).
- `review.md`: confirmed findings, fixes, and validation.
- `.gitignore`: excludes generated scratch and logs.
