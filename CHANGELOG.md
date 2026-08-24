# Changelog

All notable user-visible changes will be recorded here.

## 0.1.0rc1 - Unreleased

- Pinned the build backend to `hatchling==1.32.0` and its editable-install
  helper to `editables==0.6`, and pinned `uv==0.11.28`, standardizing locked,
  no-build-isolation development and release builds.
- Licensed the project under Apache License 2.0 with `kangqiyue` as the public
  copyright holder.
- Added an installable Python 3.12+ CLI with a complete workspace configuration
  template and safe `init-config` command.
- Added lazy session creation, workspace-bound resume, interactive session
  selection, and conversation replay from append-only events.
- Added bounded context projection, incremental checkpoints, deterministic
  fallback, context rebuild, durable goals, evaluation, and benchmark flows.
- Added workspace-scoped read, search, and patch tools plus approval-gated
  explicit-argument command execution.
- Added conservative crash recovery for interrupted model, tool, checkpoint,
  rebuild, and artifact lifecycles without replaying unknown side effects.
- Reconciled uncommitted staged checkpoints on resume so a crash cannot reserve
  the next checkpoint version, while preserving every transcript-committed
  checkpoint.
- Added an explicit event schema v2 with read-time migration for published v1
  sessions and strict validation for newly written records.
- Added credential and known host-path redaction at persistence and provider
  boundaries, staged/history secret gates, and clean-root release checks.
- Anchored `init-config` directory creation and writes to no-follow workspace
  descriptors, and made provider/session/writer cleanup failures observable as
  nonzero CLI outcomes.
- Added trusted-base pull-request scanning plus a local, redacted private
  denylist check; candidate code and GitHub workflows never receive the private
  terms.
- Added a practical clean-root release checklist with package build,
  installation smoke tests, history scanning, and exact commit/tag pushes.
- Added OpenAI-compatible loopback support, bounded provider responses,
  retry/overflow budgets, and macOS/Linux wheel validation.
