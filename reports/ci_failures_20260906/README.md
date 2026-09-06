# CI failure investigation

This task owns only this report directory. Source and workflow changes remain in
their existing repository locations.

- `README.md`: findings, upstream run links, and verification results.
- `.gitignore`: excludes local logs and caches.
- `cache/`: local reproduction output and disposable test/build data.

Upstream evidence:

- [Latest CI](https://github.com/kangqiyue/mini-agent/actions/runs/34033887924):
  all six matrix jobs stop at Ruff; the acceptance script has F401 and E501.
- [Previous CI](https://github.com/kangqiyue/mini-agent/actions/runs/34023040528):
  search tests lack ripgrep and some CLI diagnostics fail exact text assertions.
- [Previous push security](https://github.com/kangqiyue/mini-agent/actions/runs/34023040563):
  the action uses a Git range with an unavailable base revision.
- [Dependabot PR security](https://github.com/kangqiyue/mini-agent/actions/runs/34023080179):
  listing pull request commits returns HTTP 403.
- [Dependabot package updates](https://github.com/kangqiyue/mini-agent/actions/runs/34023042720):
  `tool_version_not_supported`; its managed uv is newer than the exact project
  requirement. The project now accepts that newer tool while CI stays pinned.

Changes:

- Fix both acceptance-script lint errors and run the same repository-wide Ruff
  command in the pre-commit hook as in CI.
- Install and verify ripgrep on both matrix operating systems. Include the
  manual gateway startup tests in CI's offline suite.
- Normalize ANSI styling and visual borders in diagnostic assertions. Check
  error-stream path redaction separately from the welcome panel, which
  intentionally shows the selected local workspace. Credential assertions still
  cover the complete captured output.
- Exercise both path-depth parities in the mixed JSON separator test and assert
  that the actual relative suffix survives, rather than assuming a fixed
  temporary-directory depth.
- Install a checksum-verified Gitleaks release. Scan all checked-out Git history
  and the current file tree directly, with redaction and only `contents: read`.
  No stale push base, PR API request, or automatic PR comment is involved.
- Permit newer uv versions for Dependabot while retaining CI's pinned version
  and locked dependency installation. The lockfile is unchanged.

Local validation:

- Baseline: repository-wide Ruff reproduced both errors. A wide terminal
  reproduced seven erroneous path assertions. CI-style ANSI rendering also
  exposed a highlighted-option assertion in the auto-approval tests.
- CLI: 65 tests passed with ANSI enabled at 80 columns; 8 auto-approval tests
  passed with ANSI enabled at 240 columns.
- Repository-wide Ruff and strict Pyright passed. Actionlint passed both
  workflows; its official release checksum was verified before execution.
- Both the pinned uv and Dependabot's newer uv passed `lock --check` without
  changing dependencies or the lockfile.
- The official Linux Gitleaks archive checksum was verified. Gitleaks passed
  the complete local history and a clean tracked file-tree snapshot.
- macOS / Python 3.13: the full suite passed with CI-style ANSI rendering at
  240 columns: **1049 passed, 1 opt-in provider test skipped** in 40.35 seconds.
- All pre-commit checks passed, including the new repository-wide Ruff hook.
- Linux / Python 3.12 (ARM64 container): locked installation, Ruff, and Pyright
  passed. The full suite produced **1047 passed, 2 skipped, 1 failed** in 15.56
  seconds. The failure was a Darwin-only assertion running on a Linux container
  backed by a case-insensitive macOS bind mount. Its missing platform condition
  was fixed; Linux regression then passed **3 tests with 1 Darwin-only skip**,
  and the corresponding macOS regression passed all 4 tests. The full Linux
  suite was not repeated after this test-only condition change.
- Linux source distribution and wheel builds passed. A separate wheel
  environment passed CLI help, configuration creation/loading, and all 3
  deterministic benchmark tasks. No model provider was contacted.

Raw local evidence lives only under the ignored `cache/` directory:
`baseline-cli.log`, `baseline-color.log`, `cli-ansi.log`, and
`pytest-verified.log` contain test output; `linux-run.log`, `linux-install.log`,
and `linux-pytest.log` preserve the Linux setup and full-suite output.
`linux-smoke.log` includes the platform regression and package build;
`linux-wheel-verified.log` records the final successful independent wheel
installation. `linux-benchmark.json` is its deterministic benchmark result.
`dist/` preserves the built distributions. Disposable test directories and the
Linux checkout were cleaned; downloaded tools and caches are not committed.

Remote CI is not represented as passing until the fix is pushed and GitHub
finishes its six matrix jobs and security check.
