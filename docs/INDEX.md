# Project control map

Mini Agent keeps a small documentation control plane instead of one large,
duplicated design document.

| Source | Authoritative for |
|---|---|
| Source code, `--help`, `.mini-agent/config.example.toml`, tests, and CI | Executable behavior, fields, defaults, and release checks |
| [README](../README.md) | Installation, configuration, commands, and user-visible limits; English is the entry-point source of truth and [README.zh-CN](../README.zh-CN.md) is its maintained Chinese mirror |
| [ARCHITECTURE](ARCHITECTURE.md) | Current module boundaries, lifecycles, invariants, and test ownership |
| [PRD](PRD.md) | Chinese-language product goals, acceptance criteria, non-goals, and explicitly marked future work |
| [SECURITY](../SECURITY.md) | Threat model, vulnerability reporting, and security policy |
| [CONTRIBUTING](../CONTRIBUTING.md) | Contributor workflow and required local checks |
| [RELEASING](RELEASING.md) | Clean-history, package, and publication procedure |
| [LICENSE](../LICENSE) | Apache License 2.0 terms for this project |
| [CHANGELOG](../CHANGELOG.md) | User-visible changes by release |

When prose and executable behavior disagree, treat the executable sources as
current and fix the prose in the same change. Architecture decisions do not
need a separate ADR log until there are competing implementations or a decision
that cannot be understood from the code and architecture document.

Maintainers may keep an ignored local handoff/progress note. It is not part of
the public release tree and must never contain credentials or private company
identity.
