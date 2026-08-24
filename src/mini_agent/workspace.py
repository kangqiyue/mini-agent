"""Canonical workspace path validation shared by local tools."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable
from pathlib import Path

from mini_agent.workspace_directory_fd import WorkspaceDirectoryAnchor

_VCS_METADATA_DIRECTORIES = frozenset({".git", ".hg", ".svn"})
_SENSITIVE_FILENAMES = frozenset(
    {
        ".dockercfg",
        ".envrc",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".terraformrc",
        ".vault-token",
        "auth.json",
        "client_secrets.json",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "id_xmss",
        "service-account.json",
        "service_account.json",
        "terraform.rc",
    }
)
_SENSITIVE_PATH_SUFFIXES = (
    (".aws", "config"),
    (".aws", "credentials"),
    (".azure", "accesstokens.json"),
    (".azure", "azureprofile.json"),
    (".config", "gcloud", "application_default_credentials.json"),
    (".config", "gh", "hosts.yml"),
    (".docker", "config.json"),
    (".kube", "config"),
    (".mini-agent", "config.toml"),
)
_SENSITIVE_SUFFIXES = frozenset(
    {".jks", ".kdbx", ".key", ".p12", ".p8", ".pfx", ".tfstate", ".tfvars"}
)
_ENVIRONMENT_TEMPLATE_FILENAMES = frozenset(
    {".env.example", ".env.sample", ".env.template"}
)


class WorkspacePathError(ValueError):
    pass


class SensitiveWorkspacePathError(WorkspacePathError):
    """A workspace path is a likely credential, state, or VCS metadata store."""


class Workspace:
    def __init__(self, root: Path, *, excluded_roots: Iterable[Path] = ()) -> None:
        try:
            resolved_root = root.expanduser().resolve(strict=True)
        except (OSError, ValueError) as error:
            raise WorkspacePathError("Could not resolve workspace") from error
        if not resolved_root.is_dir():
            raise WorkspacePathError("Workspace is not a directory")
        self.root = resolved_root
        try:
            self._directory_anchor = WorkspaceDirectoryAnchor(self.root)
        except OSError as error:
            raise WorkspacePathError("Could not inspect workspace safely") from error
        self._excluded_roots = tuple(
            self._resolve_excluded_root(excluded_root) for excluded_root in excluded_roots
        )

    @property
    def directory_anchor(self) -> WorkspaceDirectoryAnchor:
        """The root-identity-bound descriptor opener for security boundaries."""

        return self._directory_anchor

    def resolve_existing(self, relative_path: str, *, allow_directory: bool = False) -> Path:
        requested_path = self._require_relative_path(relative_path)
        self._require_not_runtime_sensitive_lexical(requested_path)
        try:
            resolved_path = (self.root / requested_path).resolve(strict=True)
        except (OSError, ValueError) as error:
            raise WorkspacePathError(
                f"Could not resolve workspace path: {relative_path}"
            ) from error
        self._require_inside_workspace(resolved_path)
        self._require_not_excluded(resolved_path)
        self._require_not_runtime_sensitive(resolved_path)

        try:
            mode = resolved_path.stat().st_mode
        except OSError as error:
            raise WorkspacePathError(
                f"Could not inspect workspace path: {relative_path}"
            ) from error
        if stat.S_ISDIR(mode):
            if not allow_directory:
                raise WorkspacePathError(f"Expected a file, got a directory: {relative_path}")
            return resolved_path
        if not stat.S_ISREG(mode):
            raise WorkspacePathError(f"Expected a regular file: {relative_path}")
        return resolved_path

    def relative(self, path: Path) -> str:
        try:
            resolved_path = path.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise WorkspacePathError("Could not resolve workspace path") from error
        self._require_inside_workspace(resolved_path)
        self._require_not_excluded(resolved_path)
        self._require_not_runtime_sensitive(resolved_path)
        return resolved_path.relative_to(self.root).as_posix()

    def has_symlink_component(self, relative_path: str) -> bool:
        """Whether any existing lexical component below the root is a symlink."""

        requested_path = self._require_relative_path(relative_path)
        self._require_not_runtime_sensitive_lexical(requested_path)
        current = self.root
        for component in requested_path.parts:
            if component == ".":
                continue
            current = current / component
            try:
                if current.is_symlink():
                    return True
            except OSError as error:
                raise WorkspacePathError(
                    f"Could not inspect workspace path: {relative_path}"
                ) from error
        return False

    def excluded_roots_below(self, directory: Path) -> tuple[Path, ...]:
        """Return excluded roots contained by a resolved workspace directory."""

        try:
            resolved_directory = directory.resolve(strict=True)
        except (OSError, ValueError) as error:
            raise WorkspacePathError("Could not resolve workspace path") from error
        self._require_inside_workspace(resolved_directory)
        return tuple(
            excluded_root
            for excluded_root in self._excluded_roots
            if excluded_root.is_relative_to(resolved_directory)
        )

    def _require_relative_path(self, value: str) -> Path:
        if not value.strip():
            raise WorkspacePathError("Workspace path cannot be empty")
        if "\x00" in value:
            raise WorkspacePathError("Workspace paths cannot contain NUL bytes")
        if len(os.fsencode(value)) > self._path_maximum():
            raise WorkspacePathError("Workspace path is too long")
        try:
            requested_path = Path(value)
        except (OSError, ValueError) as error:
            raise WorkspacePathError(f"Invalid workspace path: {value!r}") from error
        if requested_path.is_absolute():
            raise WorkspacePathError("Workspace tools require a relative path")
        return requested_path

    def _require_inside_workspace(self, path: Path) -> None:
        if not path.is_relative_to(self.root):
            raise WorkspacePathError("Path escapes workspace")

    def _require_not_excluded(self, path: Path) -> None:
        if any(path.is_relative_to(excluded_root) for excluded_root in self._excluded_roots):
            raise WorkspacePathError("Path is excluded from workspace tools")

    def _require_not_runtime_sensitive(self, path: Path) -> None:
        relative_path = path.relative_to(self.root)
        if is_runtime_sensitive_path(relative_path):
            raise SensitiveWorkspacePathError(
                "Sensitive workspace paths are unavailable to local tools"
            )

    def _require_not_runtime_sensitive_lexical(self, requested_path: Path) -> None:
        if is_runtime_sensitive_path(_normalized_lexical_relative_path(requested_path)):
            raise SensitiveWorkspacePathError(
                "Sensitive workspace paths are unavailable to local tools"
            )

    @staticmethod
    def _resolve_excluded_root(path: Path) -> Path:
        try:
            return path.expanduser().resolve(strict=False)
        except (OSError, ValueError) as error:
            raise WorkspacePathError("Could not resolve excluded workspace path") from error

    def _path_maximum(self) -> int:
        try:
            return os.pathconf(self.root, "PC_PATH_MAX")
        except (OSError, ValueError):
            return 4096


def is_runtime_sensitive_path(relative_path: Path) -> bool:
    """Classify a relative workspace path without reading the filesystem.

    This is a deliberately narrow runtime boundary for common credential,
    infrastructure-state, agent-config, and VCS metadata locations. Public
    certificate suffixes such as ``.pem`` remain eligible here; the content
    classifier used by read tools separately rejects PEM private keys.
    """

    path_parts = tuple(part.casefold() for part in relative_path.parts if part != ".")
    if not path_parts:
        return False
    filename = path_parts[-1]
    if any(part in _VCS_METADATA_DIRECTORIES for part in path_parts):
        return True
    if filename in _SENSITIVE_FILENAMES or filename == "_netrc":
        return True
    if _path_ends_with(path_parts, _SENSITIVE_PATH_SUFFIXES):
        return True
    if filename == ".env":
        return True
    if filename.startswith(".env."):
        return filename not in _ENVIRONMENT_TEMPLATE_FILENAMES
    if filename.startswith("client_secret_") and filename.endswith(".json"):
        return True
    if filename in {"client_secrets.json", "service-account.json", "service_account.json"}:
        return True
    if filename.endswith((".tfstate", ".tfvars")):
        return not filename.endswith((".example", ".sample", ".template"))
    return Path(filename).suffix.casefold() in _SENSITIVE_SUFFIXES


def _normalized_lexical_relative_path(requested_path: Path) -> Path:
    """Normalize ``.`` and inner ``..`` components without resolving symlinks."""

    normalized_parts: list[str] = []
    for part in requested_path.parts:
        if part == ".":
            continue
        if part == ".." and normalized_parts and normalized_parts[-1] != "..":
            normalized_parts.pop()
            continue
        normalized_parts.append(part)
    return Path(*normalized_parts)


def _path_ends_with(path_parts: tuple[str, ...], suffixes: tuple[tuple[str, ...], ...]) -> bool:
    return any(
        len(path_parts) >= len(suffix) and path_parts[-len(suffix) :] == suffix
        for suffix in suffixes
    )
