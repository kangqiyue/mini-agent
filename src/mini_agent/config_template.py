"""Install the packaged example as a workspace configuration."""

from __future__ import annotations

import os
import stat
from importlib import resources
from pathlib import Path

from mini_agent.workspace_directory_fd import (
    DirectoryIdentity,
    OpenWorkspaceParent,
    WorkspaceDirectoryAnchor,
    WorkspaceDirectoryFdError,
)

_CONFIG_DIRECTORY_NAME = ".mini-agent"
_CONFIG_FILENAME = "config.toml"
_TEMPLATE_RESOURCE_NAME = "config.example.toml"
_MAXIMUM_TEMPLATE_BYTES = 128 * 1024


class ConfigInitializationError(RuntimeError):
    """The example configuration could not be installed safely."""


class ConfigAlreadyExistsError(ConfigInitializationError):
    """The destination configuration already exists and was preserved."""


def read_packaged_config_template() -> str:
    """Return the wheel resource, with a source-checkout fallback for editable installs."""

    resource = resources.files("mini_agent").joinpath(_TEMPLATE_RESOURCE_NAME)
    try:
        template = resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        template = _read_source_checkout_template()
    except (OSError, UnicodeError) as error:
        raise ConfigInitializationError(
            "The packaged configuration template could not be read."
        ) from error
    _validate_template_text(template)
    return template


def initialize_workspace_config(workspace: Path) -> Path:
    """Create `.mini-agent/config.toml` once through a held workspace anchor.

    A resolved pathname is only an initial candidate.  The actual create and
    write operations are all relative to held, no-follow directory descriptors.
    The route is re-walked before and after the write so a root or configuration
    directory exchange never produces a successful result for a stale object.
    """

    template = read_packaged_config_template().encode("utf-8")
    resolved_workspace = _resolve_workspace(workspace)
    try:
        anchor = WorkspaceDirectoryAnchor(resolved_workspace)
        initial_config_identity = _create_config_directory_through_anchor(anchor)
        _write_config_through_current_directory(
            anchor,
            template,
            initial_config_identity=initial_config_identity,
        )
    except ConfigInitializationError:
        raise
    except (WorkspaceDirectoryFdError, OSError, ValueError) as error:
        raise ConfigInitializationError(
            "The workspace changed while the configuration was initialized."
        ) from error
    return resolved_workspace / _CONFIG_DIRECTORY_NAME / _CONFIG_FILENAME


def _read_source_checkout_template() -> str:
    source_template = (
        Path(__file__).parents[2] / _CONFIG_DIRECTORY_NAME / _TEMPLATE_RESOURCE_NAME
    )
    try:
        return source_template.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfigInitializationError(
            "The configuration template is unavailable in this installation."
        ) from error


def _validate_template_text(template: str) -> None:
    encoded_template = template.encode("utf-8")
    if not encoded_template or len(encoded_template) > _MAXIMUM_TEMPLATE_BYTES:
        raise ConfigInitializationError("The packaged configuration template has an invalid size.")
    if "\0" in template:
        raise ConfigInitializationError("The packaged configuration template contains NUL bytes.")


def _resolve_workspace(workspace: Path) -> Path:
    try:
        resolved_workspace = workspace.expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigInitializationError("The workspace could not be resolved safely.") from error
    if not resolved_workspace.is_dir():
        raise ConfigInitializationError("The workspace must be an existing directory.")
    return resolved_workspace


def _create_config_directory_through_anchor(
    anchor: WorkspaceDirectoryAnchor,
) -> DirectoryIdentity:
    """Create the config directory below the held root and remember new identity."""

    initial_config_identity: DirectoryIdentity
    try:
        with anchor.open_existing_directory(".") as workspace_descriptor:
            is_new_directory = _create_config_directory_at(workspace_descriptor)
            config_descriptor = _open_config_directory_at(workspace_descriptor)
            try:
                initial_config_identity = DirectoryIdentity.from_stat_result(
                    os.fstat(config_descriptor)
                )
            finally:
                os.close(config_descriptor)
            if is_new_directory:
                _sync_directory_descriptor(
                    workspace_descriptor,
                    "The workspace configuration directory could not be persisted durably.",
                )
    except ConfigInitializationError:
        raise
    except (WorkspaceDirectoryFdError, OSError, ValueError) as error:
        raise ConfigInitializationError(
            "The workspace configuration directory could not be created safely."
        ) from error
    return initial_config_identity


def _create_config_directory_at(workspace_descriptor: int) -> bool:
    """Create one direct no-follow child of a held workspace directory."""

    try:
        directory_status = os.stat(
            _CONFIG_DIRECTORY_NAME,
            dir_fd=workspace_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        directory_status = None
    except (NotImplementedError, TypeError) as error:
        raise WorkspaceDirectoryFdError(
            "Safe workspace directory descriptors are unavailable"
        ) from error

    if directory_status is not None and stat.S_ISLNK(directory_status.st_mode):
        raise ConfigInitializationError(
            "The workspace configuration directory must not be a symbolic link."
        )

    try:
        os.mkdir(_CONFIG_DIRECTORY_NAME, mode=0o700, dir_fd=workspace_descriptor)
        return True
    except FileExistsError:
        return False
    except (NotImplementedError, TypeError) as error:
        raise WorkspaceDirectoryFdError(
            "Safe workspace directory descriptors are unavailable"
        ) from error


def _open_config_directory_at(workspace_descriptor: int) -> int:
    flags = _directory_open_flags()
    try:
        descriptor = os.open(
            _CONFIG_DIRECTORY_NAME,
            flags,
            dir_fd=workspace_descriptor,
        )
    except (NotImplementedError, TypeError) as error:
        raise WorkspaceDirectoryFdError(
            "Safe workspace directory descriptors are unavailable"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ConfigInitializationError(
                "The workspace configuration directory is not a safe directory."
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_config_through_current_directory(
    anchor: WorkspaceDirectoryAnchor,
    template: bytes,
    *,
    initial_config_identity: DirectoryIdentity,
) -> None:
    """Write through a held parent after proving its route has not changed."""

    try:
        with anchor.open_existing_parent(
            f"{_CONFIG_DIRECTORY_NAME}/{_CONFIG_FILENAME}"
        ) as opened_parent:
            if opened_parent.parent_identity != initial_config_identity:
                raise ConfigInitializationError(
                    "The workspace configuration directory changed during initialization."
                )
            _verify_current_parent_route(anchor, opened_parent)
            _write_and_verify_config(anchor, opened_parent, template)
    except ConfigInitializationError:
        raise
    except (WorkspaceDirectoryFdError, OSError) as error:
        raise ConfigInitializationError(
            "The workspace changed while the configuration was initialized."
        ) from error


def _write_and_verify_config(
    anchor: WorkspaceDirectoryAnchor,
    opened_parent: OpenWorkspaceParent,
    template: bytes,
) -> None:
    """Write once and remove it again if the post-write route is stale."""

    _write_new_config(opened_parent.parent_descriptor, template)
    try:
        _verify_current_parent_route(anchor, opened_parent)
    except ConfigInitializationError as route_error:
        try:
            _remove_config_after_route_change(opened_parent.parent_descriptor)
        except ConfigInitializationError as cleanup_error:
            raise ConfigInitializationError(
                "The workspace configuration changed and cleanup could not be completed."
            ) from cleanup_error
        raise route_error


def _remove_config_after_route_change(directory_descriptor: int) -> None:
    """Remove a newly-created config through its held directory after a stale route."""

    try:
        os.unlink(_CONFIG_FILENAME, dir_fd=directory_descriptor)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ConfigInitializationError(
            "The changed workspace configuration could not be removed safely."
        ) from error
    _sync_directory_descriptor(
        directory_descriptor,
        "The changed workspace configuration could not be persisted durably.",
    )


def _verify_current_parent_route(
    anchor: WorkspaceDirectoryAnchor, opened_parent: OpenWorkspaceParent
) -> None:
    descriptor = -1
    try:
        # ``reopen_parent`` deliberately descends from the held root descriptor.
        # Also open the root by its current pathname to detect replacement of the
        # root itself before relying on that held descriptor.
        with anchor.open_existing_directory("."):
            pass
        descriptor = anchor.reopen_parent(opened_parent)
    except (WorkspaceDirectoryFdError, OSError) as error:
        raise ConfigInitializationError(
            "The workspace configuration directory changed during initialization."
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _directory_open_flags() -> int:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    if directory_flag is None or nofollow_flag is None:
        raise WorkspaceDirectoryFdError(
            "Safe workspace directory descriptors are unavailable"
        )
    return os.O_RDONLY | directory_flag | nofollow_flag | getattr(os, "O_CLOEXEC", 0)


def _sync_directory_descriptor(descriptor: int, error_message: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise ConfigInitializationError(error_message) from error


def _write_new_config(directory_descriptor: int, template: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        config_descriptor = os.open(
            _CONFIG_FILENAME,
            flags,
            0o600,
            dir_fd=directory_descriptor,
        )
    except FileExistsError as error:
        raise ConfigAlreadyExistsError(
            "The workspace configuration already exists and was not overwritten."
        ) from error
    except OSError as error:
        raise ConfigInitializationError(
            "The workspace configuration could not be created safely."
        ) from error

    try:
        with os.fdopen(config_descriptor, "wb") as config_file:
            config_file.write(template)
            config_file.flush()
            os.fsync(config_file.fileno())
        os.fsync(directory_descriptor)
    except OSError as error:
        _remove_incomplete_config(directory_descriptor)
        raise ConfigInitializationError(
            "The workspace configuration could not be written durably."
        ) from error


def _remove_incomplete_config(directory_descriptor: int) -> None:
    try:
        os.unlink(_CONFIG_FILENAME, dir_fd=directory_descriptor)
    except FileNotFoundError:
        return
    except OSError as error:
        raise ConfigInitializationError(
            "The incomplete workspace configuration could not be removed."
        ) from error
