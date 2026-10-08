"""Resolve repository-external storage paths at runtime."""

from __future__ import annotations

import os
from pathlib import Path

STORAGE_ENVIRONMENT_VARIABLE = "GUI_CAPTCHA_STORAGE_ROOT"
DEFAULT_STORAGE_ROOT = Path(
    str(Path(__file__).resolve().parents[3] / '')
)
STORAGE_CATEGORIES = (
    "data",
    "checkpoints",
    "artifacts",
    "runs",
    "models",
    "cache",
    "third_party",
    "archives",
    "quarantine",
    "migration",
)
RUNTIME_STORAGE_CATEGORIES = frozenset(
    {"data", "checkpoints", "runs", "models", "cache", "third_party"}
)
_LEGACY_RUNTIME_RESOURCE_PREFIXES = frozenset(
    {("artifacts", "datasets"), ("artifacts", "checkpoints")}
)


class StorageConfigurationError(RuntimeError):
    """Raised when repository-external storage is not configured safely."""


def storage_root() -> Path:
    """Return the configured storage root for non-checkpoint resources."""

    value = os.environ.get(STORAGE_ENVIRONMENT_VARIABLE)
    if not value:
        return DEFAULT_STORAGE_ROOT
    root = Path(value)
    if not root.is_absolute():
        raise StorageConfigurationError(
            f"GUI_CAPTCHA_STORAGE_ROOT must be absolute, got: {value!r}"
        )
    return root


def checkpoint_root() -> Path:
    """Return the immutable root for checkpoints created by future runs."""

    return storage_root() / "checkpoints"


def storage_path(category: str, *parts: str) -> Path:
    """Return a safe path within one fixed storage category."""

    if category not in STORAGE_CATEGORIES:
        allowed = ", ".join(STORAGE_CATEGORIES)
        raise StorageConfigurationError(
            f"unknown storage category {category!r}; choose one of: {allowed}"
        )

    safe_parts: list[str] = []
    for part in parts:
        candidate = Path(part)
        if candidate.is_absolute():
            raise StorageConfigurationError(
                f"storage path component must be relative, got: {part!r}"
            )
        if ".." in candidate.parts:
            raise StorageConfigurationError(
                f"storage path component must not traverse upward, got: {part!r}"
            )
        safe_parts.append(part)
    root = checkpoint_root() if category == "checkpoints" else storage_root() / category
    return root / Path(*safe_parts)


def storage_uri_to_path(uri: str | os.PathLike[str]) -> Path:
    """Resolve a storage-relative URI like ``data/domain/file.json``."""

    path = Path(uri)
    if path.is_absolute():
        raise StorageConfigurationError(
            f"storage URI must be relative, got: {str(uri)!r}"
        )
    if not path.parts:
        raise StorageConfigurationError("storage URI must not be empty")
    category, *parts = path.parts
    return storage_path(category, *parts)


def runtime_storage_uri_to_path(uri: str | os.PathLike[str]) -> Path:
    """Resolve one current runtime resource URI under external storage.

    Historical metadata may retain the broader ``storage_uri_to_path``
    vocabulary. Supported runtime code must use this stricter resolver so raw
    datasets and checkpoints cannot remain hidden under ``artifacts``.
    """

    path = Path(uri)
    if tuple(path.parts[:2]) in _LEGACY_RUNTIME_RESOURCE_PREFIXES:
        raise StorageConfigurationError(
            "legacy resource URI is not supported at runtime; use data/... "
            "or checkpoints/..."
        )
    if not path.parts or path.parts[0] not in RUNTIME_STORAGE_CATEGORIES:
        category = path.parts[0] if path.parts else ""
        allowed = ", ".join(sorted(RUNTIME_STORAGE_CATEGORIES))
        raise StorageConfigurationError(
            f"storage URI category {category!r} is not a runtime resource; "
            f"choose one of: {allowed}"
        )
    return storage_uri_to_path(path)


def ensure_storage_layout() -> Path:
    """Create the storage root and its fixed category directories."""

    root = storage_root()
    for category in STORAGE_CATEGORIES:
        (root / category).mkdir(parents=True, exist_ok=True)
    return root
