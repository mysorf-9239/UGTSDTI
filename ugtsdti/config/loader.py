"""ConfigLoader — load YAML and resolve `extends` deterministically.

REQ-CONF-001, REQ-CONF-003
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from ugtsdti.core.errors import InvalidConfigError

# ---------------------------------------------------------------------------
# Deep merge helper
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge *override* into *base* recursively.

    Later (override) values win for scalar keys.
    For dict values, merge recursively.
    For list values, override replaces base entirely.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Load a YAML config file and resolve ``extends`` deterministically.

    Resolution order for ``extends``:
    - entries are processed left-to-right;
    - each subsequent entry overrides the previous;
    - the main config is applied last (highest priority).

    This is deterministic: given the same list of base files and the same
    main config, the result is always identical.

    Args:
        search_paths: Optional list of directories to search for base configs
                      referenced in ``extends``.  The directory containing the
                      main config file is always searched first.
    """

    def __init__(self, search_paths: list[Path] | None = None) -> None:
        self._search_paths: list[Path] = search_paths or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Path | str) -> dict[str, Any]:
        """Load *path* and resolve ``extends``, returning a merged raw dict.

        Args:
            path: Path to the main YAML config file.

        Returns:
            Merged raw config dict with ``extends`` resolved.

        Raises:
            InvalidConfigError: If the file cannot be read, is not valid YAML,
                                 or a base config referenced in ``extends``
                                 cannot be found.
        """
        path = Path(path)
        raw = self._load_yaml(path)

        # Resolve extends before returning
        search_dirs = [path.parent] + self._search_paths
        resolved = self._resolve_extends(raw, search_dirs, visited=set())
        return resolved

    def load_dict(self, raw: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
        """Resolve ``extends`` in an already-loaded raw dict.

        Useful when the caller has already parsed YAML (e.g. from a string).

        Args:
            raw:      Already-parsed config dict.
            base_dir: Directory to use when resolving relative ``extends`` paths.

        Returns:
            Merged raw config dict with ``extends`` resolved.
        """
        search_dirs = ([base_dir] if base_dir else []) + self._search_paths
        return self._resolve_extends(raw, search_dirs, visited=set())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Read and parse a YAML file."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InvalidConfigError(
                f"Cannot read config file: {path}",
                stage="config_load",
                key=str(path),
            ) from exc

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise InvalidConfigError(
                f"Invalid YAML in config file {path}: {exc}",
                stage="config_load",
                key=str(path),
            ) from exc

        if not isinstance(data, dict):
            raise InvalidConfigError(
                f"Config file {path} must be a YAML mapping at the top level.",
                stage="config_load",
                key=str(path),
            )
        return data

    def _resolve_extends(
        self,
        raw: dict[str, Any],
        search_dirs: list[Path],
        visited: set[str],
    ) -> dict[str, Any]:
        """Recursively resolve ``extends`` entries.

        Processing order:
        1. Start with an empty base.
        2. For each entry in ``extends`` (left-to-right), load and merge it.
        3. Apply the current config on top (highest priority).

        This ensures later entries in ``extends`` override earlier ones, and
        the main config always wins.
        """
        extends_list = raw.get("extends", [])
        if not extends_list:
            return raw

        # Normalise to list of strings
        if isinstance(extends_list, str):
            extends_list = [extends_list]

        merged: dict[str, Any] = {}

        for entry in extends_list:
            base_path = self._find_base(entry, search_dirs)
            canonical = str(base_path.resolve())
            if canonical in visited:
                raise InvalidConfigError(
                    f"Circular extends detected: {entry!r} already in resolution chain.",
                    stage="config_load",
                    key=entry,
                )
            visited_copy = visited | {canonical}
            base_raw = self._load_yaml(base_path)
            base_resolved = self._resolve_extends(base_raw, [base_path.parent] + search_dirs, visited_copy)
            merged = _deep_merge(merged, base_resolved)

        # Main config wins over all bases; strip 'extends' from result
        main_without_extends = {k: v for k, v in raw.items() if k != "extends"}
        result = _deep_merge(merged, main_without_extends)
        return result

    def _find_base(self, entry: str, search_dirs: list[Path]) -> Path:
        """Locate a base config file referenced in ``extends``.

        Tries:
        1. Absolute path as-is.
        2. ``<entry>.yaml`` and ``<entry>.yml`` in each search directory.
        3. ``<entry>`` as-is in each search directory.
        """
        candidate = Path(entry)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        for directory in search_dirs:
            for suffix in ("", ".yaml", ".yml"):
                p = directory / (entry + suffix)
                if p.exists():
                    return p

        raise InvalidConfigError(
            f"Cannot find base config {entry!r} in search paths: " + ", ".join(str(d) for d in search_dirs),
            stage="config_load",
            key=entry,
        )
