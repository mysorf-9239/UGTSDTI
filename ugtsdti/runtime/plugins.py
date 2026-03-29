"""Helpers for config-driven runtime plugin registration."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, cast

from ugtsdti.core.errors import InvalidConfigError

RuntimeRegistrar = Callable[[Any, Any], None]


def apply_runtime_registrars(
    registrars: list[str] | tuple[str, ...],
    *,
    graph_registry: Any,
    interaction_registry: Any,
) -> None:
    """Import and apply config-declared runtime registrars."""
    for dotted_path in registrars:
        registrar = _load_registrar(str(dotted_path))
        registrar(graph_registry, interaction_registry)


def _load_registrar(dotted_path: str) -> RuntimeRegistrar:
    module_path, separator, attr_name = dotted_path.rpartition(".")
    if not separator or not module_path or not attr_name:
        raise InvalidConfigError(
            f"Runtime registrar path must be '<module>.<callable>', got {dotted_path!r}.",
            stage="runtime",
            component="plugins",
            key="runtime.plugin_registrars",
        )
    try:
        module = import_module(module_path)
    except Exception as exc:
        raise InvalidConfigError(
            f"Could not import runtime registrar module {module_path!r}.",
            stage="runtime",
            component="plugins",
            key=dotted_path,
        ) from exc

    registrar = getattr(module, attr_name, None)
    if not callable(registrar):
        raise InvalidConfigError(
            f"Runtime registrar {dotted_path!r} is not callable.",
            stage="runtime",
            component="plugins",
            key=dotted_path,
        )
    return cast(RuntimeRegistrar, registrar)
