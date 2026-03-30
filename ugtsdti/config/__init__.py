"""Config subsystem for UGTSDTI framework.

Processing flow:
  load YAML -> resolve extends -> validate required sections
  -> validate schema -> validate cross-sections -> normalize -> NormalizedConfig

REQ-CONF-001, REQ-CONF-002, REQ-CONF-003
"""

from __future__ import annotations

__all__ = [
    "ConfigLoader",
    "ConfigNormalizer",
    "ConfigValidator",
    "NormalizedConfig",
]


def __getattr__(name: str) -> object:
    """Lazy imports so lightweight validators do not require YAML at import time."""
    if name == "ConfigLoader":
        from ugtsdti.config.loader import ConfigLoader

        return ConfigLoader
    if name == "ConfigNormalizer":
        from ugtsdti.config.normalize import ConfigNormalizer

        return ConfigNormalizer
    if name == "ConfigValidator":
        from ugtsdti.config.validate import ConfigValidator

        return ConfigValidator
    if name == "NormalizedConfig":
        from ugtsdti.config.models import NormalizedConfig

        return NormalizedConfig
    raise AttributeError(name)
