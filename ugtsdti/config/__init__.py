"""Config subsystem for UGTSDTI framework.

Processing flow:
  load YAML -> resolve extends -> validate required sections
  -> validate schema -> validate cross-sections -> normalize -> NormalizedConfig

REQ-CONF-001, REQ-CONF-002, REQ-CONF-003
"""
from ugtsdti.config.loader import ConfigLoader
from ugtsdti.config.models import NormalizedConfig
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator

__all__ = [
    "ConfigLoader",
    "ConfigNormalizer",
    "ConfigValidator",
    "NormalizedConfig",
]
