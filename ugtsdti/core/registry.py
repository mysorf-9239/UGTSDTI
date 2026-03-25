"""Lightweight registries for slot-based UGTSDTI plugins.

The project composes models, datasets, transforms, metrics, and losses through
Hydra config groups plus explicit registry lookup. The registry intentionally
stays minimal: it validates names, merges config params with runtime overrides,
and instantiates the selected class or callable.
"""

from typing import Any, Callable


class Registry:
    """Name-to-callable registry used by the experiment builders."""

    def __init__(self, name: str):
        self._name = name
        self._module_dict: dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, module_name=None):
        """Register a class or function under an explicit or inferred name."""

        def _register(obj):
            name = module_name if module_name is not None else obj.__name__
            if name in self._module_dict:
                from loguru import logger

                logger.warning(f"Warning: {name} already registered in {self.name}! Overwriting.")
            self._module_dict[name] = obj
            return obj

        return _register

    def get(self, name: str):
        """Return a previously registered callable by name."""
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self.name}. Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._module_dict

    def build(self, cfg, **kwargs):
        """Instantiate a registered callable from a ``name``/``params`` config."""
        module_name = cfg.get("name")
        params = cfg.get("params", {})

        # Runtime overrides win over config defaults.
        final_params = dict(params)
        final_params.update(kwargs)

        module_cls = self.get(module_name)
        return module_cls(**final_params)


# Global registries imported throughout the project.
MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
METRICS = Registry("metrics")
LOSSES = Registry("losses")
