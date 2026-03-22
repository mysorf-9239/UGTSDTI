"""
Core Registry for modular Deep Learning components.
Allows plug-and-play models, datasets, and optimizers via Hydra configurations.
"""

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._module_dict: dict[str, Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, module_name=None):
        """
        Decorator to register a class or function.
        @MODELS.register("my_cnn")
        class MyCNN(nn.Module): ...
        """

        def _register(obj):
            name = module_name if module_name is not None else obj.__name__
            if name in self._module_dict:
                from loguru import logger

                logger.warning(f"Warning: {name} already registered in {self.name}! Overwriting.")
            self._module_dict[name] = obj
            return obj

        return _register

    def get(self, name: str):
        """Retrieve a registered module by name."""
        if name not in self._module_dict:
            raise KeyError(f"{name} is not registered in {self.name}. Available: {list(self._module_dict.keys())}")
        return self._module_dict[name]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._module_dict

    def build(self, cfg, **kwargs):
        """
        Instantiate a registered module using a Hydra DictConfig.
        Expects `cfg` to have `name` and (optionally) `params` keys.
        """
        module_name = cfg.get("name")
        params = cfg.get("params", {})

        # Merge YAML params with programmatic overrides
        final_params = dict(params)
        final_params.update(kwargs)

        module_cls = self.get(module_name)
        return module_cls(**final_params)


# Instantiate the global registries
MODELS = Registry("models")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
METRICS = Registry("metrics")
LOSSES = Registry("losses")
