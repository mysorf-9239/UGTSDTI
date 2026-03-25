"""Helpers for normalizing Hydra configs into executable plugin selections."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from omegaconf import DictConfig, OmegaConf


def bootstrap_registries() -> None:
    """Import plugin packages so registry decorators execute before build()."""
    import_module("ugtsdti.models")
    import_module("ugtsdti.data")
    import_module("ugtsdti.losses")


def cfg_to_container(cfg: Any) -> Any:
    """Resolve OmegaConf objects into plain Python containers when needed."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return dict(cfg)
    if isinstance(cfg, list):
        return list(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def normalize_plugin_cfg(plugin_cfg: Any) -> dict[str, Any] | None:
    """Normalize a plugin group into the registry ``name``/``params`` schema."""
    container = cfg_to_container(plugin_cfg)
    if container is None:
        return None
    if not isinstance(container, dict):
        raise TypeError(f"Plugin config must resolve to a mapping, got: {type(container)!r}")
    if container.get("name") in {None, ""}:
        return None
    return {
        "name": container["name"],
        "params": dict(container.get("params", {})),
    }


def resolve_model_cfg(cfg: DictConfig) -> dict[str, Any]:
    """Resolve slot-based Hydra groups into the final registry model config."""
    model_cfg = cfg_to_container(cfg.model)
    if not isinstance(model_cfg, dict):
        raise TypeError("cfg.model must resolve to a mapping")

    params = dict(model_cfg.get("params", {}))
    explicit_nested_cfg = any(key in params for key in ("teacher_cfg", "student_cfg", "fusion_cfg"))
    if explicit_nested_cfg:
        return model_cfg

    params["teacher_cfg"] = normalize_plugin_cfg(getattr(cfg, "teacher", None))
    params["student_cfg"] = normalize_plugin_cfg(getattr(cfg, "student", None))
    params["fusion_cfg"] = normalize_plugin_cfg(getattr(cfg, "fusion", None))
    model_cfg["params"] = params
    return model_cfg


def resolve_loss_cfg(cfg: DictConfig, trainer_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    top_level_loss = cfg_to_container(cfg.get("loss"))
    trainer_has_explicit_loss = "loss" in cfg.trainer
    trainer_loss = cfg.trainer.get("loss", trainer_cfg.get("loss", {"name": "bce"}))
    resolved_trainer_loss = cfg_to_container(trainer_loss)
    if isinstance(top_level_loss, dict) and top_level_loss.get("name"):
        if (
            trainer_has_explicit_loss
            and isinstance(resolved_trainer_loss, dict)
            and resolved_trainer_loss.get("name")
            and resolved_trainer_loss != top_level_loss
        ):
            return resolved_trainer_loss
        return top_level_loss
    if not isinstance(resolved_trainer_loss, dict):
        raise TypeError("Loss config must resolve to a mapping")
    return resolved_trainer_loss


def _plugin_alias(plugin_cfg: Any) -> str:
    """Return a stable label for reporting and benchmark artifacts."""
    container = cfg_to_container(plugin_cfg)
    if not isinstance(container, dict) or container.get("name") in {None, ""}:
        return "none"
    return str(container.get("alias") or container.get("name"))


def infer_model_label(cfg: DictConfig) -> str:
    """Infer a human-readable model label for reporting and artifacts."""
    explicit_name = cfg.get("model_name")
    if explicit_name:
        return str(explicit_name)

    model_cfg = cfg_to_container(cfg.model)
    params = model_cfg.get("params", {}) if isinstance(model_cfg, dict) else {}
    if any(key in params for key in ("teacher_cfg", "student_cfg", "fusion_cfg")):
        return str(model_cfg.get("alias") or model_cfg.get("name") or "model")

    model_alias = str(model_cfg.get("alias") or model_cfg.get("name") or "model")
    return ".".join(
        [
            model_alias,
            _plugin_alias(getattr(cfg, "teacher", None)),
            _plugin_alias(getattr(cfg, "student", None)),
            _plugin_alias(getattr(cfg, "fusion", None)),
        ]
    )


def inject_dataset_aware_model_params(model_cfg: dict[str, Any], train_dataset) -> None:
    """Inject dataset-dependent parameters required by selected plugins."""
    params = model_cfg.get("params", {})
    teacher_cfg = params.get("teacher_cfg")
    if not teacher_cfg:
        return

    if teacher_cfg.get("name") == "baseline_teacher":
        teacher_params = teacher_cfg.setdefault("params", {})
        if getattr(train_dataset, "num_unique_drugs", None) is not None:
            teacher_params["num_drugs"] = train_dataset.num_unique_drugs
        if getattr(train_dataset, "num_unique_targets", None) is not None:
            teacher_params["num_targets"] = train_dataset.num_unique_targets
