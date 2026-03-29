"""ConfigNormalizer — normalize raw config dict into NormalizedConfig.

Properties:
- idempotent: normalize(normalize(cfg)) == normalize(cfg)
- serializable: NormalizedConfig.to_dict() round-trips cleanly
- stable: to_dict() output is stable enough to hash
- MUST NOT move plugins between stages
- MUST NOT infer hidden loss mappings

REQ-CONF-003
"""
from __future__ import annotations

import copy
from typing import Any

from ugtsdti.config.models import NormalizedConfig

# ---------------------------------------------------------------------------
# Default values injected during normalization
# ---------------------------------------------------------------------------

_DEFAULT_INTERACTION: dict[str, Any] = {
    "order": [],
    "dependencies": {},
}

_DEFAULT_DECISION: dict[str, Any] = {
    "type": "identity",
    "mode": "heuristic",
    "strategy": "identity",
    "use_uncertainty": False,
    "fallback": {},
}

_DEFAULT_TRAINING: dict[str, Any] = {
    "teacher": {"freeze": True},
    "student": {"freeze": False},
    "kd": {"schedule": "constant"},
    "gate": {"trainable": False},
}

_DEFAULT_LOSS: dict[str, Any] = {
    "type": "hard",
    "hard_weight": 1.0,
    "map": {},
}

_DEFAULT_METRICS: dict[str, Any] = {
    "primary": "auroc",
    "enabled": ["auroc", "auprc", "f1"],
    "by_scenario": False,
}

_DEFAULT_DIAGNOSTICS: dict[str, Any] = {
    "disagreement": False,
    "gate_alpha": False,
    "uncertainty_error": False,
}

_DEFAULT_LOGGING: dict[str, Any] = {
    "backend": "file",
    "trace_execution": False,
    "inspect_state": False,
}

_DEFAULT_RUNTIME: dict[str, Any] = {
    "device": "cpu",
    "seed": 0,
    "deterministic": False,
    "precision": "fp32",
}


# ---------------------------------------------------------------------------
# ConfigNormalizer
# ---------------------------------------------------------------------------


class ConfigNormalizer:
    """Normalize a raw config dict into a NormalizedConfig.

    Normalization:
    - fills in documented defaults for optional fields;
    - sorts list fields for stability (e.g. interaction.order is preserved as-is
      since order matters, but modalities.available is sorted);
    - does NOT move plugins between stages;
    - does NOT infer hidden loss mappings;
    - is idempotent: calling normalize twice produces the same result.
    """

    def normalize(self, raw: dict[str, Any]) -> NormalizedConfig:
        """Normalize *raw* into a NormalizedConfig.

        Args:
            raw: Raw config dict (already resolved via ConfigLoader).

        Returns:
            NormalizedConfig instance.
        """
        cfg = copy.deepcopy(raw)

        version = str(cfg.get("version", "")).strip()
        experiment = self._normalize_experiment(cfg.get("experiment", {}))
        extends = self._normalize_extends(cfg.get("extends", []))
        sweep = dict(cfg.get("sweep", {}))
        data = dict(cfg.get("data", {}))
        scenario = self._normalize_scenario(cfg.get("scenario", {}))
        modalities = self._normalize_modalities(cfg.get("modalities", {}))
        graph = self._normalize_graph(cfg.get("graph", {}))
        roles = self._normalize_roles(cfg.get("roles", {}))
        interaction = self._normalize_interaction(cfg.get("interaction", {}))
        decision = self._normalize_decision(cfg.get("decision", {}))
        training = self._normalize_training(cfg.get("training", {}))
        loss = self._normalize_loss(cfg.get("loss", {}))
        metrics = self._normalize_metrics(cfg.get("metrics", {}))
        diagnostics = self._normalize_diagnostics(cfg.get("diagnostics", {}))
        logging_ = self._normalize_logging(cfg.get("logging", {}))
        runtime = self._normalize_runtime(cfg.get("runtime", {}))

        return NormalizedConfig(
            version=version,
            experiment=experiment,
            extends=extends,
            sweep=sweep,
            data=data,
            scenario=scenario,
            modalities=modalities,
            graph=graph,
            roles=roles,
            interaction=interaction,
            decision=decision,
            training=training,
            loss=loss,
            metrics=metrics,
            diagnostics=diagnostics,
            logging=logging_,
            runtime=runtime,
        )

    # ------------------------------------------------------------------
    # Section normalizers
    # ------------------------------------------------------------------

    def _normalize_experiment(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return dict(raw)

    def _normalize_extends(self, raw: Any) -> list[str]:
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(e) for e in raw]
        return []

    def _normalize_scenario(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        result = copy.deepcopy(raw)
        # Normalize train scenario to lowercase string
        if "train" in result:
            result["train"] = str(result["train"]).lower()
        # Normalize eval scenarios to sorted lowercase list
        if "eval" in result:
            eval_val = result["eval"]
            if isinstance(eval_val, str):
                result["eval"] = [eval_val.lower()]
            elif isinstance(eval_val, list):
                result["eval"] = sorted(str(s).lower() for s in eval_val)
        return result

    def _normalize_modalities(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        result = copy.deepcopy(raw)
        # Sort available modalities for stability
        if "available" in result and isinstance(result["available"], list):
            result["available"] = sorted(result["available"])
        # Normalize teacher/student uses lists
        for branch in ("teacher", "student"):
            if branch in result and isinstance(result[branch], dict):
                uses = result[branch].get("uses", [])
                if isinstance(uses, list):
                    result[branch]["uses"] = sorted(uses)
        return result

    def _normalize_graph(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {"nodes": {}}
        result = copy.deepcopy(raw)
        nodes = result.get("nodes", {})

        if isinstance(nodes, dict):
            # Normalize each node config
            normalized_nodes: dict[str, Any] = {}
            for node_name in sorted(nodes.keys()):
                node_cfg = nodes[node_name]
                if isinstance(node_cfg, dict):
                    normalized_nodes[node_name] = self._normalize_node(node_cfg)
                else:
                    normalized_nodes[node_name] = node_cfg
            result["nodes"] = normalized_nodes
        elif isinstance(nodes, list):
            # List-style nodes: normalize each, preserve order (order matters for DAG)
            result["nodes"] = [self._normalize_node(n) if isinstance(n, dict) else n for n in nodes]

        return result

    def _normalize_node(self, node_cfg: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(node_cfg)
        # Normalize type key: prefer 'type_key', fall back to 'type'
        if "type" in result and "type_key" not in result:
            result["type_key"] = result.pop("type")
        # Sort inputs for stability
        if "inputs" in result and isinstance(result["inputs"], list):
            result["inputs"] = sorted(result["inputs"])
        # Sort output_attrs for stability
        if "output_attrs" in result and isinstance(result["output_attrs"], list):
            result["output_attrs"] = sorted(result["output_attrs"])
        return result

    def _normalize_roles(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        result: dict[str, Any] = {}
        for role_name in sorted(raw.keys()):
            role_cfg = raw[role_name]
            if not isinstance(role_cfg, dict):
                result[role_name] = role_cfg
                continue
            normalized_role = copy.deepcopy(role_cfg)
            # Normalize outputs list (preserve order — order matters for aggregation)
            outputs = normalized_role.get("outputs", [])
            if isinstance(outputs, str):
                normalized_role["outputs"] = [outputs]
            # Default aggregation
            if "aggregation" not in normalized_role:
                normalized_role["aggregation"] = "first"
            result[role_name] = normalized_role
        return result

    def _normalize_interaction(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_INTERACTION)
        result = copy.deepcopy(raw)
        # Ensure order and dependencies exist
        if "order" not in result:
            result["order"] = []
        if "dependencies" not in result:
            result["dependencies"] = {}
        # Normalize dependencies: ensure each value is a list
        deps = result["dependencies"]
        if isinstance(deps, dict):
            for mod_name, dep_list in deps.items():
                if isinstance(dep_list, str):
                    deps[mod_name] = [dep_list]
                elif not isinstance(dep_list, list):
                    deps[mod_name] = []
        # Normalize each named module config
        order = result.get("order", [])
        for mod_name in order:
            if mod_name in result and isinstance(result[mod_name], dict):
                mod_cfg = result[mod_name]
                if not isinstance(mod_cfg.get("params"), dict):
                    mod_cfg["params"] = {}
        return result

    def _normalize_decision(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_DECISION)
        result = {**_DEFAULT_DECISION, **copy.deepcopy(raw)}
        if "type" in result and str(result.get("type")) == "identity":
            result["mode"] = "heuristic"
        # Normalize fallback
        if not isinstance(result.get("fallback"), dict):
            result["fallback"] = {}
        return result

    def _normalize_training(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_TRAINING)
        result = copy.deepcopy(_DEFAULT_TRAINING)
        # Deep merge raw into defaults
        for key, val in raw.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = {**result[key], **val}
            else:
                result[key] = copy.deepcopy(val)
        return result

    def _normalize_loss(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_LOSS)
        result = {**_DEFAULT_LOSS, **copy.deepcopy(raw)}
        if not isinstance(result.get("map"), dict):
            result["map"] = {}
        # Normalize each map entry
        loss_map = result["map"]
        for loss_key, mapping in loss_map.items():
            if isinstance(mapping, str):
                # Shorthand: "from_key" -> {"from": "from_key", "weight": 1.0}
                loss_map[loss_key] = {"from": mapping, "weight": 1.0}
            elif isinstance(mapping, dict):
                if "weight" not in mapping:
                    mapping["weight"] = 1.0
        return result

    def _normalize_metrics(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_METRICS)
        result = {**_DEFAULT_METRICS, **copy.deepcopy(raw)}
        # Sort enabled metrics for stability
        if isinstance(result.get("enabled"), list):
            result["enabled"] = sorted(result["enabled"])
        return result

    def _normalize_diagnostics(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_DIAGNOSTICS)
        return {**_DEFAULT_DIAGNOSTICS, **copy.deepcopy(raw)}

    def _normalize_logging(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_LOGGING)
        return {**_DEFAULT_LOGGING, **copy.deepcopy(raw)}

    def _normalize_runtime(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return copy.deepcopy(_DEFAULT_RUNTIME)
        result = {**_DEFAULT_RUNTIME, **copy.deepcopy(raw)}
        # Normalize device to lowercase string
        if "device" in result:
            result["device"] = str(result["device"]).lower()
        # Normalize precision to lowercase
        if "precision" in result:
            result["precision"] = str(result["precision"]).lower()
        return result
