"""ConfigValidator — 4-step validation pipeline.

Steps:
  1. required-section validation
  2. schema validation
  3. cross-section validation
  4. (normalization is handled by ConfigNormalizer)

Cross-section checks:
  - roles reference valid graph outputs
  - interaction dependencies acyclic
  - decision prerequisites match available interaction outputs
  - loss mappings reference valid produced keys
  - modality compatibility matches selected nodes
  - teacher/student availability compatible with KD/decision config
  - baseline no-op path valid when teacher/KD/uncertainty disabled

REQ-CONF-002, REQ-ARCH-002, REQ-ARCH-004
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ugtsdti.core.errors import InvalidConfigError
from ugtsdti.core.schema import validate_graph_key

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "version",
    "data",
    "scenario",
    "modalities",
    "graph",
    "roles",
    "interaction",
    "decision",
    "training",
    "loss",
]

VALID_SCENARIOS = {"s1", "s2", "s3", "s4"}

_INPUT_MODALITY_BY_KEY = {
    "drug_seq": "sequence",
    "protein_seq": "sequence",
    "drug_graph": "structure",
    "drug_id": "identifier",
    "protein_id": "identifier",
}

_INTERACTION_ALLOWED_KEYS = {"type", "type_key", "inputs", "params", "output_keys"}


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------


class ConfigValidator:
    """Validate a raw config dict in four ordered steps.

    Usage::

        validator = ConfigValidator()
        validator.validate(raw_dict)   # raises InvalidConfigError on failure

    Fail-fast: the first violation raises immediately with a descriptive error.
    """

    def __init__(
        self,
        *,
        graph_registry: Any | None = None,
        interaction_registry: Any | None = None,
    ) -> None:
        self._graph_registry = graph_registry
        self._interaction_registry = interaction_registry

    def bind_registries(
        self,
        *,
        graph_registry: Any | None = None,
        interaction_registry: Any | None = None,
    ) -> ConfigValidator:
        """Bind runtime registries used for plugin-aware validation."""
        self._graph_registry = graph_registry
        self._interaction_registry = interaction_registry
        return self

    def validate(self, cfg: dict[str, Any]) -> None:
        """Run all validation steps in order.

        Args:
            cfg: Raw (or partially resolved) config dict.

        Raises:
            InvalidConfigError: On the first validation failure.
        """
        self._validate_required_sections(cfg)
        self._validate_schema(cfg)
        self._validate_cross_sections(cfg)

    # ------------------------------------------------------------------
    # Step 1: Required sections
    # ------------------------------------------------------------------

    def _validate_required_sections(self, cfg: dict[str, Any]) -> None:
        for section in REQUIRED_SECTIONS:
            if section not in cfg:
                raise InvalidConfigError(
                    f"Required config section '{section}' is missing.",
                    stage="config_validate",
                    key=section,
                )

    # ------------------------------------------------------------------
    # Step 2: Schema validation
    # ------------------------------------------------------------------

    def _validate_schema(self, cfg: dict[str, Any]) -> None:
        """Validate types and basic structure of each section."""
        self._check_version(cfg)
        self._check_graph_schema(cfg)
        self._check_roles_schema(cfg)
        self._check_interaction_schema(cfg)
        self._check_decision_schema(cfg)
        self._check_loss_schema(cfg)
        self._check_scenario_schema(cfg)

    def _check_version(self, cfg: dict[str, Any]) -> None:
        version = cfg.get("version")
        if not isinstance(version, str) or not version.strip():
            raise InvalidConfigError(
                "Config 'version' must be a non-empty string.",
                stage="config_validate",
                key="version",
            )

    def _check_graph_schema(self, cfg: dict[str, Any]) -> None:
        graph = cfg.get("graph", {})
        if not isinstance(graph, dict):
            raise InvalidConfigError(
                "Config section 'graph' must be a mapping.",
                stage="config_validate",
                key="graph",
            )
        nodes = graph.get("nodes", {})
        if not isinstance(nodes, (dict, list)):
            raise InvalidConfigError(
                "Config 'graph.nodes' must be a mapping or list.",
                stage="config_validate",
                key="graph.nodes",
            )
        if self._graph_registry is not None:
            iterable = nodes.items() if isinstance(nodes, dict) else enumerate(nodes)
            for node_name, node_cfg in iterable:
                if not isinstance(node_cfg, dict):
                    continue
                type_key = str(node_cfg.get("type_key", node_cfg.get("type", "")))
                if not type_key:
                    continue
                try:
                    spec = self._graph_registry.get_spec(type_key)
                except Exception as exc:
                    raise InvalidConfigError(
                        f"Graph node {node_name!r} references unregistered plugin type {type_key!r}. "
                        "Load the required runtime.plugin_registrars before validation.",
                        stage="config_validate",
                        key=f"graph.nodes.{node_name}",
                    ) from exc
                declared_attrs = set(node_cfg.get("output_attrs", []))
                spec_attrs = set(getattr(spec, "output_attrs", []))
                if declared_attrs and spec_attrs and not declared_attrs.issubset(spec_attrs):
                    raise InvalidConfigError(
                        f"Graph node {node_name!r} declares output attrs {sorted(declared_attrs)} "
                        f"which are incompatible with plugin spec {sorted(spec_attrs)} for type {type_key!r}.",
                        stage="config_validate",
                        key=f"graph.nodes.{node_name}.output_attrs",
                    )

    def _check_roles_schema(self, cfg: dict[str, Any]) -> None:
        roles = cfg.get("roles", {})
        if not isinstance(roles, dict):
            raise InvalidConfigError(
                "Config section 'roles' must be a mapping.",
                stage="config_validate",
                key="roles",
            )
        for role_name, role_cfg in roles.items():
            if not isinstance(role_cfg, dict):
                raise InvalidConfigError(
                    f"Role '{role_name}' config must be a mapping.",
                    stage="config_validate",
                    key=f"roles.{role_name}",
                )
            outputs = role_cfg.get("outputs", [])
            if not isinstance(outputs, list) or len(outputs) == 0:
                raise InvalidConfigError(
                    f"Role '{role_name}' must have a non-empty 'outputs' list.",
                    stage="config_validate",
                    key=f"roles.{role_name}.outputs",
                )
            # Validate each output key follows graph naming convention
            for out_key in outputs:
                try:
                    validate_graph_key(out_key)
                except InvalidConfigError as exc:
                    raise InvalidConfigError(
                        f"Role '{role_name}' output key {out_key!r} must follow '<node>.<attr>' naming convention.",
                        stage="config_validate",
                        key=f"roles.{role_name}.outputs",
                    ) from exc
            aggregation = str(role_cfg.get("aggregation", "first"))
            allow_multi_output_first = bool(role_cfg.get("allow_multi_output_first", False))
            if aggregation == "first" and len(outputs) > 1 and not allow_multi_output_first:
                raise InvalidConfigError(
                    f"Role '{role_name}' uses aggregation='first' with multiple outputs. "
                    "Set roles.<name>.allow_multi_output_first=true to make this selection explicit.",
                    stage="config_validate",
                    key=f"roles.{role_name}.outputs",
                )

    def _check_interaction_schema(self, cfg: dict[str, Any]) -> None:
        interaction = cfg.get("interaction", {})
        if not isinstance(interaction, dict):
            raise InvalidConfigError(
                "Config section 'interaction' must be a mapping.",
                stage="config_validate",
                key="interaction",
            )
        order = interaction.get("order", [])
        if not isinstance(order, list):
            raise InvalidConfigError(
                "Config 'interaction.order' must be a list.",
                stage="config_validate",
                key="interaction.order",
            )
        deps = interaction.get("dependencies", {})
        if not isinstance(deps, dict):
            raise InvalidConfigError(
                "Config 'interaction.dependencies' must be a mapping.",
                stage="config_validate",
                key="interaction.dependencies",
            )
        for mod_name in order:
            mod_cfg = interaction.get(mod_name)
            if not isinstance(mod_cfg, dict):
                raise InvalidConfigError(
                    f"Interaction module '{mod_name}' config must be a mapping.",
                    stage="config_validate",
                    key=f"interaction.{mod_name}",
                )
            extra_keys = set(mod_cfg) - _INTERACTION_ALLOWED_KEYS
            if extra_keys:
                raise InvalidConfigError(
                    f"Interaction module '{mod_name}' has unsupported top-level fields {sorted(extra_keys)}. "
                    "Runtime options must be nested under 'params'.",
                    stage="config_validate",
                    key=f"interaction.{mod_name}",
                )
            params = mod_cfg.get("params", {})
            if not isinstance(params, dict):
                raise InvalidConfigError(
                    f"Interaction module '{mod_name}.params' must be a mapping.",
                    stage="config_validate",
                    key=f"interaction.{mod_name}.params",
                )
            type_key = str(mod_cfg.get("type", mod_cfg.get("type_key", "")))
            if not type_key:
                raise InvalidConfigError(
                    f"Interaction module '{mod_name}' is missing a type key.",
                    stage="config_validate",
                    key=f"interaction.{mod_name}.type",
                )
            if type_key == "uncertainty.mc_dropout":
                raise InvalidConfigError(
                    "Interaction type 'uncertainty.mc_dropout' is no longer accepted. "
                    "Use 'uncertainty.sample_variance' for sampled logits or "
                    "'uncertainty.confidence_proxy' for heuristic confidence-derived uncertainty.",
                    stage="config_validate",
                    key=f"interaction.{mod_name}.type",
                )
            if self._interaction_registry is not None:
                try:
                    self._interaction_registry.get_spec(type_key)
                except Exception as exc:
                    raise InvalidConfigError(
                        f"Interaction module '{mod_name}' references unregistered plugin type {type_key!r}. "
                        "Load the required runtime.plugin_registrars before validation.",
                        stage="config_validate",
                        key=f"interaction.{mod_name}.type",
                    ) from exc

    def _check_decision_schema(self, cfg: dict[str, Any]) -> None:
        decision = cfg.get("decision", {})
        if not isinstance(decision, dict):
            raise InvalidConfigError(
                "Config section 'decision' must be a mapping.",
                stage="config_validate",
                key="decision",
            )
        mode = str(decision.get("mode", "heuristic")).lower()
        if mode not in {"heuristic", "learned"}:
            raise InvalidConfigError(
                f"Decision mode {mode!r} is unsupported. Expected 'heuristic' or 'learned'.",
                stage="config_validate",
                key="decision.mode",
            )

    def _check_loss_schema(self, cfg: dict[str, Any]) -> None:
        loss = cfg.get("loss", {})
        if not isinstance(loss, dict):
            raise InvalidConfigError(
                "Config section 'loss' must be a mapping.",
                stage="config_validate",
                key="loss",
            )
        loss_map = loss.get("map", {})
        if not isinstance(loss_map, dict):
            raise InvalidConfigError(
                "Config 'loss.map' must be a mapping.",
                stage="config_validate",
                key="loss.map",
            )

    def _check_scenario_schema(self, cfg: dict[str, Any]) -> None:
        scenario = cfg.get("scenario", {})
        if not isinstance(scenario, dict):
            raise InvalidConfigError(
                "Config section 'scenario' must be a mapping.",
                stage="config_validate",
                key="scenario",
            )

    # ------------------------------------------------------------------
    # Step 3: Cross-section validation
    # ------------------------------------------------------------------

    def _validate_cross_sections(self, cfg: dict[str, Any]) -> None:
        produced_graph_keys = self._collect_graph_output_keys(cfg)
        self._check_roles_reference_graph_outputs(cfg, produced_graph_keys)
        self._check_interaction_acyclic(cfg)
        self._check_decision_prerequisites(cfg)
        self._check_loss_mappings(cfg)
        self._check_modality_compatibility(cfg)
        self._check_teacher_student_availability(cfg)
        self._check_gate_semantics(cfg)
        self._check_baseline_noop_path(cfg)

    # --- 3a: roles reference valid graph outputs ----------------------

    def _collect_graph_output_keys(self, cfg: dict[str, Any]) -> set[str]:
        """Collect all explicitly declared keys that graph nodes produce."""
        graph = cfg.get("graph", {})
        nodes = graph.get("nodes", {})
        produced: set[str] = set()

        if isinstance(nodes, dict):
            for node_name, node_cfg in nodes.items():
                if not isinstance(node_cfg, dict):
                    continue
                for attr in node_cfg.get("output_attrs", []):
                    produced.add(f"{node_name}.{attr}")
        elif isinstance(nodes, list):
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_name = node.get("name", "")
                for attr in node.get("output_attrs", []):
                    produced.add(f"{node_name}.{attr}")

        return produced

    def _check_roles_reference_graph_outputs(self, cfg: dict[str, Any], produced_graph_keys: set[str]) -> None:
        roles = cfg.get("roles", {})
        for role_name, role_cfg in roles.items():
            if not isinstance(role_cfg, dict):
                continue
            for out_key in role_cfg.get("outputs", []):
                if out_key not in produced_graph_keys:
                    raise InvalidConfigError(
                        f"Role '{role_name}' references graph output {out_key!r} "
                        f"which is not produced by any graph node. "
                        f"Produced keys: {sorted(produced_graph_keys)}",
                        stage="config_validate",
                        component="roles",
                        key=out_key,
                    )

    # --- 3b: interaction dependencies acyclic -------------------------

    def _check_interaction_acyclic(self, cfg: dict[str, Any]) -> None:
        interaction = cfg.get("interaction", {})
        order = interaction.get("order", [])
        deps = interaction.get("dependencies", {})

        if not order:
            return

        # Build adjacency: module -> list of modules it depends on
        # Validate that all deps are in the order list
        order_set = set(order)
        for mod_name, mod_deps in deps.items():
            if not isinstance(mod_deps, list):
                mod_deps = [mod_deps]
            for dep in mod_deps:
                if dep not in order_set:
                    raise InvalidConfigError(
                        f"Interaction module '{mod_name}' depends on '{dep}' which is not in interaction.order.",
                        stage="config_validate",
                        component="interaction",
                        key=f"interaction.dependencies.{mod_name}",
                    )

        # Topological sort to detect cycles (Kahn's algorithm)
        # Build in-degree map
        in_degree: dict[str, int] = {m: 0 for m in order}
        adjacency: dict[str, list[str]] = {m: [] for m in order}

        for mod_name, mod_deps in deps.items():
            if mod_name not in order_set:
                continue
            if not isinstance(mod_deps, list):
                mod_deps = [mod_deps]
            for dep in mod_deps:
                if dep in order_set:
                    # dep must come before mod_name
                    adjacency[dep].append(mod_name)
                    in_degree[mod_name] += 1

        queue = [m for m in order if in_degree[m] == 0]
        visited_count = 0
        while queue:
            node = queue.pop(0)
            visited_count += 1
            for successor in adjacency[node]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if visited_count != len(order):
            raise InvalidConfigError(
                "Interaction dependency graph contains a cycle. "
                f"Modules in cycle: {[m for m in order if in_degree[m] > 0]}",
                stage="config_validate",
                component="interaction",
                key="interaction.dependencies",
            )

    # --- 3c: decision prerequisites -----------------------------------

    def _collect_interaction_output_keys(self, cfg: dict[str, Any]) -> set[str]:
        """Collect interaction output keys from config."""
        interaction = cfg.get("interaction", {})
        produced: set[str] = set()

        # From modules list (list-style config)
        for mod in interaction.get("modules", []):
            for k in mod.get("output_keys", []):
                produced.add(k)

        # From named interaction modules (dict-style config)
        order = interaction.get("order", [])
        for mod_name in order:
            mod_cfg = interaction.get(mod_name, {})
            if isinstance(mod_cfg, dict):
                params = _interaction_module_params(mod_cfg)
                if params.get("enabled", True) is False:
                    continue
                for k in mod_cfg.get("output_keys", []):
                    produced.add(k)
                # Well-known outputs by type
                mod_type = mod_cfg.get("type", "")
                if "kd" in mod_type or mod_name == "kd":
                    produced.update(["interaction.kd.loss_component", "kd.teacher_target", "kd.student_target"])
                if "uncertainty" in mod_type or mod_name == "uncertainty":
                    produced.update(["teacher.var", "student.var"])

        return produced

    def _check_decision_prerequisites(self, cfg: dict[str, Any]) -> None:
        decision = cfg.get("decision", {})
        use_uncertainty = decision.get("use_uncertainty", False)

        if use_uncertainty:
            interaction_outputs = self._collect_interaction_output_keys(cfg)
            # Check that at least one uncertainty output is available
            uncertainty_keys = {"teacher.var", "student.var"}
            if not uncertainty_keys.intersection(interaction_outputs):
                # Check if uncertainty module is configured
                interaction = cfg.get("interaction", {})
                order = interaction.get("order", [])
                has_uncertainty = any(
                    isinstance(interaction.get(m, {}), dict)
                    and _interaction_module_params(interaction.get(m, {})).get("enabled", True)
                    and ("uncertainty" in m or "uncertainty" in interaction.get(m, {}).get("type", ""))
                    for m in order
                )
                if not has_uncertainty:
                    raise InvalidConfigError(
                        "Decision config requires 'use_uncertainty: true' but no uncertainty "
                        "interaction module is configured in interaction.order.",
                        stage="config_validate",
                        component="decision",
                        key="decision.use_uncertainty",
                    )

    # --- 3d: loss mappings reference valid produced keys --------------

    def _collect_all_produced_keys(self, cfg: dict[str, Any]) -> set[str]:
        """Collect all keys produced by graph + interaction stages."""
        produced = self._collect_graph_output_keys(cfg)
        produced.update(self._collect_interaction_output_keys(cfg))

        # Role outputs
        roles = cfg.get("roles", {})
        for role_name in roles:
            produced.add(f"{role_name}.logits")

        # Decision outputs
        produced.add("logits")
        produced.add("gate.alpha")

        return produced

    def _check_loss_mappings(self, cfg: dict[str, Any]) -> None:
        loss = cfg.get("loss", {})
        loss_map = loss.get("map", {})
        if not loss_map:
            return

        produced_keys = self._collect_all_produced_keys(cfg)

        for loss_key, mapping in loss_map.items():
            if isinstance(mapping, dict):
                source_key = mapping.get("from", "")
            elif isinstance(mapping, str):
                source_key = mapping
            else:
                raise InvalidConfigError(
                    f"Loss map entry '{loss_key}' must be a string or mapping with 'from' key.",
                    stage="config_validate",
                    component="loss",
                    key=f"loss.map.{loss_key}",
                )

            if source_key and source_key not in produced_keys:
                raise InvalidConfigError(
                    f"Loss map entry '{loss_key}' references key {source_key!r} which is not produced by any stage.",
                    stage="config_validate",
                    component="loss",
                    key=f"loss.map.{loss_key}",
                )

    # --- 3e: modality compatibility -----------------------------------

    def _check_modality_compatibility(self, cfg: dict[str, Any]) -> None:
        modalities = cfg.get("modalities", {})
        available = set(modalities.get("available", []))

        teacher_uses = set(modalities.get("teacher", {}).get("uses", []))
        student_uses = set(modalities.get("student", {}).get("uses", []))

        # Teacher and student modalities must be subsets of available
        if teacher_uses and not teacher_uses.issubset(available):
            extra = teacher_uses - available
            raise InvalidConfigError(
                f"Teacher uses modalities {extra} that are not in modalities.available.",
                stage="config_validate",
                component="modalities",
                key="modalities.teacher.uses",
            )

        if student_uses and not student_uses.issubset(available):
            extra = student_uses - available
            raise InvalidConfigError(
                f"Student uses modalities {extra} that are not in modalities.available.",
                stage="config_validate",
                component="modalities",
                key="modalities.student.uses",
            )

        required_modalities: set[str] = set()
        graph = cfg.get("graph", {})
        nodes = graph.get("nodes", {})

        iterable = nodes.values() if isinstance(nodes, dict) else nodes
        for node in iterable:
            if not isinstance(node, dict):
                continue
            for input_key in node.get("inputs", []):
                modality = _INPUT_MODALITY_BY_KEY.get(input_key)
                if modality is not None:
                    required_modalities.add(modality)
            for input_kind in node.get("input_kinds", []):
                modality = _INPUT_MODALITY_BY_KEY.get(input_kind)
                if modality is not None:
                    required_modalities.add(modality)

        missing_modalities = required_modalities - available
        if missing_modalities:
            raise InvalidConfigError(
                f"Graph requires modalities {sorted(missing_modalities)} that are not in modalities.available.",
                stage="config_validate",
                component="modalities",
                key="modalities.available",
            )

    # --- 3f: teacher/student availability vs KD/decision config ------

    def _check_teacher_student_availability(self, cfg: dict[str, Any]) -> None:
        roles = cfg.get("roles", {})
        has_teacher = "teacher" in roles
        has_student = "student" in roles

        interaction = cfg.get("interaction", {})
        order = interaction.get("order", [])

        # Check KD requires teacher
        kd_enabled = False
        for mod_name in order:
            mod_cfg = interaction.get(mod_name, {})
            if isinstance(mod_cfg, dict):
                mod_type = mod_cfg.get("type", "")
                enabled = _interaction_module_params(mod_cfg).get("enabled", True)
                if ("kd" in mod_name or "kd" in mod_type) and enabled:
                    kd_enabled = True
                    break

        if kd_enabled and not has_teacher:
            raise InvalidConfigError(
                "KD interaction is enabled but no 'teacher' role is configured in roles section.",
                stage="config_validate",
                component="interaction",
                key="interaction.kd",
            )

        if kd_enabled and not has_student:
            raise InvalidConfigError(
                "KD interaction is enabled but no 'student' role is configured in roles section.",
                stage="config_validate",
                component="interaction",
                key="interaction.kd",
            )

        # Decision strategy requiring teacher
        decision = cfg.get("decision", {})
        strategy = decision.get("strategy", "")
        decision_type = decision.get("type", "")

        if (strategy in ("soft",) or "gate" in decision_type) and not (has_teacher or has_student):
            raise InvalidConfigError(
                f"Decision strategy '{strategy or decision_type}' requires at least one role logits branch.",
                stage="config_validate",
                component="decision",
                key="decision.strategy",
            )

    # --- 3g: gate semantics clarity ----------------------------------

    def _check_gate_semantics(self, cfg: dict[str, Any]) -> None:
        decision = cfg.get("decision", {})
        training = cfg.get("training", {})
        decision_type = str(decision.get("type", "identity"))
        decision_mode = str(decision.get("mode", "heuristic")).lower()
        gate_trainable = bool(training.get("gate", {}).get("trainable", False))

        if decision_mode == "learned" or decision_type == "gate.learned":
            raise InvalidConfigError(
                "Learned gate mode is declared in config, but no learned gate runtime is implemented yet.",
                stage="config_validate",
                component="decision",
                key="decision.mode" if decision_mode == "learned" else "decision.type",
            )

        if gate_trainable:
            raise InvalidConfigError(
                "Config sets training.gate.trainable=true, but the current runtime only supports heuristic gates. "
                "Set decision.mode='heuristic' and training.gate.trainable=false until a learned gate runtime exists.",
                stage="config_validate",
                component="training",
                key="training.gate.trainable",
            )

    # --- 3h: baseline no-op path valid when teacher/KD disabled ------

    def _check_baseline_noop_path(self, cfg: dict[str, Any]) -> None:
        """Validate that the baseline no-op path is coherent.

        When teacher and KD are disabled, the pipeline must still be valid:
        - student role must be present
        - decision stage must be able to operate without teacher
        """
        roles = cfg.get("roles", {})
        has_teacher = "teacher" in roles
        has_student = "student" in roles

        interaction = cfg.get("interaction", {})
        order = interaction.get("order", [])

        # Determine if KD is effectively disabled
        kd_disabled = True
        for mod_name in order:
            mod_cfg = interaction.get(mod_name, {})
            if isinstance(mod_cfg, dict):
                mod_type = mod_cfg.get("type", "")
                enabled = _interaction_module_params(mod_cfg).get("enabled", True)
                if ("kd" in mod_name or "kd" in mod_type) and enabled:
                    kd_disabled = False
                    break

        # If no teacher and KD is disabled, student must be present
        if not has_teacher and kd_disabled:
            if not has_student:
                raise InvalidConfigError(
                    "Baseline no-op path requires at least a 'student' role when teacher and KD are disabled.",
                    stage="config_validate",
                    component="roles",
                    key="roles",
                )


def _interaction_module_params(mod_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return canonical nested runtime params for an interaction module."""
    params = mod_cfg.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def validate_only(path: str | Path) -> None:
    """Load and validate a config file without normalization side effects."""
    from ugtsdti.config.loader import ConfigLoader

    cfg = ConfigLoader().load(Path(path))
    ConfigValidator().validate(cfg)
