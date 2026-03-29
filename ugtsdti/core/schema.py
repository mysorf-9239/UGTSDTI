"""StateSpec, StateSchemaBuilder, BatchSpec — schema and naming contracts.

Canonical key namespace rules (REQ-ARCH-004, REQ-STATE-004, REQ-DATA-003):
- Graph keys:       <node>.<attr>          (must contain exactly one dot)
- Role keys:        <role>.logits          (must end with ".logits")
- Decision keys:    "logits", "gate.alpha", "gate.*"
- Loss keys:        "loss.*"
- Metrics keys:     "metrics.*"
- Diagnostics keys: "diagnostics.*"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ugtsdti.core.errors import InvalidConfigError

# ---------------------------------------------------------------------------
# Compiled patterns for canonical namespace validation
# ---------------------------------------------------------------------------

# Graph output: <node>.<attr>  — at least one char on each side of a single dot
_GRAPH_KEY_RE = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")

# Role output: <role>.logits
_ROLE_KEY_RE = re.compile(r"^[A-Za-z0-9_]+\.logits$")

# Reserved prefixes that must be used for their respective namespaces
_RESERVED_PREFIXES = ("loss.", "metrics.", "diagnostics.", "gate.")

# Decision top-level keys (exact matches allowed at decision stage)
_DECISION_EXACT = {"logits", "gate.alpha"}
_DECISION_PREFIX = "gate."


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_graph_key(key: str) -> None:
    """Raise InvalidConfigError if *key* does not match ``<node>.<attr>``."""
    if not _GRAPH_KEY_RE.match(key):
        raise InvalidConfigError(
            f"Graph key {key!r} must match '<node>.<attr>' (e.g. 'encoder.embedding').",
            stage="graph",
            key=key,
        )


def validate_role_key(key: str) -> None:
    """Raise InvalidConfigError if *key* does not match ``<role>.logits``."""
    if not _ROLE_KEY_RE.match(key):
        raise InvalidConfigError(
            f"Role key {key!r} must match '<role>.logits' (e.g. 'student.logits').",
            stage="role_binding",
            key=key,
        )


def validate_loss_key(key: str) -> None:
    """Raise InvalidConfigError if *key* does not start with ``loss.``."""
    if not key.startswith("loss."):
        raise InvalidConfigError(
            f"Loss key {key!r} must start with 'loss.'.",
            stage="postprocess",
            key=key,
        )


def validate_metrics_key(key: str) -> None:
    """Raise InvalidConfigError if *key* does not start with ``metrics.``."""
    if not key.startswith("metrics."):
        raise InvalidConfigError(
            f"Metrics key {key!r} must start with 'metrics.'.",
            stage="postprocess",
            key=key,
        )


def validate_diagnostics_key(key: str) -> None:
    """Raise InvalidConfigError if *key* does not start with ``diagnostics.``."""
    if not key.startswith("diagnostics."):
        raise InvalidConfigError(
            f"Diagnostics key {key!r} must start with 'diagnostics.'.",
            stage="postprocess",
            key=key,
        )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class StateSpec:
    """Specification for a single key in the pipeline State.

    Attributes:
        key:      The state key name.
        stage:    The pipeline stage that produces this key.
        required: Whether this key must be present at the relevant boundary.
        shape:    Expected tensor shape (None means unconstrained).
        dtype:    Expected dtype string, e.g. "float32" (None means unconstrained).
        semantic: Human-readable description of what this key represents.
    """

    key: str
    stage: str
    required: bool
    shape: tuple[Any, ...] | None
    dtype: str | None
    semantic: str


@dataclass
class BatchSpec:
    """Contract for batch payloads entering the pipeline.

    Attributes:
        required_common:   Keys that must always be present (e.g. "labels", "scenario").
        conditional_keys:  Mapping of key -> condition description.  A key is
                           required only when the described condition is true
                           (e.g. a downstream node needs it).
    """

    required_common: list[str] = field(default_factory=list)
    conditional_keys: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

# Always-required batch keys (REQ-DATA-003)
_ALWAYS_REQUIRED_BATCH_KEYS: list[str] = ["labels", "scenario"]

# Conditionally required batch keys and their conditions
_CONDITIONAL_BATCH_KEYS: dict[str, str] = {
    "drug_seq": "required if any downstream node consumes drug sequence",
    "protein_seq": "required if any downstream node consumes protein sequence",
    "drug_graph": "required if any downstream node consumes drug graph",
    "drug_id": "required if any downstream node consumes drug id",
    "protein_id": "required if any downstream node consumes protein id",
}


class StateSchemaBuilder:
    """Derives a list of StateSpec from a normalized experiment config.

    The schema is not hardcoded; it is assembled from:
    - core keys always present (labels, scenario, logits);
    - graph outputs declared by selected node plugins;
    - role outputs derived from the roles section;
    - interaction outputs declared by selected interaction modules;
    - decision outputs (logits, optional gate.alpha);
    - loss/metrics/diagnostics outputs from postprocess config.

    REQ-STATE-004, REQ-DATA-003, REQ-ARCH-004
    """

    def build_for_experiment(self, cfg: dict[str, Any]) -> list[StateSpec]:
        """Build a list of StateSpec from *cfg* (a normalized config dict).

        The method is deterministic: the same *cfg* always produces an
        equivalent schema (same keys, same attributes).

        Args:
            cfg: Normalized experiment configuration dictionary.

        Returns:
            Sorted list of StateSpec instances.
        """
        specs: list[StateSpec] = []

        # --- Batch / core keys -------------------------------------------
        specs.append(
            StateSpec(
                key="labels",
                stage="batch",
                required=True,
                shape=None,
                dtype=None,
                semantic="Ground-truth interaction labels for the batch.",
            )
        )
        specs.append(
            StateSpec(
                key="scenario",
                stage="batch",
                required=True,
                shape=None,
                dtype=None,
                semantic="Scenario identifier (S1–S4) for each sample.",
            )
        )

        # --- Graph outputs ------------------------------------------------
        graph_cfg = cfg.get("graph", {})
        nodes = graph_cfg.get("nodes", [])
        for node in nodes:
            node_name = node.get("name", "")
            node_type = node.get("type_key", node.get("type", ""))
            output_attrs = node.get("output_attrs", [])
            for attr in output_attrs:
                key = f"{node_name}.{attr}"
                specs.append(
                    StateSpec(
                        key=key,
                        stage="graph",
                        required=True,
                        shape=None,
                        dtype=None,
                        semantic=f"Output '{attr}' from graph node '{node_name}' (type: {node_type}).",
                    )
                )

        # --- Role outputs -------------------------------------------------
        roles_cfg = cfg.get("roles", {})
        for role_name in roles_cfg:
            key = f"{role_name}.logits"
            specs.append(
                StateSpec(
                    key=key,
                    stage="role_binding",
                    required=True,
                    shape=None,
                    dtype=None,
                    semantic=f"Canonical logits for role '{role_name}'.",
                )
            )

        # --- Interaction outputs ------------------------------------------
        interaction_cfg = cfg.get("interaction", {})
        modules = interaction_cfg.get("modules", [])
        for mod in modules:
            for out_key in mod.get("output_keys", []):
                specs.append(
                    StateSpec(
                        key=out_key,
                        stage="interaction",
                        required=False,
                        shape=None,
                        dtype=None,
                        semantic=f"Output from interaction module '{mod.get('name', '')}'.",
                    )
                )

        # --- Decision outputs ---------------------------------------------
        specs.append(
            StateSpec(
                key="logits",
                stage="decision",
                required=True,
                shape=None,
                dtype=None,
                semantic="Final prediction logits produced by the decision stage.",
            )
        )
        decision_cfg = cfg.get("decision", {})
        if decision_cfg.get("emit_gate_alpha", False):
            specs.append(
                StateSpec(
                    key="gate.alpha",
                    stage="decision",
                    required=False,
                    shape=None,
                    dtype=None,
                    semantic="Trust weight alpha in [0, 1] from the gate mechanism.",
                )
            )

        # --- Loss outputs -------------------------------------------------
        loss_cfg = cfg.get("loss", {})
        if loss_cfg:
            specs.append(
                StateSpec(
                    key="loss.total",
                    stage="postprocess",
                    required=True,
                    shape=None,
                    dtype="float32",
                    semantic="Total composed loss for the batch.",
                )
            )
            specs.append(
                StateSpec(
                    key="loss.hard",
                    stage="postprocess",
                    required=True,
                    shape=None,
                    dtype="float32",
                    semantic="Hard (supervised) loss component.",
                )
            )
            for mapped_key in loss_cfg.get("map", {}).keys():
                loss_key = f"loss.{mapped_key}"
                if not any(s.key == loss_key for s in specs):
                    specs.append(
                        StateSpec(
                            key=loss_key,
                            stage="postprocess",
                            required=False,
                            shape=None,
                            dtype="float32",
                            semantic=f"Loss component '{mapped_key}'.",
                        )
                    )

        # Sort for determinism
        specs.sort(key=lambda s: (s.stage, s.key))
        return specs

    def build_batch_spec(self, cfg: dict[str, Any]) -> BatchSpec:
        """Build a BatchSpec from *cfg*.

        Always-required keys: labels, scenario.
        Conditional keys are activated based on which graph nodes are selected.
        """
        graph_cfg = cfg.get("graph", {})
        nodes = graph_cfg.get("nodes", [])
        needed_input_kinds: set[str] = set()
        for node in nodes:
            for key in node.get("inputs", []):
                if key in _CONDITIONAL_BATCH_KEYS:
                    needed_input_kinds.add(key)
            for kind in node.get("input_kinds", []):
                if kind in _CONDITIONAL_BATCH_KEYS:
                    needed_input_kinds.add(kind)

        conditional: dict[str, str] = {}
        for key in sorted(needed_input_kinds):
            conditional[key] = _CONDITIONAL_BATCH_KEYS[key]

        return BatchSpec(
            required_common=list(_ALWAYS_REQUIRED_BATCH_KEYS),
            conditional_keys=conditional,
        )
