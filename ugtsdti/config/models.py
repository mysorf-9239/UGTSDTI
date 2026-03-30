"""NormalizedConfig dataclass — internal representation used by runtime.

All top-level sections from design Section 7.2.

REQ-CONF-001, REQ-CONF-003
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedConfig:
    """Normalized, validated, and resolved experiment configuration.

    This is the internal representation that the runtime uses.
    It must be:
    - deterministic (same input -> same output)
    - idempotent (normalize(normalize(cfg)) == normalize(cfg))
    - serializable (can be round-tripped through YAML/dict)
    - stable enough to hash (via to_dict())

    Required sections: version, data, scenario, modalities, graph,
                       roles, interaction, decision, training, loss
    Optional sections: experiment, extends, sweep, metrics, diagnostics,
                       logging, runtime
    """

    # Required sections
    version: str
    data: dict[str, Any]
    scenario: dict[str, Any]
    modalities: dict[str, Any]
    graph: dict[str, Any]
    roles: dict[str, Any]
    interaction: dict[str, Any]
    decision: dict[str, Any]
    training: dict[str, Any]
    loss: dict[str, Any]

    # Optional sections (default to empty dict)
    experiment: dict[str, Any] = field(default_factory=dict)
    extends: list[str] = field(default_factory=list)
    sweep: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for YAML round-trip or hashing."""
        return {
            "version": self.version,
            "experiment": self.experiment,
            "extends": self.extends,
            "sweep": self.sweep,
            "data": self.data,
            "scenario": self.scenario,
            "modalities": self.modalities,
            "graph": self.graph,
            "roles": self.roles,
            "interaction": self.interaction,
            "decision": self.decision,
            "training": self.training,
            "loss": self.loss,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "logging": self.logging,
            "runtime": self.runtime,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormalizedConfig":
        """Construct from a plain dict (e.g. after YAML load)."""
        return cls(
            version=d["version"],
            data=d["data"],
            scenario=d["scenario"],
            modalities=d["modalities"],
            graph=d["graph"],
            roles=d["roles"],
            interaction=d["interaction"],
            decision=d["decision"],
            training=d["training"],
            loss=d["loss"],
            experiment=d.get("experiment", {}),
            extends=d.get("extends", []),
            sweep=d.get("sweep", {}),
            metrics=d.get("metrics", {}),
            diagnostics=d.get("diagnostics", {}),
            logging=d.get("logging", {}),
            runtime=d.get("runtime", {}),
        )
