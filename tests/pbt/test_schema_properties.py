"""Property-based tests for schema builder determinism."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from ugtsdti.core.schema import StateSchemaBuilder


@st.composite
def _schema_cfgs(draw):
    nodes = [
        {
            "name": "student_encoder",
            "type_key": "mlp_encoder",
            "output_attrs": ["embedding"],
            "input_kinds": ["drug_seq"],
        },
        {
            "name": "student_head",
            "type_key": "mlp_head",
            "output_attrs": ["logits"],
            "input_kinds": [],
        },
    ]
    roles = {"student": {"source": "student_head.logits"}}
    loss_map = {}

    if draw(st.booleans()):
        loss_map["kd"] = "interaction.kd.loss_component"

    return {
        "graph": {"nodes": nodes},
        "roles": roles,
        "interaction": {"order": [], "dependencies": {}},
        "decision": {"emit_gate_alpha": draw(st.booleans())},
        "loss": {"map": loss_map} if loss_map else {},
    }


@given(_schema_cfgs())
def test_state_schema_builder_is_idempotent_for_same_valid_config(cfg):
    builder = StateSchemaBuilder()

    schema1 = builder.build_for_experiment(cfg)
    schema2 = builder.build_for_experiment(cfg)

    serialized1 = [(spec.key, spec.stage, spec.required, spec.shape, spec.dtype, spec.semantic) for spec in schema1]
    serialized2 = [(spec.key, spec.stage, spec.required, spec.shape, spec.dtype, spec.semantic) for spec in schema2]
    assert serialized1 == serialized2
