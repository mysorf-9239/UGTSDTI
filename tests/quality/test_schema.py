"""Unit tests for naming contracts and StateSchemaBuilder.

Covers:
- malformed graph key (missing dot) is rejected
- role key not matching <role>.logits is rejected
- StateSchemaBuilder with same config produces equivalent schema

REQ-STATE-004, REQ-ARCH-004, REQ-CONF-003
"""
import pytest

from ugtsdti.core.errors import InvalidConfigError
from ugtsdti.core.schema import (
    StateSchemaBuilder,
    validate_diagnostics_key,
    validate_graph_key,
    validate_loss_key,
    validate_metrics_key,
    validate_role_key,
)

# ---------------------------------------------------------------------------
# Graph key validation
# ---------------------------------------------------------------------------


class TestGraphKeyValidation:
    def test_valid_graph_key_passes(self):
        validate_graph_key("encoder.embedding")
        validate_graph_key("student_head.logits")
        validate_graph_key("node123.attr456")

    def test_missing_dot_raises(self):
        with pytest.raises(InvalidConfigError) as exc_info:
            validate_graph_key("encoderembedding")
        assert "encoderembedding" in str(exc_info.value)

    def test_leading_dot_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_graph_key(".embedding")

    def test_trailing_dot_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_graph_key("encoder.")

    def test_multiple_dots_raises(self):
        # <node>.<attr> allows only one dot segment on each side
        with pytest.raises(InvalidConfigError):
            validate_graph_key("a.b.c")

    def test_empty_string_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_graph_key("")

    def test_error_contains_key(self):
        with pytest.raises(InvalidConfigError) as exc_info:
            validate_graph_key("badkey")
        assert exc_info.value.key == "badkey"
        assert exc_info.value.stage == "graph"


# ---------------------------------------------------------------------------
# Role key validation
# ---------------------------------------------------------------------------


class TestRoleKeyValidation:
    def test_valid_role_keys_pass(self):
        validate_role_key("student.logits")
        validate_role_key("teacher.logits")
        validate_role_key("branch_a.logits")

    def test_missing_logits_suffix_raises(self):
        with pytest.raises(InvalidConfigError) as exc_info:
            validate_role_key("student.embedding")
        assert "student.embedding" in str(exc_info.value)

    def test_no_dot_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_role_key("studentlogits")

    def test_wrong_suffix_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_role_key("student.logit")  # singular, not "logits"

    def test_extra_prefix_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_role_key("a.b.logits")

    def test_error_contains_stage_and_key(self):
        with pytest.raises(InvalidConfigError) as exc_info:
            validate_role_key("bad_role_key")
        err = exc_info.value
        assert err.stage == "role_binding"
        assert err.key == "bad_role_key"


# ---------------------------------------------------------------------------
# Loss / metrics / diagnostics key validation
# ---------------------------------------------------------------------------


class TestNamespaceKeyValidation:
    def test_valid_loss_key(self):
        validate_loss_key("loss.total")
        validate_loss_key("loss.kd")

    def test_invalid_loss_key_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_loss_key("total")

    def test_valid_metrics_key(self):
        validate_metrics_key("metrics.auroc")

    def test_invalid_metrics_key_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_metrics_key("auroc")

    def test_valid_diagnostics_key(self):
        validate_diagnostics_key("diagnostics.disagreement")

    def test_invalid_diagnostics_key_raises(self):
        with pytest.raises(InvalidConfigError):
            validate_diagnostics_key("disagreement")


# ---------------------------------------------------------------------------
# StateSchemaBuilder determinism
# ---------------------------------------------------------------------------


def _minimal_cfg() -> dict:
    return {
        "graph": {
            "nodes": [
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
        },
        "roles": {
            "student": {"source": "student_head.logits"},
        },
        "interaction": {"order": [], "dependencies": {}},
        "decision": {"emit_gate_alpha": False},
        "loss": {},
    }


class TestStateSchemaBuilder:
    def test_same_config_produces_equivalent_schema(self):
        builder = StateSchemaBuilder()
        cfg = _minimal_cfg()
        schema1 = builder.build_for_experiment(cfg)
        schema2 = builder.build_for_experiment(cfg)
        # Same keys in same order
        assert [s.key for s in schema1] == [s.key for s in schema2]
        # Same attributes
        for s1, s2 in zip(schema1, schema2, strict=True):
            assert s1.key == s2.key
            assert s1.stage == s2.stage
            assert s1.required == s2.required
            assert s1.semantic == s2.semantic

    def test_schema_contains_core_keys(self):
        builder = StateSchemaBuilder()
        schema = builder.build_for_experiment(_minimal_cfg())
        keys = {s.key for s in schema}
        assert "labels" in keys
        assert "scenario" in keys
        assert "logits" in keys

    def test_schema_contains_graph_outputs(self):
        builder = StateSchemaBuilder()
        schema = builder.build_for_experiment(_minimal_cfg())
        keys = {s.key for s in schema}
        assert "student_encoder.embedding" in keys
        assert "student_head.logits" in keys

    def test_schema_contains_role_outputs(self):
        builder = StateSchemaBuilder()
        schema = builder.build_for_experiment(_minimal_cfg())
        keys = {s.key for s in schema}
        assert "student.logits" in keys

    def test_schema_sorted_deterministically(self):
        builder = StateSchemaBuilder()
        schema = builder.build_for_experiment(_minimal_cfg())
        keys = [s.key for s in schema]
        # Just verify the list is stable across two calls
        schema2 = builder.build_for_experiment(_minimal_cfg())
        assert keys == [s.key for s in schema2]

    def test_gate_alpha_included_when_configured(self):
        builder = StateSchemaBuilder()
        cfg = _minimal_cfg()
        cfg["decision"]["emit_gate_alpha"] = True
        schema = builder.build_for_experiment(cfg)
        keys = {s.key for s in schema}
        assert "gate.alpha" in keys

    def test_gate_alpha_excluded_when_not_configured(self):
        builder = StateSchemaBuilder()
        schema = builder.build_for_experiment(_minimal_cfg())
        keys = {s.key for s in schema}
        assert "gate.alpha" not in keys

    def test_loss_keys_included_when_loss_configured(self):
        builder = StateSchemaBuilder()
        cfg = _minimal_cfg()
        cfg["loss"] = {"map": {"kd": "interaction.kd.loss_component"}}
        schema = builder.build_for_experiment(cfg)
        keys = {s.key for s in schema}
        assert "loss.total" in keys
        assert "loss.hard" in keys

    def test_build_batch_spec_always_required_keys(self):
        builder = StateSchemaBuilder()
        batch_spec = builder.build_batch_spec(_minimal_cfg())
        assert "labels" in batch_spec.required_common
        assert "scenario" in batch_spec.required_common

    def test_build_batch_spec_conditional_keys_present(self):
        builder = StateSchemaBuilder()
        batch_spec = builder.build_batch_spec(_minimal_cfg())
        assert "drug_seq" in batch_spec.conditional_keys
        assert "protein_seq" not in batch_spec.conditional_keys

    def test_build_batch_spec_only_includes_selected_conditional_keys(self):
        builder = StateSchemaBuilder()
        cfg = _minimal_cfg()
        cfg["graph"]["nodes"] = [
            {
                "name": "drug_encoder",
                "type_key": "encoder.drug",
                "inputs": ["drug_seq"],
                "output_attrs": ["embedding"],
            },
            {
                "name": "head",
                "type_key": "head.mlp",
                "inputs": ["drug_encoder.embedding"],
                "output_attrs": ["logits"],
            },
        ]
        batch_spec = builder.build_batch_spec(cfg)
        assert batch_spec.required_common == ["labels", "scenario"]
        assert batch_spec.conditional_keys == {
            "drug_seq": "required if any downstream node consumes drug sequence",
        }

    def test_build_batch_spec_supports_mapping_style_graph_nodes(self):
        builder = StateSchemaBuilder()
        cfg = {
            "graph": {
                "nodes": {
                    "student_encoder": {
                        "type_key": "encoder.drug",
                        "inputs": ["drug_seq", "protein_seq"],
                        "output_attrs": ["embedding"],
                    }
                }
            }
        }

        batch_spec = builder.build_batch_spec(cfg)

        assert batch_spec.required_common == ["labels", "scenario"]
        assert batch_spec.conditional_keys == {
            "drug_seq": "required if any downstream node consumes drug sequence",
            "protein_seq": "required if any downstream node consumes protein sequence",
        }

    def test_schema_collects_canonical_interaction_outputs(self):
        builder = StateSchemaBuilder()
        cfg = _minimal_cfg()
        cfg["interaction"] = {
            "order": ["kd", "uncertainty", "diagnostics"],
            "dependencies": {"diagnostics": ["kd", "uncertainty"]},
            "kd": {"type": "kd.standard", "params": {"enabled": True}},
            "uncertainty": {
                "type": "uncertainty.mc_dropout",
                "params": {"enabled": True, "targets": {"teacher": True, "student": False}},
            },
            "diagnostics": {"type": "diagnostics.basic", "params": {"enabled": True, "emit_calibration": True}},
        }

        keys = {spec.key for spec in builder.build_for_experiment(cfg)}

        assert "interaction.kd.loss_component" in keys
        assert "kd.teacher_target" in keys
        assert "teacher.var" in keys
        assert "interaction.disagreement" in keys
