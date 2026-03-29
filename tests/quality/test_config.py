"""Unit tests for cross-section config validation.

Covers:
- each cross-section rule with a violating config
- baseline no-op path valid when teacher/KD/uncertainty disabled
- ConfigNormalizer idempotency
- ConfigLoader extends resolution

REQ-CONF-002
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from ugtsdti.config.loader import ConfigLoader, _deep_merge
from ugtsdti.config.models import NormalizedConfig
from ugtsdti.config.normalize import ConfigNormalizer
from ugtsdti.config.validate import ConfigValidator
from ugtsdti.core.errors import InvalidConfigError

# ---------------------------------------------------------------------------
# Fixtures — minimal valid configs
# ---------------------------------------------------------------------------


def _minimal_student_only_cfg() -> dict:
    """Minimal valid config: student-only, no teacher, no KD, no uncertainty."""
    return {
        "version": "v1",
        "data": {"dataset": "davis"},
        "scenario": {"train": "s1", "eval": ["s1"]},
        "modalities": {
            "available": ["sequence"],
            "student": {"uses": ["sequence"]},
        },
        "graph": {
            "nodes": {
                "student_encoder": {
                    "type": "encoder.baseline",
                    "inputs": ["drug_seq", "protein_seq"],
                    "output_attrs": ["embedding"],
                },
                "student_head": {
                    "type": "head.linear",
                    "inputs": ["student_encoder.embedding"],
                    "output_attrs": ["logits"],
                },
            }
        },
        "roles": {
            "student": {
                "outputs": ["student_head.logits"],
                "aggregation": "first",
            }
        },
        "interaction": {
            "order": [],
            "dependencies": {},
        },
        "decision": {
            "type": "identity",
            "strategy": "identity",
        },
        "training": {
            "student": {"freeze": False},
        },
        "loss": {
            "type": "hard",
            "hard_weight": 1.0,
            "map": {},
        },
    }


def _full_teacher_student_cfg() -> dict:
    """Full config with teacher, student, KD, and uncertainty."""
    return {
        "version": "v1",
        "data": {"dataset": "davis"},
        "scenario": {"train": "s1", "eval": ["s1", "s2", "s3", "s4"]},
        "modalities": {
            "available": ["sequence", "structure"],
            "teacher": {"uses": ["sequence", "structure"]},
            "student": {"uses": ["sequence"]},
        },
        "graph": {
            "nodes": {
                "student_encoder": {
                    "type": "encoder.baseline",
                    "inputs": ["drug_seq", "protein_seq"],
                    "output_attrs": ["embedding"],
                },
                "teacher_encoder": {
                    "type": "encoder.teacher",
                    "inputs": ["drug_graph", "protein_seq"],
                    "output_attrs": ["embedding"],
                },
                "student_head": {
                    "type": "head.linear",
                    "inputs": ["student_encoder.embedding"],
                    "output_attrs": ["logits"],
                },
                "teacher_head": {
                    "type": "head.linear",
                    "inputs": ["teacher_encoder.embedding"],
                    "output_attrs": ["logits"],
                },
            }
        },
        "roles": {
            "teacher": {
                "outputs": ["teacher_head.logits"],
                "aggregation": "first",
            },
            "student": {
                "outputs": ["student_head.logits"],
                "aggregation": "first",
            },
        },
        "interaction": {
            "order": ["uncertainty", "kd"],
            "dependencies": {"kd": ["uncertainty"]},
            "uncertainty": {
                "type": "uncertainty.mc_dropout",
                "samples": 10,
                "enabled": True,
                "targets": {"teacher": True, "student": True},
            },
            "kd": {
                "type": "kd.standard",
                "temperature": 4.0,
                "mode": "logits",
                "enabled": True,
            },
        },
        "decision": {
            "type": "gate.uncertainty",
            "strategy": "soft",
            "trainable": True,
            "use_uncertainty": True,
            "fallback": {"no_teacher": "student", "no_student": "teacher"},
        },
        "training": {
            "teacher": {"freeze": True},
            "student": {"freeze": False},
            "kd": {"schedule": "warmup"},
            "gate": {"trainable": True},
        },
        "loss": {
            "type": "composite",
            "hard_weight": 0.7,
            "map": {
                "kd": {
                    "from": "interaction.kd.loss_component",
                    "weight": 0.3,
                }
            },
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

validator = ConfigValidator()
normalizer = ConfigNormalizer()


def assert_valid(cfg: dict) -> None:
    validator.validate(cfg)


def assert_invalid(cfg: dict, match: str = "") -> None:
    with pytest.raises(InvalidConfigError) as exc_info:
        validator.validate(cfg)
    if match:
        assert match.lower() in str(exc_info.value).lower(), f"Expected {match!r} in error: {exc_info.value}"


# ===========================================================================
# Step 1: Required section validation
# ===========================================================================


class TestRequiredSections:
    def test_valid_minimal_config_passes(self):
        assert_valid(_minimal_student_only_cfg())

    def test_valid_full_config_passes(self):
        assert_valid(_full_teacher_student_cfg())

    @pytest.mark.parametrize(
        "section",
        [
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
        ],
    )
    def test_missing_required_section_fails(self, section: str):
        cfg = _minimal_student_only_cfg()
        del cfg[section]
        assert_invalid(cfg, section)


# ===========================================================================
# Step 2: Schema validation
# ===========================================================================


class TestSchemaValidation:
    def test_non_string_version_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["version"] = 1
        assert_invalid(cfg, "version")

    def test_empty_version_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["version"] = "  "
        assert_invalid(cfg, "version")

    def test_graph_not_dict_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["graph"] = "not_a_dict"
        assert_invalid(cfg, "graph")

    def test_roles_not_dict_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["roles"] = "not_a_dict"
        assert_invalid(cfg, "roles")

    def test_role_empty_outputs_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["roles"]["student"]["outputs"] = []
        assert_invalid(cfg, "outputs")

    def test_role_output_bad_naming_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["roles"]["student"]["outputs"] = ["bad_key_no_dot"]
        assert_invalid(cfg)

    def test_interaction_order_not_list_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"]["order"] = "not_a_list"
        assert_invalid(cfg, "order")

    def test_loss_map_not_dict_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["loss"]["map"] = "not_a_dict"
        assert_invalid(cfg, "map")


# ===========================================================================
# Step 3: Cross-section validation
# ===========================================================================


class TestRolesReferenceGraphOutputs:
    """3a: roles must reference valid graph outputs."""

    def test_role_references_nonexistent_graph_output_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["roles"]["student"]["outputs"] = ["nonexistent_node.logits"]
        assert_invalid(cfg, "nonexistent_node.logits")

    def test_role_references_valid_graph_output_passes(self):
        assert_valid(_minimal_student_only_cfg())

    def test_teacher_role_references_valid_output_passes(self):
        assert_valid(_full_teacher_student_cfg())

    def test_role_references_wrong_node_name_fails(self):
        cfg = _minimal_student_only_cfg()
        # student_head exists but wrong_head does not
        cfg["roles"]["student"]["outputs"] = ["wrong_head.logits"]
        assert_invalid(cfg, "wrong_head.logits")


class TestInteractionAcyclic:
    """3b: interaction dependencies must be acyclic."""

    def test_acyclic_interaction_passes(self):
        assert_valid(_full_teacher_student_cfg())

    def test_direct_cycle_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["a", "b"],
            "dependencies": {
                "a": ["b"],
                "b": ["a"],
            },
        }
        assert_invalid(cfg, "cycle")

    def test_self_cycle_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["a"],
            "dependencies": {"a": ["a"]},
        }
        assert_invalid(cfg)

    def test_three_node_cycle_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["a", "b", "c"],
            "dependencies": {
                "b": ["a"],
                "c": ["b"],
                "a": ["c"],
            },
        }
        assert_invalid(cfg, "cycle")

    def test_dep_not_in_order_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["kd"],
            "dependencies": {"kd": ["uncertainty"]},  # uncertainty not in order
        }
        assert_invalid(cfg, "uncertainty")

    def test_linear_chain_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["a", "b", "c"],
            "dependencies": {"b": ["a"], "c": ["b"]},
        }
        assert_valid(cfg)


class TestDecisionPrerequisites:
    """3c: decision prerequisites must match available interaction outputs."""

    def test_use_uncertainty_without_uncertainty_module_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["decision"]["use_uncertainty"] = True
        # No uncertainty module in interaction
        assert_invalid(cfg, "uncertainty")

    def test_use_uncertainty_with_uncertainty_module_passes(self):
        cfg = _full_teacher_student_cfg()
        assert_valid(cfg)

    def test_use_uncertainty_with_disabled_module_in_params_fails(self):
        cfg = _full_teacher_student_cfg()
        cfg["interaction"]["uncertainty"] = {
            "type": "uncertainty.mc_dropout",
            "params": {"enabled": False, "targets": {"teacher": True, "student": True}},
        }
        assert_invalid(cfg, "uncertainty")

    def test_no_use_uncertainty_without_module_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["decision"]["use_uncertainty"] = False
        assert_valid(cfg)


class TestLossMappings:
    """3d: loss mappings must reference valid produced keys."""

    def test_loss_map_references_valid_interaction_key_passes(self):
        assert_valid(_full_teacher_student_cfg())

    def test_loss_map_references_nonexistent_key_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["loss"]["map"] = {"kd": {"from": "interaction.kd.loss_component", "weight": 0.3}}
        # No KD module configured, so interaction.kd.loss_component is not produced
        assert_invalid(cfg, "interaction.kd.loss_component")

    def test_loss_map_referencing_disabled_kd_output_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["kd"],
            "dependencies": {},
            "kd": {
                "type": "kd.standard",
                "params": {
                    "temperature": 4.0,
                    "mode": "logits",
                    "enabled": False,
                },
            },
        }
        cfg["loss"]["map"] = {"kd": {"from": "interaction.kd.loss_component", "weight": 0.3}}
        assert_invalid(cfg, "interaction.kd.loss_component")

    def test_loss_map_empty_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["loss"]["map"] = {}
        assert_valid(cfg)

    def test_loss_map_references_role_output_passes(self):
        cfg = _minimal_student_only_cfg()
        # student.logits is produced by role binding
        cfg["loss"]["map"] = {"aux": {"from": "student.logits", "weight": 0.1}}
        assert_valid(cfg)


class TestModalityCompatibility:
    """3e: modality compatibility must match selected nodes."""

    def test_teacher_uses_unavailable_modality_fails(self):
        cfg = _full_teacher_student_cfg()
        cfg["modalities"]["teacher"]["uses"] = ["sequence", "structure", "unknown_modality"]
        assert_invalid(cfg, "unknown_modality")

    def test_student_uses_unavailable_modality_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["modalities"]["student"]["uses"] = ["sequence", "nonexistent"]
        assert_invalid(cfg, "nonexistent")

    def test_valid_modality_subset_passes(self):
        assert_valid(_full_teacher_student_cfg())

    def test_graph_input_requires_available_modality(self):
        cfg = _minimal_student_only_cfg()
        cfg["graph"]["nodes"]["student_encoder"]["inputs"] = ["drug_graph", "protein_seq"]
        assert_invalid(cfg, "modalities.available")

    def test_empty_modalities_fail_when_graph_requires_modalities(self):
        cfg = _minimal_student_only_cfg()
        cfg["modalities"] = {"available": []}
        assert_invalid(cfg, "modalities.available")


class TestTeacherStudentAvailability:
    """3f: teacher/student availability must be compatible with KD/decision config."""

    def test_kd_enabled_without_teacher_fails(self):
        cfg = _minimal_student_only_cfg()
        # Add KD but no teacher role
        cfg["interaction"] = {
            "order": ["kd"],
            "dependencies": {},
            "kd": {
                "type": "kd.standard",
                "temperature": 4.0,
                "mode": "logits",
                "enabled": True,
            },
        }
        # Need to add teacher graph outputs for KD to not fail on other checks
        assert_invalid(cfg, "teacher")

    def test_kd_disabled_without_teacher_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["kd"],
            "dependencies": {},
            "kd": {
                "type": "kd.standard",
                "params": {
                    "temperature": 4.0,
                    "mode": "logits",
                    "enabled": False,
                },
            },
        }
        assert_valid(cfg)

    def test_soft_decision_without_teacher_and_no_fallback_fails(self):
        cfg = _minimal_student_only_cfg()
        cfg["decision"] = {
            "type": "gate.uncertainty",
            "strategy": "soft",
            "trainable": True,
            "fallback": {},  # no fallback configured
        }
        assert_invalid(cfg, "teacher")

    def test_soft_decision_without_teacher_but_with_fallback_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["decision"] = {
            "type": "gate.uncertainty",
            "strategy": "soft",
            "trainable": True,
            "fallback": {"no_teacher": "student"},
        }
        assert_valid(cfg)

    def test_full_config_with_teacher_and_kd_passes(self):
        assert_valid(_full_teacher_student_cfg())

    def test_teacher_optional_in_student_only_config(self):
        cfg = _minimal_student_only_cfg()
        assert "teacher" not in cfg["roles"]
        assert_valid(cfg)

    def test_teacher_can_use_richer_modalities_than_student(self):
        cfg = _full_teacher_student_cfg()
        cfg["modalities"]["teacher"]["uses"] = ["sequence", "structure"]
        cfg["modalities"]["student"]["uses"] = ["sequence"]
        assert_valid(cfg)


class TestBaselineNoopPath:
    """3g: baseline no-op path valid when teacher/KD/uncertainty disabled."""

    def test_student_only_no_teacher_no_kd_passes(self):
        """Baseline: student only, no teacher, no KD, no uncertainty."""
        assert_valid(_minimal_student_only_cfg())

    def test_no_student_no_teacher_no_kd_fails(self):
        """No student and no teacher with KD disabled must fail."""
        cfg = _minimal_student_only_cfg()
        del cfg["roles"]["student"]
        assert_invalid(cfg)

    def test_student_only_with_noop_interaction_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {"order": [], "dependencies": {}}
        assert_valid(cfg)

    def test_student_only_identity_decision_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["decision"] = {"type": "identity", "strategy": "identity"}
        assert_valid(cfg)

    def test_student_only_hard_loss_only_passes(self):
        cfg = _minimal_student_only_cfg()
        cfg["loss"] = {"type": "hard", "hard_weight": 1.0, "map": {}}
        assert_valid(cfg)


# ===========================================================================
# ConfigNormalizer tests
# ===========================================================================


class TestConfigNormalizerIdempotency:
    """normalize(normalize(cfg)) == normalize(cfg)"""

    def test_idempotent_minimal_config(self):
        cfg = _minimal_student_only_cfg()
        norm1 = normalizer.normalize(cfg)
        norm2 = normalizer.normalize(norm1.to_dict())
        assert norm1.to_dict() == norm2.to_dict()

    def test_idempotent_full_config(self):
        cfg = _full_teacher_student_cfg()
        norm1 = normalizer.normalize(cfg)
        norm2 = normalizer.normalize(norm1.to_dict())
        assert norm1.to_dict() == norm2.to_dict()

    def test_normalized_config_has_all_required_fields(self):
        cfg = _minimal_student_only_cfg()
        norm = normalizer.normalize(cfg)
        assert norm.version == "v1"
        assert isinstance(norm.data, dict)
        assert isinstance(norm.graph, dict)
        assert isinstance(norm.roles, dict)
        assert isinstance(norm.interaction, dict)
        assert isinstance(norm.decision, dict)
        assert isinstance(norm.training, dict)
        assert isinstance(norm.loss, dict)

    def test_normalized_config_optional_fields_have_defaults(self):
        cfg = _minimal_student_only_cfg()
        norm = normalizer.normalize(cfg)
        assert isinstance(norm.metrics, dict)
        assert isinstance(norm.diagnostics, dict)
        assert isinstance(norm.logging, dict)
        assert isinstance(norm.runtime, dict)
        assert isinstance(norm.experiment, dict)
        assert isinstance(norm.extends, list)
        assert isinstance(norm.sweep, dict)

    def test_normalize_fills_interaction_defaults(self):
        cfg = _minimal_student_only_cfg()
        norm = normalizer.normalize(cfg)
        assert "order" in norm.interaction
        assert "dependencies" in norm.interaction

    def test_normalize_fills_decision_defaults(self):
        cfg = _minimal_student_only_cfg()
        norm = normalizer.normalize(cfg)
        assert "type" in norm.decision
        assert "fallback" in norm.decision

    def test_normalize_fills_runtime_defaults(self):
        cfg = _minimal_student_only_cfg()
        norm = normalizer.normalize(cfg)
        assert "device" in norm.runtime
        assert "seed" in norm.runtime
        assert "deterministic" in norm.runtime

    def test_normalize_sorts_eval_scenarios(self):
        cfg = _minimal_student_only_cfg()
        cfg["scenario"]["eval"] = ["s4", "s2", "s1", "s3"]
        norm = normalizer.normalize(cfg)
        assert norm.scenario["eval"] == ["s1", "s2", "s3", "s4"]

    def test_normalize_sorts_modalities_available(self):
        cfg = _minimal_student_only_cfg()
        cfg["modalities"]["available"] = ["structure", "sequence"]
        norm = normalizer.normalize(cfg)
        assert norm.modalities["available"] == ["sequence", "structure"]

    def test_normalize_loss_map_shorthand_expanded(self):
        cfg = _minimal_student_only_cfg()
        cfg["interaction"] = {
            "order": ["kd"],
            "dependencies": {},
            "kd": {"type": "kd.standard", "temperature": 4.0, "mode": "logits", "enabled": True},
        }
        cfg["roles"]["teacher"] = {"outputs": ["student_head.logits"], "aggregation": "first"}
        cfg["loss"]["map"] = {"kd": "interaction.kd.loss_component"}
        norm = normalizer.normalize(cfg)
        # Shorthand string should be expanded to dict
        assert isinstance(norm.loss["map"]["kd"], dict)
        assert norm.loss["map"]["kd"]["from"] == "interaction.kd.loss_component"
        assert "weight" in norm.loss["map"]["kd"]

    def test_normalize_does_not_move_plugins_between_stages(self):
        """Normalization MUST NOT move plugins between stages."""
        cfg = _full_teacher_student_cfg()
        norm = normalizer.normalize(cfg)
        # KD must remain in interaction, not appear in decision or loss
        assert "kd" not in norm.decision
        assert "kd" not in norm.loss
        # Uncertainty must remain in interaction
        assert "uncertainty" not in norm.decision

    def test_normalize_does_not_infer_hidden_loss_mappings(self):
        """Normalization MUST NOT add loss mappings that weren't in the config."""
        cfg = _minimal_student_only_cfg()
        norm = normalizer.normalize(cfg)
        # No KD in config, so no kd loss mapping should appear
        assert "kd" not in norm.loss.get("map", {})

    def test_to_dict_from_dict_roundtrip(self):
        """parse -> normalize -> serialize -> parse keeps semantic equivalence."""
        cfg = _full_teacher_student_cfg()
        norm1 = normalizer.normalize(cfg)
        d = norm1.to_dict()
        norm2 = NormalizedConfig.from_dict(d)
        assert norm1.to_dict() == norm2.to_dict()

    def test_normalize_stable_for_hashing(self):
        """Same config always produces same dict (stable for hashing)."""
        cfg = _minimal_student_only_cfg()
        norm1 = normalizer.normalize(cfg)
        norm2 = normalizer.normalize(copy.deepcopy(cfg))
        assert norm1.to_dict() == norm2.to_dict()


# ===========================================================================
# ConfigLoader tests
# ===========================================================================


class TestConfigLoader:
    def test_load_simple_yaml(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(_minimal_student_only_cfg()))
        loader = ConfigLoader()
        result = loader.load(cfg_file)
        assert result["version"] == "v1"

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        loader = ConfigLoader()
        with pytest.raises(InvalidConfigError):
            loader.load(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text("key: [unclosed")
        loader = ConfigLoader()
        with pytest.raises(InvalidConfigError):
            loader.load(cfg_file)

    def test_load_non_mapping_yaml_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        loader = ConfigLoader()
        with pytest.raises(InvalidConfigError):
            loader.load(cfg_file)

    def test_extends_resolution_later_overrides_earlier(self, tmp_path: Path):
        """Later entries in extends override earlier ones."""
        base1 = tmp_path / "base1.yaml"
        base1.write_text(yaml.dump({"version": "v1", "data": {"dataset": "base1"}}))

        base2 = tmp_path / "base2.yaml"
        base2.write_text(yaml.dump({"version": "v1", "data": {"dataset": "base2"}}))

        main = tmp_path / "main.yaml"
        main.write_text(
            yaml.dump(
                {
                    "extends": ["base1", "base2"],
                    "version": "v1",
                }
            )
        )

        loader = ConfigLoader()
        result = loader.load(main)
        # base2 overrides base1
        assert result["data"]["dataset"] == "base2"

    def test_extends_main_config_wins(self, tmp_path: Path):
        """Main config always wins over base configs."""
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"version": "v1", "data": {"dataset": "base"}}))

        main = tmp_path / "main.yaml"
        main.write_text(
            yaml.dump(
                {
                    "extends": ["base"],
                    "version": "v1",
                    "data": {"dataset": "main"},
                }
            )
        )

        loader = ConfigLoader()
        result = loader.load(main)
        assert result["data"]["dataset"] == "main"

    def test_extends_deep_merge(self, tmp_path: Path):
        """Extends merges nested dicts rather than replacing them."""
        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.dump(
                {
                    "version": "v1",
                    "training": {"teacher": {"freeze": True}, "student": {"freeze": False}},
                }
            )
        )

        main = tmp_path / "main.yaml"
        main.write_text(
            yaml.dump(
                {
                    "extends": ["base"],
                    "version": "v1",
                    "training": {"student": {"freeze": True}},  # override only student
                }
            )
        )

        loader = ConfigLoader()
        result = loader.load(main)
        # teacher.freeze from base should still be present
        assert result["training"]["teacher"]["freeze"] is True
        # student.freeze overridden by main
        assert result["training"]["student"]["freeze"] is True

    def test_extends_circular_raises(self, tmp_path: Path):
        """Circular extends must raise InvalidConfigError."""
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(yaml.dump({"extends": ["b"], "version": "v1"}))
        b.write_text(yaml.dump({"extends": ["a"], "version": "v1"}))

        loader = ConfigLoader()
        with pytest.raises(InvalidConfigError, match="[Cc]ircular"):
            loader.load(a)

    def test_extends_missing_base_raises(self, tmp_path: Path):
        main = tmp_path / "main.yaml"
        main.write_text(
            yaml.dump(
                {
                    "extends": ["nonexistent_base"],
                    "version": "v1",
                }
            )
        )
        loader = ConfigLoader()
        with pytest.raises(InvalidConfigError):
            loader.load(main)

    def test_extends_not_in_result(self, tmp_path: Path):
        """After resolution, 'extends' key should not appear in result."""
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"version": "v1", "data": {"dataset": "base"}}))

        main = tmp_path / "main.yaml"
        main.write_text(
            yaml.dump(
                {
                    "extends": ["base"],
                    "version": "v1",
                }
            )
        )

        loader = ConfigLoader()
        result = loader.load(main)
        assert "extends" not in result


# ===========================================================================
# Deep merge helper tests
# ===========================================================================


class TestDeepMerge:
    def test_scalar_override(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result["a"] == 2

    def test_nested_dict_merge(self):
        result = _deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 99}})
        assert result["a"]["x"] == 1
        assert result["a"]["y"] == 99

    def test_list_override(self):
        result = _deep_merge({"a": [1, 2, 3]}, {"a": [4, 5]})
        assert result["a"] == [4, 5]

    def test_new_key_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result["a"] == 1
        assert result["b"] == 2

    def test_base_not_mutated(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"x": 2}})
        assert base["a"]["x"] == 1
