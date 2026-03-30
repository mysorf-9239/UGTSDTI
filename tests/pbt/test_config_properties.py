"""Property-based tests for config normalization."""

from __future__ import annotations

import copy

import yaml
from hypothesis import given
from hypothesis import strategies as st

from tests.quality.test_config import _full_teacher_student_cfg, _minimal_student_only_cfg
from ugtsdti.config.normalize import ConfigNormalizer

normalizer = ConfigNormalizer()


@st.composite
def _valid_normalizer_cfgs(draw):
    cfg = copy.deepcopy(draw(st.sampled_from([_minimal_student_only_cfg(), _full_teacher_student_cfg()])))

    cfg["scenario"]["eval"] = draw(
        st.lists(st.sampled_from(["s1", "s2", "s3", "s4"]), min_size=1, max_size=4, unique=True)
    )
    cfg["runtime"] = {
        "device": draw(st.sampled_from(["cpu", "CPU", "cuda", "Cuda:0"])),
        "precision": draw(st.sampled_from(["fp32", "FP32", "mixed", "MIXED"])),
        "seed": draw(st.integers(min_value=0, max_value=100)),
    }
    cfg["logging"] = {
        "backend": draw(st.sampled_from(["file", "wandb"])),
        "trace_execution": draw(st.booleans()),
        "inspect_state": draw(st.booleans()),
    }
    cfg["metrics"] = {
        "enabled": draw(st.lists(st.sampled_from(["auroc", "auprc", "f1"]), min_size=1, max_size=3, unique=True)),
        "by_scenario": draw(st.booleans()),
    }
    return cfg


@given(_valid_normalizer_cfgs())
def test_normalize_is_idempotent_for_valid_configs(cfg):
    normalized = normalizer.normalize(cfg).to_dict()
    renormalized = normalizer.normalize(normalized).to_dict()

    assert renormalized == normalized


@given(_valid_normalizer_cfgs())
def test_normalize_round_trip_through_yaml_preserves_semantics(cfg):
    normalized = normalizer.normalize(cfg).to_dict()
    serialized = yaml.safe_dump(normalized, sort_keys=True)
    reparsed = yaml.safe_load(serialized)
    renormalized = normalizer.normalize(reparsed).to_dict()

    assert renormalized == normalized


@given(_valid_normalizer_cfgs(), st.permutations(["drug_seq", "protein_seq"]))
def test_normalize_preserves_graph_input_order(cfg, ordered_inputs):
    cfg["graph"]["nodes"]["student_encoder"]["inputs"] = list(ordered_inputs)

    normalized = normalizer.normalize(cfg).to_dict()

    assert normalized["graph"]["nodes"]["student_encoder"]["inputs"] == list(ordered_inputs)
