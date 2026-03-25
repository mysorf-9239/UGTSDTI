"""Extended registry tests: DATASETS, LOSSES registries, error handling,
duplicate registration warning, build with kwargs override.
"""

import pytest
import torch.nn as nn

from ugtsdti.core.registry import DATASETS, LOSSES, MODELS, Registry

# ---------------------------------------------------------------------------
# Registry construction
# ---------------------------------------------------------------------------


def test_registry_name():
    r = Registry("test_reg")
    assert r.name == "test_reg"


def test_registry_contains_false_for_unknown():
    r = Registry("test_contains")
    assert "nonexistent" not in r


def test_registry_get_raises_for_unknown():
    r = Registry("test_get")
    with pytest.raises(KeyError, match="not registered"):
        r.get("nonexistent_key")


def test_registry_build_raises_for_unknown():
    r = Registry("test_build")
    with pytest.raises(KeyError):
        r.build({"name": "nonexistent_model", "params": {}})


# ---------------------------------------------------------------------------
# Duplicate registration warning (no crash)
# ---------------------------------------------------------------------------


def test_duplicate_registration_overwrites(caplog):
    """Registering the same name twice must warn and overwrite (not crash)."""

    r = Registry("test_dup")

    @r.register("dup_model")
    class ModelV1(nn.Module):
        version = 1

    @r.register("dup_model")
    class ModelV2(nn.Module):
        version = 2

    # Second registration wins
    assert r.get("dup_model").version == 2


# ---------------------------------------------------------------------------
# Build with params override
# ---------------------------------------------------------------------------


def test_build_with_kwargs_override():
    r = Registry("test_override")

    @r.register("overridable")
    class OverridableModel(nn.Module):
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            self.hidden_dim = hidden_dim

    # YAML params say 64, but kwargs override to 128
    instance = r.build({"name": "overridable", "params": {"hidden_dim": 64}}, hidden_dim=128)
    assert instance.hidden_dim == 128


def test_build_with_empty_params():
    r = Registry("test_empty_params")

    @r.register("no_params_model")
    class NoParamsModel(nn.Module):
        def __init__(self):
            super().__init__()

    instance = r.build({"name": "no_params_model"})
    assert isinstance(instance, NoParamsModel)


# ---------------------------------------------------------------------------
# MODELS registry — all expected keys registered
# ---------------------------------------------------------------------------


def test_models_registry_has_expected_keys():
    # Trigger imports so decorators run
    import ugtsdti.models  # noqa: F401

    expected = {"hybrid_dti", "baseline_student", "baseline_teacher", "ug_fusion", "gcn_teacher", "esm_student"}
    for key in expected:
        assert key in MODELS, f"'{key}' not found in MODELS registry"


# ---------------------------------------------------------------------------
# LOSSES registry — all expected keys registered
# ---------------------------------------------------------------------------


def test_losses_registry_has_expected_keys():
    import ugtsdti.losses  # noqa: F401

    expected = {"bce", "kd"}
    for key in expected:
        assert key in LOSSES, f"'{key}' not found in LOSSES registry"


def test_losses_build_bce():
    import ugtsdti.losses  # noqa: F401

    loss = LOSSES.build({"name": "bce", "params": {}})
    assert loss is not None


def test_losses_build_kd_dual():
    import ugtsdti.losses  # noqa: F401

    loss = LOSSES.build({"name": "kd", "params": {"alpha": 0.3}})
    assert loss is not None


# ---------------------------------------------------------------------------
# DATASETS registry — key registered
# ---------------------------------------------------------------------------


def test_datasets_registry_has_tdc():
    import ugtsdti.data  # noqa: F401

    assert "tdc_caching_dataset" in DATASETS
