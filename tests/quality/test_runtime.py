"""Unit tests for runtime identity, checkpointing, and logging."""

from __future__ import annotations

import json

import pytest

from ugtsdti.core.errors import CheckpointCorruptedError
from ugtsdti.logging import CompositeLogger, FileLogger, WandbLogger
from ugtsdti.runtime import (
    CheckpointBundle,
    CheckpointIO,
    RuntimeAdapter,
    build_experiment_identity,
    build_reproducibility_key,
    seed_worker,
)


def test_reproducibility_key_is_stable():
    key1 = build_reproducibility_key(
        config_hash="abc",
        dataset_version="dataset-v1",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )
    key2 = build_reproducibility_key(
        config_hash="abc",
        dataset_version="dataset-v1",
        preprocessing_version="prep-v1",
        split_version="split-v1",
        seed=7,
    )
    assert key1 == key2


def test_experiment_identity_separates_run_id_from_config_hash():
    identity1 = build_experiment_identity({"model": "a"})
    identity2 = build_experiment_identity({"model": "a"})
    assert identity1.config_hash == identity2.config_hash
    assert identity1.run_id != identity2.run_id


def test_experiment_identity_ignores_runtime_checkpoint_path_for_hash():
    base_cfg = {"model": "a", "runtime": {"device": "cpu"}}
    resumed_cfg = {"model": "a", "runtime": {"device": "cpu", "checkpoint_path": "/tmp/model.pt"}}

    identity1 = build_experiment_identity(base_cfg)
    identity2 = build_experiment_identity(resumed_cfg)

    assert identity1.config_hash == identity2.config_hash


def test_experiment_identity_ignores_runtime_operational_paths_for_hash():
    base_cfg = {
        "model": "a",
        "runtime": {
            "device": "cpu",
            "artifacts_dir": "/tmp/run_a/artifacts",
            "checkpoint_dir": "/tmp/run_a/checkpoints",
            "data_dir": "/tmp/run_a/data",
        },
    }
    moved_cfg = {
        "model": "a",
        "runtime": {
            "device": "cpu",
            "artifacts_dir": "/tmp/run_b/artifacts",
            "checkpoint_dir": "/tmp/run_b/checkpoints",
            "data_dir": "/tmp/run_b/data",
        },
    }

    identity1 = build_experiment_identity(base_cfg)
    identity2 = build_experiment_identity(moved_cfg)

    assert identity1.config_hash == identity2.config_hash


def test_experiment_identity_preserves_order_sensitive_graph_inputs_in_hash():
    base_cfg = {
        "graph": {
            "nodes": {
                "encoder": {
                    "inputs": ["drug_seq", "protein_seq"],
                }
            }
        }
    }
    reordered_cfg = {
        "graph": {
            "nodes": {
                "encoder": {
                    "inputs": ["protein_seq", "drug_seq"],
                }
            }
        }
    }

    identity1 = build_experiment_identity(base_cfg)
    identity2 = build_experiment_identity(reordered_cfg)

    assert identity1.config_hash != identity2.config_hash


def test_experiment_identity_canonicalizes_order_insensitive_eval_scenarios_for_hash():
    base_cfg = {"scenario": {"eval": ["s1", "s4", "s2"]}}
    reordered_cfg = {"scenario": {"eval": ["s2", "s1", "s4"]}}

    identity1 = build_experiment_identity(base_cfg)
    identity2 = build_experiment_identity(reordered_cfg)

    assert identity1.config_hash == identity2.config_hash


def test_runtime_adapter_resolves_operational_knobs():
    adapted = RuntimeAdapter().adapt(
        {
            "runtime": {
                "device": "cpu",
                "seed": 3,
                "artifacts_dir": "artifacts/test",
                "pin_memory": True,
                "data_dir": "data/input",
                "checkpoint_path": "checkpoints/latest.pt",
            }
        }
    )
    assert adapted["device"] == "cpu"
    assert adapted["seed"] == 3
    assert adapted["artifacts_dir"].endswith("artifacts/test")
    assert adapted["pin_memory"] is True
    assert adapted["data_dir"].endswith("data/input")
    assert adapted["checkpoint_path"].endswith("checkpoints/latest.pt")


def test_checkpoint_corrupt_bundle_raises(tmp_path):
    path = tmp_path / "broken.ckpt.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path)


def test_checkpoint_config_or_dataset_mismatch_raises(tmp_path):
    bundle = CheckpointBundle(
        model_state={"w": 1},
        optimizer_state={},
        scheduler_state={},
        rng_state={},
        epoch=1,
        step=2,
        identity={"config_hash": "abc", "reproducibility_key": "rep-1"},
        config={"model": "baseline"},
        dataset_metadata={"dataset": "davis"},
        split_metadata={"split_version": "v1"},
    )
    path = CheckpointIO().save(bundle, tmp_path / "checkpoint.json")

    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path, expected_config_hash="other")

    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path, expected_dataset="kiba")

    with pytest.raises(CheckpointCorruptedError):
        CheckpointIO().load(path, expected_reproducibility_key="other")


def test_checkpoint_io_round_trips_tensor_state(tmp_path):
    torch = pytest.importorskip("torch")
    bundle = CheckpointBundle(
        model_state={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state={"momentum": torch.tensor([0.5])},
        scheduler_state={},
        rng_state={},
        epoch=1,
        step=2,
        identity={"config_hash": "abc"},
        config={"model": "baseline"},
        dataset_metadata={"dataset": "davis"},
        split_metadata={"split_version": "v1"},
    )

    path = CheckpointIO().save(bundle, tmp_path / "checkpoint.pt")
    loaded = CheckpointIO().load(path)

    assert torch.equal(loaded.model_state["weight"], torch.tensor([1.0, 2.0]))
    assert torch.equal(loaded.optimizer_state["momentum"], torch.tensor([0.5]))


def test_checkpoint_load_unknown_model_node_is_rejected_by_executor():
    from ugtsdti.graph.builder import GraphBuilder
    from ugtsdti.graph.planner import GraphPlanner
    from ugtsdti.interaction.base import InteractionPluginSpec
    from ugtsdti.interaction.noop import NoOpInteraction
    from ugtsdti.interaction.registry import InteractionPlanner, InteractionRegistry
    from ugtsdti.runtime.defaults import build_default_graph_registry
    from ugtsdti.trainer.trainer import PipelineExecutor

    graph_registry = build_default_graph_registry()
    interaction_registry = InteractionRegistry()
    interaction_registry.register(
        InteractionPluginSpec(type_key="noop", output_keys_fn=lambda params: []),
        NoOpInteraction,
    )
    cfg = {
        "graph": {
            "nodes": [
                {"name": "student_encoder", "type_key": "encoder.baseline", "inputs": ["drug_seq", "protein_seq"]},
                {"name": "student_head", "type_key": "head.linear", "inputs": ["student_encoder.embedding"]},
            ]
        },
        "interaction": {
            "order": ["noop"],
            "dependencies": {},
            "noop": {"type": "noop", "inputs": ["student.logits"]},
        },
    }
    cfg["graph_plan"] = GraphPlanner(graph_registry).plan(GraphBuilder(graph_registry).build(cfg["graph"]))
    cfg["interaction_plan"] = InteractionPlanner(interaction_registry).plan(
        cfg["interaction"],
        available_inputs={"student.logits"},
    )
    executor = PipelineExecutor(graph_registry=graph_registry, interaction_registry=interaction_registry)

    with pytest.raises(CheckpointCorruptedError, match="unknown graph node"):
        executor.load_model_state(cfg, {"missing_node": {"weight": 1}})


def test_file_logger_and_composite_logger_write_outputs(tmp_path):
    file_logger = FileLogger(tmp_path / "logs")
    composite = CompositeLogger([file_logger, WandbLogger(project="test", enabled=False)])
    composite.log_metrics({"loss.total": 0.5}, step=3)
    composite.log_text("summary", "hello")
    composite.close()

    metrics_lines = (tmp_path / "logs" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(metrics_lines[0])
    assert payload["step"] == 3
    assert payload["metrics"]["loss.total"] == 0.5
    assert (tmp_path / "logs" / "summary.txt").read_text(encoding="utf-8") == "hello"


def test_seed_worker_returns_unique_derived_seed():
    assert seed_worker(1, 10) == 11
