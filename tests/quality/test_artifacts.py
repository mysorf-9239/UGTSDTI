"""Tests for artifact bundles and performance/concurrency invariants."""
from __future__ import annotations

import json

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.state import State, StateWriter
from ugtsdti.graph.engine import GraphEngine
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import GraphPlan, NodeDefinition, NodePluginSpec
from ugtsdti.nodes.base import NodeRuntime
from ugtsdti.runtime import ArtifactWriter, build_experiment_identity, build_reproducibility_key


class SourceRuntime(NodeRuntime):
    seen_outputs: list[object] = []

    def forward(self, inputs, context):
        del context
        payload = {"values": list(inputs["payload"])}
        self.__class__.seen_outputs.append(payload)
        return {"payload": payload}


class SinkRuntime(NodeRuntime):
    seen_input_ids: list[int] = []

    def forward(self, inputs, context):
        del context
        payload = inputs["source.payload"]
        self.__class__.seen_input_ids.append(id(payload))
        return {"copied": payload["values"][0]}


def test_artifact_bundle_contains_reproducibility_metadata(tmp_path):
    logs_dir = tmp_path / "logs-src"
    logs_dir.mkdir()
    (logs_dir / "events.txt").write_text("ok", encoding="utf-8")

    identity = build_experiment_identity({"model": "baseline"})
    bundle_dir = ArtifactWriter(tmp_path / "artifacts").write_bundle(
        identity=identity.to_dict(),
        config={"version": "1.0", "decision": {"type": "identity"}},
        metrics={"metrics.auroc": 0.9},
        diagnostics={"diagnostics.disagreement": 0.1},
        split_manifest={"dataset": "davis", "split_version": "v1"},
        model_state={"weight": [1, 2, 3]},
        execution_trace={"stage_order": ["batch", "graph", "decision"]},
        state_boundary_summaries={"decision": ["logits"]},
        logs_dir=logs_dir,
    )

    expected = {
        "config.yaml",
        "identity.json",
        "metrics.json",
        "diagnostics.json",
        "split_manifest.json",
        "model.pt",
        "execution_trace.json",
        "state_boundaries.json",
        "logs",
    }
    assert expected.issubset({path.name for path in bundle_dir.iterdir()})
    assert json.loads((bundle_dir / "identity.json").read_text(encoding="utf-8"))["run_id"] == identity.run_id
    assert (bundle_dir / "logs" / "events.txt").read_text(encoding="utf-8") == "ok"


def test_reproducibility_key_stays_stable_while_run_identity_changes():
    identity1 = build_experiment_identity({"decision": {"type": "identity"}})
    identity2 = build_experiment_identity({"decision": {"type": "identity"}})

    key1 = build_reproducibility_key(
        config_hash=identity1.config_hash,
        dataset_version="davis-v1",
        preprocessing_version="prep-v1",
        split_version="s1-v1",
        seed=7,
    )
    key2 = build_reproducibility_key(
        config_hash=identity2.config_hash,
        dataset_version="davis-v1",
        preprocessing_version="prep-v1",
        split_version="s1-v1",
        seed=7,
    )

    assert identity1.run_id != identity2.run_id
    assert identity1.timestamp != identity2.timestamp
    assert key1 == key2


def test_graph_engine_materializes_declared_inputs_without_state_deep_copy():
    SourceRuntime.seen_outputs = []
    SinkRuntime.seen_input_ids = []

    registry = NodeRegistry()
    registry.register(
        NodePluginSpec(type_key="source.runtime", output_attrs=["payload"]),
        SourceRuntime,
    )
    registry.register(
        NodePluginSpec(type_key="sink.runtime", output_attrs=["copied"]),
        SinkRuntime,
    )
    plan = GraphPlan(
        node_definitions=[
            NodeDefinition(name="source", type_key="source.runtime", inputs=["payload"]),
            NodeDefinition(name="sink", type_key="sink.runtime", inputs=["source.payload"]),
        ],
        order=["source", "sink"],
        produced_keys={"source": ["source.payload"], "sink": ["sink.copied"]},
        producers={"source.payload": "source", "sink.copied": "sink"},
        edges={"source": [], "sink": ["source"]},
    )
    state = State()
    writer = StateWriter(state)
    writer.commit("batch", {"payload": [7]})

    GraphEngine(registry).run(
        plan,
        state,
        writer,
        ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False),
    )

    assert SourceRuntime.seen_outputs
    assert SinkRuntime.seen_input_ids == [id(state.get("source.payload"))]


def test_graph_engine_runs_do_not_share_mutable_state_between_executions():
    registry = NodeRegistry()
    registry.register(
        NodePluginSpec(type_key="source.runtime", output_attrs=["payload"]),
        SourceRuntime,
    )
    registry.register(
        NodePluginSpec(type_key="sink.runtime", output_attrs=["copied"]),
        SinkRuntime,
    )
    plan = GraphPlan(
        node_definitions=[
            NodeDefinition(name="source", type_key="source.runtime", inputs=["payload"]),
            NodeDefinition(name="sink", type_key="sink.runtime", inputs=["source.payload"]),
        ],
        order=["source", "sink"],
        produced_keys={"source": ["source.payload"], "sink": ["sink.copied"]},
        producers={"source.payload": "source", "sink.copied": "sink"},
        edges={"source": [], "sink": ["source"]},
    )
    engine = GraphEngine(registry)

    state1 = State()
    writer1 = StateWriter(state1)
    writer1.commit("batch", {"payload": [1]})
    engine.run(plan, state1, writer1, ExecutionContext(mode="train", seed=0, device="cpu", deterministic=False))

    state2 = State()
    writer2 = StateWriter(state2)
    writer2.commit("batch", {"payload": [9]})
    engine.run(plan, state2, writer2, ExecutionContext(mode="train", seed=1, device="cpu", deterministic=False))

    state1.get("source.payload")["values"][0] = 99

    assert state1.get("sink.copied") == 1
    assert state2.get("sink.copied") == 9
    assert state2.get("source.payload")["values"][0] == 9
