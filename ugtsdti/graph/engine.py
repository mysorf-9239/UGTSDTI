"""GraphEngine — runtime execution loop for the graph stage.

REQ-GRAPH-002, REQ-GRAPH-003, REQ-STATE-003, REQ-QUAL-002, REQ-QUAL-004
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import (
    InvalidConfigError,
    MissingDependencyError,
    NumericalInstabilityError,
)
from ugtsdti.core.schema import validate_graph_key
from ugtsdti.core.state import State, StateWriter
from ugtsdti.graph.registry import NodeRegistry
from ugtsdti.graph.specs import GraphPlan

# ---------------------------------------------------------------------------
# Trace event
# ---------------------------------------------------------------------------


@dataclass
class NodeExecutionEvent:
    """Per-node execution event emitted in debug/trace mode.

    Attributes:
        node_name:        Name of the node that was executed.
        inputs_consumed:  State keys that were materialized as inputs.
        outputs_produced: State keys that were committed to State.
    """

    node_name: str
    inputs_consumed: list[str]
    outputs_produced: list[str]


@dataclass
class GraphTrace:
    """Execution trace for the entire graph stage.

    Attributes:
        node_order: Execution order of nodes.
        events:     Per-node execution events in execution order.
    """

    node_order: list[str] = field(default_factory=list)
    events: list[NodeExecutionEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GraphEngine
# ---------------------------------------------------------------------------


class GraphEngine:
    """Executes a GraphPlan against a State using a NodeRegistry.

    Runtime loop per node:
    1. Materialize declared inputs from State (shallow — no full State copy).
    2. Call NodeRuntime.forward(inputs, context).
    3. Validate outputs against NodePluginSpec.output_attrs.
    4. Qualify output keys as ``<node_name>.<attr>``.
    5. Commit qualified outputs via StateWriter.

    Validation hooks:
    - Input presence: all declared inputs must exist in State.
    - Undeclared output attrs: forward() must not return attrs not in spec.
    - Output key collision: handled by StateWriter.
    - Numerical safety: NaN/Inf check when strict_mode=True.

    MUST NOT deep-copy the full State at each node.

    Usage::

        engine = GraphEngine(registry, debug=True)
        trace = engine.run(plan, state, writer, context)
    """

    def __init__(
        self,
        registry: NodeRegistry,
        *,
        debug: bool = False,
        strict_mode: bool = False,
    ) -> None:
        """
        Args:
            registry:    NodeRegistry with registered specs and runtime classes.
            debug:       If True, emit per-node execution events and build a trace.
            strict_mode: If True, validate numerical safety (NaN/Inf) on outputs.
        """
        self._registry = registry
        self._debug = debug
        self._strict_mode = strict_mode

        # Cache of already-built runtimes (node_name -> NodeRuntime)
        self._runtime_cache: dict[str, Any] = {}

    def run(
        self,
        plan: GraphPlan,
        state: State,
        writer: StateWriter,
        context: ExecutionContext,
    ) -> GraphTrace:
        """Execute the graph plan and return a trace.

        Args:
            plan:    Validated, ordered GraphPlan from GraphPlanner.
            state:   Current pipeline State (read-only surface).
            writer:  StateWriter for committing outputs.
            context: ExecutionContext for the current forward pass.

        Returns:
            GraphTrace (empty events list when debug=False).
        """
        trace = GraphTrace(node_order=list(plan.order))
        defn_map = plan.definition_map()

        for node_name in plan.order:
            defn = defn_map[node_name]
            spec = self._registry.get_spec(defn.type_key)

            # 1. Materialize declared inputs (shallow — no full State copy)
            inputs = self._materialize_inputs(node_name, defn.inputs, state)

            # 2. Get or build runtime
            runtime = self._get_or_build_runtime(defn)

            # 3. Forward
            raw_outputs: dict[str, Any] = runtime.forward(inputs, context)

            # 4. Validate outputs against spec
            self._validate_outputs(node_name, spec.output_attrs, raw_outputs)

            # 5. Qualify output keys and optionally check numerical safety
            qualified: dict[str, Any] = {}
            for attr, value in raw_outputs.items():
                key = f"{node_name}.{attr}"
                validate_graph_key(key)
                if self._strict_mode:
                    _check_numerical_safety(key, value, node_name)
                qualified[key] = value

            # 6. Commit via StateWriter
            writer.commit(node_name, qualified)

            # 7. Emit trace event if debug mode
            if self._debug:
                trace.events.append(
                    NodeExecutionEvent(
                        node_name=node_name,
                        inputs_consumed=list(defn.inputs),
                        outputs_produced=list(qualified.keys()),
                    )
                )

        return trace

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _materialize_inputs(
        self,
        node_name: str,
        declared_inputs: list[str],
        state: State,
    ) -> dict[str, Any]:
        """Return a dict of {key: value} for all declared inputs.

        Only reads the declared keys — does NOT copy the full State.

        Raises:
            MissingDependencyError: If a declared input key is absent from State.
        """
        inputs: dict[str, Any] = {}
        for key in declared_inputs:
            if not state.has(key):
                raise MissingDependencyError(
                    f"Node {node_name!r} requires input {key!r} " f"which is not present in State.",
                    stage="graph",
                    component=node_name,
                    key=key,
                )
            inputs[key] = state.get(key)
        return inputs

    def _get_or_build_runtime(self, defn: Any) -> Any:
        """Return cached runtime or build a new one."""
        if defn.name not in self._runtime_cache:
            self._runtime_cache[defn.name] = self._registry.build_runtime(defn)
        return self._runtime_cache[defn.name]

    def _validate_outputs(
        self,
        node_name: str,
        declared_attrs: list[str],
        raw_outputs: dict[str, Any],
    ) -> None:
        """Ensure forward() only returned declared output attrs.

        Raises:
            InvalidConfigError: If an undeclared attr is returned.
        """
        declared_set = set(declared_attrs)
        for attr in raw_outputs:
            if attr not in declared_set:
                raise InvalidConfigError(
                    f"Node {node_name!r} returned undeclared output attr {attr!r}. "
                    f"Declared attrs: {sorted(declared_set)}.",
                    stage="graph",
                    component=node_name,
                    key=f"{node_name}.{attr}",
                )


# ---------------------------------------------------------------------------
# Numerical safety helper
# ---------------------------------------------------------------------------


def _check_numerical_safety(key: str, value: Any, node_name: str) -> None:
    """Raise NumericalInstabilityError if *value* contains NaN or Inf.

    Only checks objects that expose an ``isnan``/``isinf`` interface
    (e.g. torch.Tensor, numpy.ndarray).  Plain Python scalars are checked
    via the ``math`` module.
    """
    import math

    try:
        # torch.Tensor / numpy.ndarray path
        import_ok = False
        try:
            import torch

            if isinstance(value, torch.Tensor):
                import_ok = True
                if not torch.isfinite(value).all():
                    raise NumericalInstabilityError(
                        f"Node {node_name!r} produced non-finite value for key {key!r}.",
                        stage="graph",
                        component=node_name,
                        key=key,
                    )
        except ImportError:
            pass

        if not import_ok:
            try:
                import numpy as np

                if isinstance(value, np.ndarray):
                    if not np.isfinite(value).all():
                        raise NumericalInstabilityError(
                            f"Node {node_name!r} produced non-finite value for key {key!r}.",
                            stage="graph",
                            component=node_name,
                            key=key,
                        )
                    return
            except ImportError:
                pass

            # Plain Python scalar
            if isinstance(value, (int, float)):
                if not math.isfinite(value):
                    raise NumericalInstabilityError(
                        f"Node {node_name!r} produced non-finite value for key {key!r}.",
                        stage="graph",
                        component=node_name,
                        key=key,
                    )
    except NumericalInstabilityError:
        raise
    except Exception:
        # If we can't check, skip silently
        pass
