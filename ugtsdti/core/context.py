"""ExecutionContext — lightweight runtime controls passed through the hot path.

REQ-REPRO-001, REQ-QUAL-002
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ExecutionContext:
    """Runtime controls needed at every stage of the pipeline.

    Attributes:
        mode:          One of "train", "eval", or "infer".
        seed:          Global random seed for this run.
        device:        Target device string, e.g. "cpu" or "cuda:0".
        deterministic: Whether to enable deterministic algorithm mode.
        precision:     Floating-point precision mode; defaults to "fp32".
    """

    mode: Literal["train", "eval", "infer"]
    seed: int
    device: str
    deterministic: bool
    precision: Literal["fp32", "mixed"] = field(default="fp32")
