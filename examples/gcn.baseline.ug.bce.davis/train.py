"""
Hybrid — GCN teacher + baseline student, UG fusion, BCE loss, DAVIS dataset.

Slot convention: <teacher>.<student>.<fusion>.<loss>.<data>
  teacher : gcn
  student : baseline
  fusion  : ug
  loss    : bce
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples.gcn.baseline.ug.bce.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=hybrid",
    "teacher=gcn",
    "student=baseline",
    "fusion=ug",
    "loss=bce",
    "data=tdc_davis_s4",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
