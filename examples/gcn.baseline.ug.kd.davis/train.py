"""
Hybrid — GCN teacher + baseline student, UG fusion, KD loss, DAVIS dataset.

Slot convention: <teacher>.<student>.<fusion>.<loss>.<data>
  teacher : gcn
  student : baseline
  fusion  : ug
  loss    : kd
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples.gcn.baseline.ug.kd.davis.train
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
    "loss=kd",
    "data=tdc_davis_s4",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
