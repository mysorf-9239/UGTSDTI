"""
Hybrid — GCN teacher + baseline student, BCE loss, UG Fusion, DAVIS.

Slot convention: <teacher>.<student>.<fusion>.<loss>.<data>
  teacher : gcn
  student : baseline
  fusion  : ug
  loss    : bce
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples.gcn.baseline.bce.ug.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=gcn.baseline.ug",
    "data=tdc_davis",
    "trainer=default_trainer",
    "trainer.loss.name=bce",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
