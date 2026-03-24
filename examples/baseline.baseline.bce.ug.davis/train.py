"""
Hybrid — baseline teacher + baseline student, BCE loss, UG Fusion, DAVIS.

Slot convention: <teacher>.<student>.<fusion>.<loss>.<data>
  teacher : baseline
  student : baseline
  fusion  : ug
  loss    : bce
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples.baseline.baseline.bce.ug.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=baseline.baseline.ug",
    "data=tdc_davis",
    "trainer=default_trainer",
    "trainer.loss.name=bce",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
