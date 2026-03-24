"""
Only Student — ESM-2 protein encoder, BCE loss, no gate, DAVIS dataset.

Slot convention: <teacher>.<student>.<loss>.<gate>.<data>
  teacher : _ (none)
  student : esm
  loss    : bce
  gate    : _ (none)
  data    : davis

Usage:
    conda run -n ugtsdti python -m examples._.esm.bce._.davis.train
"""

import subprocess
import sys

CMD = [
    "python",
    "-m",
    "ugtsdti.main",
    "model=_.esm._",
    "data=tdc_davis",
    "trainer=default_trainer",
    "trainer.loss.name=bce",
]

if __name__ == "__main__":
    sys.exit(subprocess.call(CMD))
