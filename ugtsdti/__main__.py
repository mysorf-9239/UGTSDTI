"""Module entrypoint for ``python -m ugtsdti``."""

from __future__ import annotations

import sys

from ugtsdti.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
