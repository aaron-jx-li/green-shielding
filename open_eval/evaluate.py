"""CLI shim: run from repo root as ``python open_eval/evaluate.py``."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from open_eval.cli.evaluate import main

if __name__ == "__main__":
    main()
