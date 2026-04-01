"""CLI shim: ``python static_eval/analyze_bootstrap.py`` from repo root."""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from static_eval.analysis.bootstrap import main

if __name__ == "__main__":
    main()
