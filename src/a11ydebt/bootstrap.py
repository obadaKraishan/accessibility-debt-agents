"""
Import shim so every script runs standalone with `python <script>.py`.

Each phase script starts with:

    import bootstrap  # noqa: F401   (adds src/ to sys.path)

which is friendlier than requiring `pip install -e .` before the first run.
A `pip install -e .` still works and is preferred for real development.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
