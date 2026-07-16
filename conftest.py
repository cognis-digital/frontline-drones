"""Pytest bootstrap: make the repo-root and scripts/ modules importable.

The package ``frontline_drones`` is importable once installed (``pip install -e .``),
but the standalone tools live at the repo root (``install_models.py``,
``livesearch.py``) and in ``scripts/`` (``validate.py``). Add both locations to
``sys.path`` so the test suite can import and exercise them directly.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)
