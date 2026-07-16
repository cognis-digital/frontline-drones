"""Enable ``python -m frontline_drones`` as an entry point to the CLI."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
