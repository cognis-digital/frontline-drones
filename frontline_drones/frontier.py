"""Build a large harvest frontier from the drone-detection dataset catalog.

The detection-datasets index and the per-platform signature tables give a rich set
of real sources; crossing them with a global geographic tile grid produces tens of
thousands of concrete harvest endpoints for :mod:`frontline_drones.growth`.
Defensive awareness/analysis only; no targeting content.
"""
from __future__ import annotations

import csv
import io
import os
from typing import List, Optional, Sequence, Tuple

from .growth import Endpoint, expand

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_CSVS = ("detection-datasets.csv", "acoustic-signatures.csv", "rf-signatures.csv",
         "radar-signatures.csv", "cuas-detection-systems.csv")


def _tile_grid(step_deg: int = 5) -> Tuple[str, ...]:
    if step_deg <= 0 or 180 % step_deg != 0:
        raise ValueError("step_deg must be a positive divisor of 180")
    tiles: List[str] = []
    lat = -90
    while lat < 90:
        lon = -180
        while lon < 180:
            tiles.append("{0}_{1}".format(lat, lon))
            lon += step_deg
        lat += step_deg
    return tuple(tiles)


GEO_TILES: Tuple[str, ...] = _tile_grid(5)


def source_pairs() -> List[Tuple[str, str]]:
    """(name, url) pairs harvested from the shipped dataset/signature CSVs."""
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for fname in _CSVS:
        path = os.path.join(_DATA, fname)
        if not os.path.exists(path):
            continue
        with io.open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or row.get("source_url") or "").strip()
                name = (row.get("name") or row.get("id") or "").strip()
                if url.startswith("http") and name and name not in seen:
                    seen.add(name)
                    pairs.append((name, url))
    return pairs


def build_frontier(vocabulary: Optional[Sequence[str]] = None) -> List[Endpoint]:
    """Cross the real dataset/signature sources with a geographic vocabulary."""
    vocab = tuple(vocabulary) if vocabulary is not None else GEO_TILES
    return expand(source_pairs(), vocab, param_name="tile", base_only=True)
