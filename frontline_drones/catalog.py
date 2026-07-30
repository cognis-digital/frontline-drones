"""Loading, filtering, searching and aggregating the catalog CSVs.

Pure standard library. Every function operates on plain ``list[dict[str, str]]``
rows exactly as read from the CSV files, so results round-trip losslessly back to
CSV/JSON and stay faithful to the published source data.
"""

from __future__ import annotations

import csv
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

Row = dict[str, str]


@dataclass(frozen=True)
class Dataset:
    """Metadata describing one catalog CSV.

    Attributes:
        name: Short canonical name used on the CLI (e.g. ``"military"``).
        filename: CSV file name inside ``data/``.
        key: Primary-key column whose values are unique per row.
        title: Human-readable label.
        default_columns: Columns shown by the table/markdown renderers by default.
        aliases: Alternate names accepted by :func:`resolve_dataset`.
    """

    name: str
    filename: str
    key: str
    title: str
    default_columns: tuple[str, ...]
    aliases: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, Dataset] = {
    "military": Dataset(
        name="military",
        filename="military_drones.csv",
        key="id",
        title="Military systems (UAVs, loitering munitions, USVs)",
        default_columns=("id", "name", "country", "role", "range_km", "source_url"),
        aliases=("mil", "military_drones", "drones"),
    ),
    "commercial": Dataset(
        name="commercial",
        filename="commercial_platforms.csv",
        key="id",
        title="Commercial platforms & open autonomy stacks",
        default_columns=("id", "name", "vendor", "category", "license", "docs_url"),
        aliases=("com", "commercial_platforms", "platforms", "open"),
    ),
    "nvidia": Dataset(
        name="nvidia",
        filename="nvidia_hf_models.csv",
        key="repo_id",
        title="NVIDIA open Hugging Face models",
        default_columns=("repo_id", "modality", "category", "license", "url"),
        aliases=("nv", "models", "hf", "nvidia_hf_models"),
    ),
    # --- Defensive counter-UAS DETECTION reference datasets -----------------
    # Classification/situational-awareness reference only: these catalog what a
    # PASSIVE detector would observe (RF/acoustic/radar signatures) and which
    # public corpora and detection products exist. They contain no transmit,
    # jam, spoof, or engagement parameters. See docs/detection-signatures.md.
    "rf": Dataset(
        name="rf",
        filename="rf-signatures.csv",
        key="id",
        title="RF control-link signatures (passive classification reference)",
        default_columns=("id", "name", "control_band", "video_protocol",
                         "frequency_hopping", "source_url"),
        aliases=("rf_signatures", "rf-signatures", "rfsig", "controllink"),
    ),
    "acoustic": Dataset(
        name="acoustic",
        filename="acoustic-signatures.csv",
        key="id",
        title="Acoustic BPF signatures (passive detection reference)",
        default_columns=("id", "name", "rotor_class", "bpf_fundamental_hz",
                         "detect_range_m", "source_url"),
        aliases=("acoustic_signatures", "acoustic-signatures", "sound", "bpf"),
    ),
    "radar": Dataset(
        name="radar",
        filename="radar-signatures.csv",
        key="id",
        title="Micro-Doppler radar signatures + bird discriminants (reference)",
        default_columns=("id", "name", "platform_class", "recommended_band",
                         "bird_discriminant", "source_url"),
        aliases=("radar_signatures", "radar-signatures", "microdoppler", "micro-doppler"),
    ),
    "corpora": Dataset(
        name="corpora",
        filename="detection-datasets.csv",
        key="id",
        title="Public C-UAS detection datasets / training corpora index",
        default_columns=("id", "name", "modality", "platforms", "license", "url"),
        aliases=("detection_datasets", "detection-datasets", "training", "corpus"),
    ),
    "detectors": Dataset(
        name="detectors",
        filename="cuas-detection-systems.csv",
        key="id",
        title="Detection-only C-UAS systems (descriptive, non-endorsing)",
        default_columns=("id", "name", "vendor", "sensor_mix", "detection_role", "source_url"),
        aliases=("cuas", "detection_systems", "detection-systems", "systems"),
    ),
}


def resolve_dataset(name: str) -> Dataset:
    """Return the :class:`Dataset` for ``name`` (canonical name or alias).

    Raises:
        KeyError: if the name matches no dataset, with the valid names listed.
    """
    key = name.strip().lower()
    if key in DATASETS:
        return DATASETS[key]
    for ds in DATASETS.values():
        if key in ds.aliases:
            return ds
    valid = ", ".join(sorted(DATASETS))
    raise KeyError(f"unknown dataset {name!r}; choose one of: {valid}")


def load_dataset(name: str, *, data_dir: str | None = None) -> list[Row]:
    """Load one dataset's rows as a list of dicts (header order preserved)."""
    ds = resolve_dataset(name)
    path = os.path.join(data_dir or DATA_DIR, ds.filename)
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def load_all(*, data_dir: str | None = None) -> dict[str, list[Row]]:
    """Load every dataset, keyed by canonical dataset name."""
    return {name: load_dataset(name, data_dir=data_dir) for name in DATASETS}


def _matches(value: str, term: str, *, exact: bool, ignore_case: bool) -> bool:
    if ignore_case:
        value, term = value.lower(), term.lower()
    return value == term if exact else term in value


def filter_rows(
    rows: Iterable[Row],
    criteria: dict[str, str] | None = None,
    *,
    exact: bool = False,
    ignore_case: bool = True,
) -> list[Row]:
    """Return rows matching every ``column -> term`` pair in ``criteria``.

    By default matching is case-insensitive substring containment; pass
    ``exact=True`` for whole-value equality. Missing columns never match, so a
    criterion on an absent column yields no rows rather than raising.
    """
    criteria = criteria or {}
    out: list[Row] = []
    for row in rows:
        if all(
            _matches(row.get(col, ""), term, exact=exact, ignore_case=ignore_case)
            for col, term in criteria.items()
        ):
            out.append(row)
    return out


def search(
    rows: Iterable[Row],
    term: str,
    *,
    fields: Iterable[str] | None = None,
    ignore_case: bool = True,
) -> list[Row]:
    """Return rows where ``term`` appears in any (or the given) field values."""
    needle = term.lower() if ignore_case else term
    field_set = set(fields) if fields is not None else None
    out: list[Row] = []
    for row in rows:
        for col, val in row.items():
            if field_set is not None and col not in field_set:
                continue
            hay = val.lower() if ignore_case else val
            if needle in hay:
                out.append(row)
                break
    return out


def distinct(rows: Iterable[Row], column: str, *, split: str | None = None) -> list[str]:
    """Return the sorted unique, non-empty values found in ``column``.

    When ``split`` is given, each cell is split on it first (useful for
    multi-valued columns such as ``operators`` = ``"RU, IR"``).
    """
    values: set[str] = set()
    for row in rows:
        cell = (row.get(column) or "").strip()
        if not cell:
            continue
        parts = [p.strip() for p in cell.split(split)] if split else [cell]
        values.update(p for p in parts if p)
    return sorted(values)


def stats(rows: Iterable[Row], column: str, *, split: str | None = None) -> Counter:
    """Count occurrences of each value in ``column`` (most common first).

    Empty cells are ignored. With ``split`` set, multi-valued cells contribute
    one count per part.
    """
    counter: Counter = Counter()
    for row in rows:
        cell = (row.get(column) or "").strip()
        if not cell:
            continue
        parts = [p.strip() for p in cell.split(split)] if split else [cell]
        counter.update(p for p in parts if p)
    return counter


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def leading_number(value: str | None) -> float | None:
    """Extract the first numeric token from a messy spec cell, or ``None``.

    Published specs are frequently ranges or annotated (``"40-70 (Izd-53 >200
    reported)"``). This pulls the first number so rows can be sorted or compared
    numerically without discarding the original text.
    """
    if not value:
        return None
    m = _NUMBER_RE.search(value)
    return float(m.group()) if m else None


def sort_rows(
    rows: Iterable[Row],
    column: str,
    *,
    numeric: bool = False,
    reverse: bool = False,
) -> list[Row]:
    """Return rows sorted by ``column``.

    With ``numeric=True`` the leading number of each cell is used (rows with no
    number sort last regardless of direction); otherwise a case-insensitive
    string sort is applied.
    """
    rows = list(rows)
    present: list[tuple[object, Row]] = []
    missing: list[Row] = []
    for row in rows:
        if numeric:
            value = leading_number(row.get(column))
            blank = value is None
        else:
            value = (row.get(column) or "").strip().lower()
            blank = not value
        (missing.append(row) if blank else present.append((value, row)))
    present.sort(key=lambda kv: kv[0], reverse=reverse)  # type: ignore[arg-type,return-value]
    # Rows lacking a comparable value always sort last, in both directions.
    return [row for _, row in present] + missing
