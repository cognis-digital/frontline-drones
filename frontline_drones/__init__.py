"""frontline_drones — a queryable interface over the citation-grade drone catalog.

This package turns the descriptive CSV datasets shipped in ``data/`` into a small,
dependency-free query engine and command-line tool. It lets you filter, search,
aggregate and export the catalog (military systems, commercial/open platforms and
the NVIDIA Hugging Face model index) in JSON, Markdown, CSV or a plain table.

Scope: this is a *reference* query tool over publicly reported specifications. It
contains no operational, flight-control, guidance or targeting logic. See
``DISCLAIMER.md``.

Public API::

    from frontline_drones import (
        load_dataset, load_all, DATASETS,
        filter_rows, search, stats, distinct, sort_rows,
        to_json, to_markdown, to_csv, to_table, to_findings,
    )
"""

from __future__ import annotations

from .catalog import (
    DATASETS,
    Dataset,
    distinct,
    filter_rows,
    load_all,
    load_dataset,
    resolve_dataset,
    search,
    sort_rows,
    stats,
)
from .export import to_csv, to_findings, to_json, to_markdown, to_table

__version__ = "0.2.0"

__all__ = [
    "DATASETS",
    "Dataset",
    "__version__",
    "distinct",
    "filter_rows",
    "load_all",
    "load_dataset",
    "resolve_dataset",
    "search",
    "sort_rows",
    "stats",
    "to_csv",
    "to_findings",
    "to_json",
    "to_markdown",
    "to_table",
]
