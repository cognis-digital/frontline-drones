# Architecture

`frontline-drones` is a **data-first** project: the source of truth is a set of
citation-grade CSV datasets. Everything else — the validator, the model
installer, and the query CLI — is thin, dependency-light tooling layered on top
of that data. This document explains the layout and the data flow.

## Layout

```
frontline-drones/
├── data/                       # the source of truth (CC BY 4.0)
│   ├── military_drones.csv     #   key: id
│   ├── commercial_platforms.csv#   key: id
│   └── nvidia_hf_models.csv    #   key: repo_id
├── scripts/
│   └── validate.py             # stdlib schema/URL/uniqueness validator (CI gate)
├── install_models.py           # stdlib menu that installs & runs NVIDIA HF models
├── livesearch.py               # stdlib keyless live web-search / feed ingestion
├── frontline_drones/           # the query engine + CLI (stdlib, importable)
│   ├── catalog.py              #   load / filter / search / distinct / stats / sort
│   ├── export.py               #   render: json / csv / markdown / table / findings
│   ├── cli.py                  #   argparse CLI -> `frontline-drones`
│   └── __main__.py             #   `python -m frontline_drones`
├── docs/                       # human-readable write-ups
├── tests/                      # pytest suite covering every module
└── pyproject.toml              # packaging + console entry point + tool config
```

## The three layers

### 1. Data layer (`data/*.csv`)

Plain CSV, one row per catalogued item, one primary source per row. The schema is
deliberately flat and human-diffable so that additions are reviewable in a pull
request. Each dataset declares a **primary key** (`id`, or `repo_id` for the
NVIDIA models) that must be unique and non-empty.

Published specifications are frequently ranges or annotated
(`"40-70 (Izd-53 >200 reported)"`). The data keeps the original text verbatim;
numeric interpretation is done at query time (see `leading_number`) so nothing is
lost.

### 2. Validation & tooling layer

- **`scripts/validate.py`** — the CI gate. For each dataset it checks that the
  file parses, the required columns exist, the primary key is present and unique,
  and the source/URL column holds an `http(s)` URL. Standard library only, ASCII
  output, non-zero exit on any problem.
- **`install_models.py`** — turns the descriptive NVIDIA index into working
  tooling. A menu walks *category → model → action* and either writes a real,
  runnable `transformers` snippet (for categories that run there) or downloads the
  weights and points at the model card (for categories that need a special
  runtime). `--selftest` runs non-interactive checks in CI.
- **`livesearch.py`** — a self-contained, keyless real-time web-search and
  feed-ingestion helper (Google News RSS, generic RSS/Atom, DuckDuckGo HTML
  fallback) with de-duplication and recency filtering. Standard library only.

### 3. Query layer (`frontline_drones/`)

A small, importable engine with a clean separation between *querying* and
*rendering*:

- **`catalog.py`** — a `Dataset` registry (name, filename, key, default columns,
  aliases) plus pure functions that operate on `list[dict]` rows:
  `load_dataset` / `load_all`, `filter_rows`, `search`, `distinct`, `stats`,
  `sort_rows`, and the `leading_number` helper.
- **`export.py`** — renderers that take rows and return text: `to_json`,
  `to_csv`, `to_markdown`, `to_table`, plus `to_findings` for a normalized,
  integration-friendly record shape.
- **`cli.py`** — an `argparse` front end wiring the two together into the
  `datasets`, `list`, `search`, `stats`, and `export` subcommands.

## Data flow

```
data/*.csv ──load_dataset──▶ list[dict] ──filter/search/sort/stats──▶ list[dict]
                                                                          │
                                            to_json/csv/markdown/table/findings
                                                                          │
                                                              stdout or --out file
                                                                          │
                                                    (findings) ─▶ integration SDK
```

Rows stay as plain dictionaries end-to-end, so results round-trip losslessly back
to CSV or JSON and remain faithful to the published source data.

## Design constraints

- **Standard library at runtime.** The query engine and all three tools import
  nothing outside the Python standard library. `pytest`/`ruff` are dev-only.
- **Additive and stable.** The CSV schemas, the validator's required-column
  contract, the installer, and the CLI surface are treated as public contracts.
- **Reference scope only.** No operational, flight, guidance, or targeting logic
  lives anywhere in the codebase; see [DISCLAIMER.md](../DISCLAIMER.md).
