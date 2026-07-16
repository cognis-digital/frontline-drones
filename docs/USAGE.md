# Usage

`frontline-drones` ships a small, dependency-free query tool over the catalog
CSVs, plus two standalone helpers (`scripts/validate.py`, `install_models.py`).
This page documents the query CLI in detail; see the
[README](../README.md) for the catalog contents and
[ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Install

The query engine and CLI are pure standard library. Installing the package just
puts the `frontline-drones` command on your `PATH`:

```bash
pip install -e .            # from a checkout
frontline-drones --version
```

You can also run it without installing:

```bash
python -m frontline_drones datasets
```

## Datasets

Three datasets are available; each has short aliases.

| Name | Aliases | Key column | Contents |
|---|---|---|---|
| `military` | `mil`, `drones` | `id` | Military UAVs, loitering munitions, USVs |
| `commercial` | `com`, `platforms`, `open` | `id` | Commercial drones + open autonomy stacks |
| `nvidia` | `nv`, `models`, `hf` | `repo_id` | NVIDIA open Hugging Face models |

```bash
frontline-drones datasets
```

## Subcommands

### `list` — filter, sort, render

```bash
# All military systems as an aligned table (default format)
frontline-drones list military

# US loitering munitions, JSON, id + role only
frontline-drones list military --where country=US --where role=loitering \
  --columns id,role --format json

# The five longest-ranged systems (numeric sort on a messy column)
frontline-drones list military --sort range_km --numeric --reverse --limit 5 \
  --columns id,name,range_km
```

Options (shared by `list`, `search`, `export`):

| Option | Meaning |
|---|---|
| `--where COL=VALUE` | Filter; repeatable and AND-ed. Case-insensitive substring by default. |
| `--exact` | Make `--where` match whole values instead of substrings. |
| `--sort COL` | Sort by a column. |
| `--numeric` | Sort by the leading number in the column (rows with no number sort last). |
| `--reverse` | Reverse the sort direction. |
| `--columns A,B,C` | Restrict/reorder the rendered columns. |
| `--limit N` | Cap the number of rows. |
| `--format {table,json,md,csv,findings}` | Output format (default `table`). |
| `--out PATH` | Write to a file (UTF-8, `\n` line endings) instead of stdout. |

### `search` — full-text search

```bash
# Any row mentioning "segformer" anywhere
frontline-drones search nvidia segformer --format json

# Restrict the search to specific fields
frontline-drones search military AeroVironment --fields manufacturer
```

### `stats` — value counts

```bash
# Systems per country, most common first
frontline-drones stats military --by country

# Operators, splitting multi-valued cells ("RU, IR") into individual counts
frontline-drones stats military --by operators --split , --format json
```

### `export` — render to a file

`export` is `list` with the same options; it is a convenient name for writing to
a file:

```bash
frontline-drones export commercial --where vendor=DJI --format md --out dji.md
frontline-drones export nvidia --format csv --out nvidia_models.csv
```

## Output formats

- **`table`** — fixed-width, aligned plain text (default; good for terminals).
- **`json`** — array of row objects; key order preserved; non-ASCII kept verbatim.
- **`csv`** — round-trips back through any CSV reader; `\n` line endings.
- **`md`** — GitHub-flavoured Markdown table (pipes in values are escaped).
- **`findings`** — normalized records (`id`, `title`, `source`, `url`, `dataset`,
  `attributes`) designed to map cleanly into downstream tooling. See
  [INTEGRATIONS.md](../INTEGRATIONS.md).

Example — pipe findings into an integration SDK:

```bash
frontline-drones list military --where role=loitering --format findings \
  | cognis-connect emit --to stix --dry-run
```

## Exit codes

- `0` — success.
- `2` — usage error (bad `--where` syntax, unknown dataset, missing subcommand).

## Programmatic use

Everything the CLI does is available as a library:

```python
from frontline_drones import load_dataset, filter_rows, stats, to_json

rows = load_dataset("military")
us_loiter = filter_rows(rows, {"country": "US", "role": "loitering"})
print(to_json(us_loiter, columns=["id", "name", "range_km"]))

print(stats(rows, "country").most_common(3))
```
