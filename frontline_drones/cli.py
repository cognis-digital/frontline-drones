"""Command-line interface for the frontline-drones catalog.

Exposes the query engine as the ``frontline-drones`` command (and
``python -m frontline_drones``). Subcommands:

* ``datasets``          — list the available datasets.
* ``list <dataset>``    — filter/sort/limit rows and render them.
* ``search <dataset> Q``— full-text search across row values.
* ``stats <dataset>``   — value counts for a column (optionally multi-valued).
* ``export <dataset>``  — render selected columns to a file or stdout.

Every rendering subcommand supports ``--format {table,json,md,csv,findings}``.
``--format json`` / ``findings`` produce machine-readable output suitable for
piping into downstream integrations (see INTEGRATIONS.md).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__
from .catalog import (
    DATASETS,
    filter_rows,
    load_dataset,
    resolve_dataset,
    search,
    sort_rows,
    stats,
)
from .export import to_csv, to_findings, to_json, to_markdown, to_table

FORMATS = ("table", "json", "md", "csv", "findings")


def _parse_where(pairs: Sequence[str] | None) -> dict[str, str]:
    """Turn ``["country=RU", "role=loiter"]`` into a criteria dict."""
    criteria: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--where expects COLUMN=VALUE, got {item!r}")
        col, _, val = item.partition("=")
        col = col.strip()
        if not col:
            raise ValueError(f"--where expects COLUMN=VALUE, got {item!r}")
        criteria[col] = val
    return criteria


def _render(rows, ds, args) -> str:
    columns = None
    if getattr(args, "columns", None):
        columns = [c.strip() for c in args.columns.split(",") if c.strip()]
    fmt = args.format
    if fmt == "json":
        return to_json(rows, columns=columns)
    if fmt == "csv":
        return to_csv(rows, columns=columns)
    if fmt == "md":
        return to_markdown(rows, columns=columns or list(ds.default_columns))
    if fmt == "findings":
        title_field = "name" if "name" in (rows[0] if rows else {}) else ds.key
        url_field = ds.default_columns[-1]
        return to_json(
            to_findings(
                rows, dataset=ds.name, key=ds.key, title_field=title_field, url_field=url_field
            )
        )
    # table (default)
    return to_table(rows, columns=columns or list(ds.default_columns))


def _apply_common(rows, args):
    if getattr(args, "where", None):
        rows = filter_rows(rows, _parse_where(args.where), exact=getattr(args, "exact", False))
    if getattr(args, "sort", None):
        rows = sort_rows(rows, args.sort, numeric=getattr(args, "numeric", False),
                         reverse=getattr(args, "reverse", False))
    limit = getattr(args, "limit", None)
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def _add_query_opts(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("dataset", help="dataset name or alias (see `datasets`)")
    sub.add_argument("--where", action="append", metavar="COL=VALUE",
                     help="filter (repeatable); case-insensitive substring by default")
    sub.add_argument("--exact", action="store_true", help="make --where match whole values")
    sub.add_argument("--sort", metavar="COL", help="sort by a column")
    sub.add_argument("--numeric", action="store_true", help="sort by leading number in the column")
    sub.add_argument("--reverse", action="store_true", help="reverse the sort")
    sub.add_argument("--columns", metavar="A,B,C", help="comma-separated columns to render")
    sub.add_argument("--limit", type=int, default=None, metavar="N", help="cap the number of rows")
    sub.add_argument("--format", choices=FORMATS, default="table", help="output format")
    sub.add_argument("--out", metavar="PATH", help="write to a file instead of stdout")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frontline-drones",
        description="Query the citation-grade drone catalog (military, commercial, NVIDIA models).",
    )
    p.add_argument("--version", action="version", version=f"frontline-drones {__version__}")
    subs = p.add_subparsers(dest="command", required=True)

    subs.add_parser("datasets", help="list available datasets")

    lst = subs.add_parser("list", help="filter and render rows")
    _add_query_opts(lst)

    srch = subs.add_parser("search", help="full-text search across row values")
    _add_query_opts(srch)
    srch.add_argument("query", help="text to search for")
    srch.add_argument("--fields", metavar="A,B", help="restrict search to these columns")

    st = subs.add_parser("stats", help="value counts for a column")
    st.add_argument("dataset", help="dataset name or alias")
    st.add_argument("--by", required=True, metavar="COL", help="column to count")
    st.add_argument("--split", metavar="SEP", help="split multi-valued cells on SEP (e.g. ', ')")
    st.add_argument("--format", choices=("table", "json", "csv"), default="table")
    st.add_argument("--out", metavar="PATH", help="write to a file instead of stdout")

    exp = subs.add_parser("export", help="render selected columns to a file or stdout")
    _add_query_opts(exp)

    return p


def _emit(text: str, out: str | None) -> None:
    if out:
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")
    else:
        print(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "datasets":
        lines = [f"{ds.name:<11} {len(load_dataset(ds.name)):>3} rows  {ds.title}"
                 for ds in DATASETS.values()]
        print("\n".join(lines))
        return 0

    try:
        ds = resolve_dataset(args.dataset)
    except KeyError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover - argparse.error raises SystemExit

    rows = load_dataset(ds.name)

    if args.command == "stats":
        counter = stats(rows, args.by, split=args.split)
        if args.format == "json":
            text = to_json([{ "value": v, "count": str(c)} for v, c in counter.most_common()])
        elif args.format == "csv":
            text = to_csv([{"value": v, "count": str(c)} for v, c in counter.most_common()],
                          columns=["value", "count"])
        else:
            width = max((len(v) for v in counter), default=5)
            text = "\n".join(f"{v.ljust(width)}  {c}" for v, c in counter.most_common())
            text += f"\n\n{sum(counter.values())} value(s) across {len(counter)} distinct."
        _emit(text, args.out)
        return 0

    if args.command == "search":
        fields = [f.strip() for f in args.fields.split(",")] if getattr(args, "fields", None) else None
        rows = search(rows, args.query, fields=fields)

    try:
        rows = _apply_common(rows, args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # pragma: no cover

    _emit(_render(rows, ds, args), args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
