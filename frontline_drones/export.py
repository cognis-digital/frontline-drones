"""Render catalog rows into JSON, Markdown, CSV, a plain table, or findings.

All renderers take ``list[dict]`` rows and return a ``str`` (except
:func:`to_findings`, which returns structured records). Column selection is
honoured everywhere so output stays stable and diff-friendly.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence

Row = dict[str, str]


def _columns(rows: Sequence[Row], columns: Iterable[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    if not rows:
        return []
    # Preserve first-row (header) order; union any extra keys seen later.
    ordered = list(rows[0].keys())
    seen = set(ordered)
    for row in rows[1:]:
        for col in row:
            if col not in seen:
                ordered.append(col)
                seen.add(col)
    return ordered


def _project(rows: Sequence[Row], columns: list[str]) -> list[Row]:
    return [{c: row.get(c, "") for c in columns} for row in rows]


def to_json(rows: Sequence[Row], *, columns: Iterable[str] | None = None, indent: int = 2) -> str:
    """Serialise rows to a JSON array of objects (UTF-8 safe, key order kept)."""
    cols = _columns(rows, columns)
    return json.dumps(_project(rows, cols), indent=indent, ensure_ascii=False)


def to_csv(rows: Sequence[Row], *, columns: Iterable[str] | None = None) -> str:
    """Serialise rows back to CSV text with a header line."""
    cols = _columns(rows, columns)
    if not cols:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in cols})
    return buf.getvalue()


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def to_markdown(rows: Sequence[Row], *, columns: Iterable[str] | None = None) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    cols = _columns(rows, columns)
    if not cols:
        return ""
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def to_table(rows: Sequence[Row], *, columns: Iterable[str] | None = None) -> str:
    """Render rows as a fixed-width, monospaced plain-text table."""
    cols = _columns(rows, columns)
    if not cols:
        return ""
    widths = {c: len(c) for c in cols}
    for row in rows:
        for c in cols:
            widths[c] = max(widths[c], len(row.get(c, "")))
    def fmt(values: dict) -> str:
        return "  ".join(str(values.get(c, "")).ljust(widths[c]) for c in cols).rstrip()
    header = fmt({c: c for c in cols})
    rule = "  ".join("-" * widths[c] for c in cols).rstrip()
    body = [fmt(row) for row in rows]
    return "\n".join([header, rule, *body])


def to_findings(
    rows: Sequence[Row],
    *,
    dataset: str,
    key: str,
    title_field: str,
    url_field: str,
) -> list[dict]:
    """Map rows to normalized finding records for downstream integration.

    The shape is intentionally generic (``id``/``title``/``source``/``url``/
    ``dataset``/``attributes``) so an integration SDK can map it into its own
    canonical record without knowing this catalog's per-dataset columns.
    """
    findings: list[dict] = []
    for row in rows:
        findings.append(
            {
                "id": row.get(key, ""),
                "title": row.get(title_field, "") or row.get(key, ""),
                "source": f"frontline-drones/{dataset}",
                "url": row.get(url_field, ""),
                "dataset": dataset,
                "attributes": dict(row),
            }
        )
    return findings
