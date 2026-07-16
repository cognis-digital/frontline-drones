"""Tests for the renderers (JSON, CSV, Markdown, table, findings)."""

from __future__ import annotations

import csv
import io
import json

import pytest

from frontline_drones import export

ROWS = [
    {"id": "a1", "name": "Alpha", "url": "https://example.com/a"},
    {"id": "b2", "name": "Bra|vo", "url": "https://example.com/b"},
]


def test_to_json_roundtrips_and_preserves_order():
    text = export.to_json(ROWS)
    parsed = json.loads(text)
    assert parsed == ROWS
    assert list(parsed[0].keys()) == ["id", "name", "url"]


def test_to_json_column_projection():
    parsed = json.loads(export.to_json(ROWS, columns=["id"]))
    assert parsed == [{"id": "a1"}, {"id": "b2"}]


def test_to_json_non_ascii_preserved():
    parsed = json.loads(export.to_json([{"x": "Palianytsia — café"}]))
    assert parsed[0]["x"] == "Palianytsia — café"


def test_to_csv_roundtrips_through_dictreader():
    text = export.to_csv(ROWS)
    back = list(csv.DictReader(io.StringIO(text)))
    assert back == ROWS


def test_to_csv_uses_lf_line_endings():
    text = export.to_csv(ROWS)
    assert "\r\n" not in text
    assert text.count("\n") == 3  # header + 2 rows


def test_to_markdown_structure_and_pipe_escaping():
    md = export.to_markdown(ROWS)
    lines = md.splitlines()
    assert lines[0] == "| id | name | url |"
    assert lines[1] == "| --- | --- | --- |"
    # the literal pipe in "Bra|vo" must be escaped so the table stays valid
    assert "Bra\\|vo" in md


def test_to_table_alignment_has_header_and_rule():
    table = export.to_table(ROWS)
    lines = table.splitlines()
    assert lines[0].startswith("id")
    assert set(lines[1]) <= {"-", " "}
    assert "Alpha" in lines[2]


def test_empty_rows_render_safely():
    assert export.to_json([]) == "[]"
    assert export.to_csv([]) == ""
    assert export.to_markdown([]) == ""
    assert export.to_table([]) == ""


def test_columns_union_when_rows_have_extra_keys():
    rows = [{"a": "1"}, {"a": "2", "b": "3"}]
    # first row defines base order; extra key from later row is appended
    parsed = json.loads(export.to_json(rows))
    assert list(parsed[1].keys()) == ["a", "b"]
    assert parsed[0] == {"a": "1", "b": ""}


def test_to_findings_shape():
    findings = export.to_findings(
        ROWS, dataset="demo", key="id", title_field="name", url_field="url"
    )
    assert findings[0] == {
        "id": "a1",
        "title": "Alpha",
        "source": "frontline-drones/demo",
        "url": "https://example.com/a",
        "dataset": "demo",
        "attributes": ROWS[0],
    }


def test_to_findings_title_falls_back_to_key():
    rows = [{"id": "x", "name": "", "url": "u"}]
    findings = export.to_findings(rows, dataset="d", key="id", title_field="name", url_field="url")
    assert findings[0]["title"] == "x"


@pytest.mark.parametrize("renderer", [export.to_json, export.to_csv, export.to_markdown, export.to_table])
def test_renderers_return_strings(renderer):
    assert isinstance(renderer(ROWS), str)
