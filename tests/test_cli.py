"""End-to-end tests driving the argparse CLI via ``main()``."""

from __future__ import annotations

import json

import pytest

from frontline_drones import cli


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, out


def test_datasets_lists_all(capsys):
    code, out = run(capsys, "datasets")
    assert code == 0
    assert "military" in out and "commercial" in out and "nvidia" in out
    assert "24 rows" in out.replace("  24", " 24") or "24" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "frontline-drones" in capsys.readouterr().out


def test_no_command_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_list_default_table(capsys):
    code, out = run(capsys, "list", "military")
    assert code == 0
    assert "id" in out and "shahed-136" in out


def test_list_json_is_valid_and_filtered(capsys):
    code, out = run(capsys, "list", "military", "--where", "country=US", "--format", "json")
    assert code == 0
    data = json.loads(out)
    assert data and all("US" in r["country"] for r in data)


def test_list_where_repeatable_anded(capsys):
    _, out = run(capsys, "list", "military", "--where", "country=US",
                 "--where", "role=loitering", "--format", "json")
    data = json.loads(out)
    ids = {r["id"] for r in data}
    assert "switchblade-300" in ids
    assert all("loiter" in r["role"].lower() for r in data)


def test_list_limit_and_columns(capsys):
    _, out = run(capsys, "list", "military", "--limit", "3", "--columns", "id", "--format", "csv")
    lines = out.strip().splitlines()
    assert lines[0] == "id"
    assert len(lines) == 1 + 3


def test_list_sort_numeric(capsys):
    _, out = run(capsys, "list", "military", "--sort", "range_km", "--numeric",
                 "--columns", "id,range_km", "--format", "json")
    data = json.loads(out)
    nums = []
    from frontline_drones.catalog import leading_number
    for r in data:
        n = leading_number(r["range_km"])
        if n is not None:
            nums.append(n)
    assert nums == sorted(nums)


def test_search_subcommand(capsys):
    _, out = run(capsys, "search", "nvidia", "segformer", "--format", "json")
    data = json.loads(out)
    assert data and any("segformer" in r["repo_id"].lower() for r in data)


def test_search_fields_restriction(capsys):
    _, out = run(capsys, "search", "military", "AeroVironment", "--fields", "manufacturer",
                 "--format", "json")
    data = json.loads(out)
    assert data and all("aerovironment" in r["manufacturer"].lower() for r in data)


def test_stats_table(capsys):
    _, out = run(capsys, "stats", "military", "--by", "country")
    assert "distinct" in out


def test_stats_json(capsys):
    _, out = run(capsys, "stats", "nvidia", "--by", "category", "--format", "json")
    data = json.loads(out)
    assert data[0]["count"]  # most common first
    total = sum(int(d["count"]) for d in data)
    assert total == 19


def test_stats_split_multivalue(capsys):
    _, out = run(capsys, "stats", "military", "--by", "operators", "--split", ",",
                 "--format", "json")
    data = json.loads(out)
    values = {d["value"] for d in data}
    assert "UA" in values and "RU" in values


def test_findings_format(capsys):
    _, out = run(capsys, "list", "commercial", "--where", "vendor=DJI", "--format", "findings")
    data = json.loads(out)
    assert data and all(f["source"] == "frontline-drones/commercial" for f in data)
    assert all(set(f) == {"id", "title", "source", "url", "dataset", "attributes"} for f in data)


def test_export_to_file(capsys, tmp_path):
    out_file = tmp_path / "mil.json"
    code, _ = run(capsys, "export", "military", "--format", "json", "--out", str(out_file))
    assert code == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data) == 24


def test_bad_where_syntax_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["list", "military", "--where", "novalue"])
    assert exc.value.code == 2


def test_unknown_dataset_exits_2():
    with pytest.raises(SystemExit) as exc:
        cli.main(["list", "starships"])
    assert exc.value.code == 2


def test_parse_where_helper():
    assert cli._parse_where(["a=b", "c=d=e"]) == {"a": "b", "c": "d=e"}
    with pytest.raises(ValueError):
        cli._parse_where(["noequals"])
