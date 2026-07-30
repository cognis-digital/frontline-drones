"""Tests for the catalog loading / filtering / aggregation engine."""

from __future__ import annotations

import pytest

from frontline_drones import catalog


def test_datasets_registry_is_well_formed():
    assert set(catalog.DATASETS) == {
        "military", "commercial", "nvidia",
        "rf", "acoustic", "radar", "corpora", "detectors",
    }
    for name, ds in catalog.DATASETS.items():
        assert ds.name == name
        assert ds.filename.endswith(".csv")
        assert ds.key in ds.default_columns
        assert name not in ds.aliases


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("mil", "military"),
        ("MILITARY", "military"),
        ("platforms", "commercial"),
        ("hf", "nvidia"),
        ("nvidia_hf_models", "nvidia"),
    ],
)
def test_resolve_dataset_accepts_aliases_and_case(alias, expected):
    assert catalog.resolve_dataset(alias).name == expected


def test_resolve_dataset_unknown_lists_choices():
    with pytest.raises(KeyError) as exc:
        catalog.resolve_dataset("spaceships")
    msg = str(exc.value)
    assert "military" in msg and "commercial" in msg and "nvidia" in msg


def test_load_dataset_shapes():
    mil = catalog.load_dataset("military")
    assert len(mil) == 24
    assert {"id", "name", "country", "role", "source_url"} <= set(mil[0])
    com = catalog.load_dataset("commercial")
    assert len(com) == 14
    nv = catalog.load_dataset("nvidia")
    assert len(nv) == 19


def test_load_all_returns_every_dataset():
    everything = catalog.load_all()
    assert set(everything) == set(catalog.DATASETS)
    assert everything["military"] == catalog.load_dataset("military")


def test_load_dataset_honours_custom_data_dir(tmp_path):
    csv_path = tmp_path / "military_drones.csv"
    csv_path.write_text("id,name,source_url\nx,X drone,https://example.com\n", encoding="utf-8")
    rows = catalog.load_dataset("military", data_dir=str(tmp_path))
    assert rows == [{"id": "x", "name": "X drone", "source_url": "https://example.com"}]


def test_filter_rows_substring_case_insensitive():
    mil = catalog.load_dataset("military")
    ru = catalog.filter_rows(mil, {"country": "ru"})
    assert ru, "expected some RU-operated systems"
    assert all("ru" in r["country"].lower() for r in ru)


def test_filter_rows_multiple_criteria_are_anded():
    mil = catalog.load_dataset("military")
    both = catalog.filter_rows(mil, {"country": "US", "role": "loitering"})
    assert {r["id"] for r in both} >= {"switchblade-300", "switchblade-600"}
    assert all("loiter" in r["role"].lower() for r in both)


def test_filter_rows_exact_mode():
    mil = catalog.load_dataset("military")
    contains = catalog.filter_rows(mil, {"role": "isr"})
    exact = catalog.filter_rows(mil, {"role": "isr"}, exact=True)
    assert len(exact) < len(contains)
    assert all(r["role"] == "isr" for r in exact)


def test_filter_rows_missing_column_yields_nothing():
    mil = catalog.load_dataset("military")
    assert catalog.filter_rows(mil, {"nonexistent": "x"}) == []


def test_search_across_all_fields():
    nv = catalog.load_dataset("nvidia")
    hits = catalog.search(nv, "segformer")
    assert any("segformer" in r["repo_id"].lower() for r in hits)


def test_search_restricted_fields():
    mil = catalog.load_dataset("military")
    # "US" appears in operators and country; restrict to manufacturer only.
    hits = catalog.search(mil, "AeroVironment", fields=["manufacturer"])
    assert hits and all("aerovironment" in r["manufacturer"].lower() for r in hits)


def test_search_no_match_returns_empty():
    assert catalog.search(catalog.load_dataset("nvidia"), "zzz-not-here") == []


def test_distinct_and_split():
    mil = catalog.load_dataset("military")
    countries = catalog.distinct(mil, "country")
    assert "US" in countries and countries == sorted(countries)
    # operators is comma-separated; splitting expands multi-operator rows.
    joined = catalog.distinct(mil, "operators")
    split = catalog.distinct(mil, "operators", split=",")
    assert len(split) <= len(joined) or "UA" in split


def test_stats_counts_most_common_first():
    mil = catalog.load_dataset("military")
    counts = catalog.stats(mil, "country")
    ordered = [c for _, c in counts.most_common()]
    assert ordered == sorted(ordered, reverse=True)
    assert counts.total() == len(mil)


def test_stats_ignores_empty_cells():
    rows = [{"x": "a"}, {"x": ""}, {"x": "a"}, {"x": None}]  # type: ignore[list-item]
    counts = catalog.stats(rows, "x")
    assert counts == {"a": 2}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2500", 2500.0),
        ("40-70 (Izd-53 >200 reported)", 40.0),
        ("~50 kg", 50.0),
        ("-12.5 offset", -12.5),
        ("n/a (ISR)", None),
        ("", None),
        (None, None),
    ],
)
def test_leading_number(value, expected):
    assert catalog.leading_number(value) == expected


def test_sort_rows_numeric_puts_missing_last():
    rows = [{"r": "100"}, {"r": "n/a"}, {"r": "50"}]
    asc = catalog.sort_rows(rows, "r", numeric=True)
    assert [r["r"] for r in asc] == ["50", "100", "n/a"]


def test_sort_rows_numeric_reverse_keeps_missing_last():
    rows = [{"r": "100"}, {"r": "n/a"}, {"r": "50"}]
    desc = catalog.sort_rows(rows, "r", numeric=True, reverse=True)
    assert desc[-1]["r"] == "n/a"
    assert desc[0]["r"] == "100"


def test_sort_rows_string():
    rows = [{"n": "Charlie"}, {"n": "alpha"}, {"n": "Bravo"}]
    out = catalog.sort_rows(rows, "n")
    assert [r["n"] for r in out] == ["alpha", "Bravo", "Charlie"]
