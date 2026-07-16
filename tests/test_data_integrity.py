"""Integrity checks on the shipped catalog data itself (beyond schema).

These assert catalog-quality invariants that the CSV validator does not: unique
keys, sane URLs, non-empty descriptive fields, and dataset-specific conventions.
"""

from __future__ import annotations

import re

import pytest

from frontline_drones import catalog

DATASETS = list(catalog.DATASETS.values())


@pytest.mark.parametrize("ds", DATASETS, ids=lambda d: d.name)
def test_keys_are_unique(ds):
    rows = catalog.load_dataset(ds.name)
    keys = [r[ds.key] for r in rows]
    assert len(keys) == len(set(keys)), f"duplicate {ds.key} in {ds.name}"


@pytest.mark.parametrize("ds", DATASETS, ids=lambda d: d.name)
def test_no_key_is_blank(ds):
    rows = catalog.load_dataset(ds.name)
    assert all(r[ds.key].strip() for r in rows)


@pytest.mark.parametrize("ds", DATASETS, ids=lambda d: d.name)
def test_default_columns_present_in_every_row(ds):
    rows = catalog.load_dataset(ds.name)
    for row in rows:
        assert set(ds.default_columns) <= set(row), ds.name


def _url_columns():
    return {"military": "source_url", "commercial": "docs_url", "nvidia": "url"}


@pytest.mark.parametrize("name,col", _url_columns().items())
def test_urls_are_http(name, col):
    rows = catalog.load_dataset(name)
    for row in rows:
        assert re.match(r"^https?://", row[col]), f"{name}:{row}"


def test_military_countries_are_uppercase_codes():
    rows = catalog.load_dataset("military")
    for row in rows:
        # country codes like "US", "RU", "IR/RU", "TR"
        assert re.fullmatch(r"[A-Z]{2}(/[A-Z]{2})*", row["country"]), row["country"]


def test_military_roles_are_snake_case():
    rows = catalog.load_dataset("military")
    for row in rows:
        assert re.fullmatch(r"[a-z0-9_]+", row["role"]), row["role"]


def test_nvidia_repo_ids_are_namespaced():
    rows = catalog.load_dataset("nvidia")
    for row in rows:
        assert row["repo_id"].startswith("nvidia/")
        assert row["url"].endswith(row["repo_id"])


def test_commercial_license_and_category_nonempty():
    rows = catalog.load_dataset("commercial")
    for row in rows:
        assert row["license"].strip()
        assert row["category"].strip()


def test_every_row_has_a_display_name():
    for ds in DATASETS:
        rows = catalog.load_dataset(ds.name)
        name_col = "name" if ds.name != "nvidia" else "repo_id"
        assert all(r[name_col].strip() for r in rows)
