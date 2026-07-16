"""Tests for scripts/validate.py — the stdlib catalog validator."""

from __future__ import annotations

import importlib

validate = importlib.import_module("validate")


def test_real_catalog_passes():
    assert validate.main() == 0


def test_each_real_file_has_no_problems():
    for fname, spec in validate.SPEC.items():
        assert validate.check(fname, spec) == 0, fname


def _spec_for(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "DATA", str(tmp_path))
    return {"key": "id", "url": "source_url", "required": {"id", "name", "source_url"}}


def test_missing_file_is_a_problem(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    assert validate.check("nope.csv", spec) == 1


def test_empty_file_is_a_problem(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    (tmp_path / "f.csv").write_text("id,name,source_url\n", encoding="utf-8")
    assert validate.check("f.csv", spec) == 1


def test_missing_required_column(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    (tmp_path / "f.csv").write_text("id,name\nx,Y\n", encoding="utf-8")
    assert validate.check("f.csv", spec) >= 1


def test_empty_key_flagged(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    (tmp_path / "f.csv").write_text(
        "id,name,source_url\n,Y,https://x.io\n", encoding="utf-8"
    )
    assert validate.check("f.csv", spec) >= 1


def test_duplicate_key_flagged(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    (tmp_path / "f.csv").write_text(
        "id,name,source_url\nx,A,https://x.io\nx,B,https://y.io\n", encoding="utf-8"
    )
    assert validate.check("f.csv", spec) >= 1


def test_non_url_source_flagged(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    (tmp_path / "f.csv").write_text(
        "id,name,source_url\nx,A,not-a-url\n", encoding="utf-8"
    )
    assert validate.check("f.csv", spec) >= 1


def test_clean_file_passes(tmp_path, monkeypatch):
    spec = _spec_for(tmp_path, monkeypatch)
    (tmp_path / "f.csv").write_text(
        "id,name,source_url\nx,A,https://x.io\ny,B,http://y.io\n", encoding="utf-8"
    )
    assert validate.check("f.csv", spec) == 0


def test_spec_covers_all_three_datasets():
    assert set(validate.SPEC) == {
        "military_drones.csv",
        "commercial_platforms.csv",
        "nvidia_hf_models.csv",
    }
