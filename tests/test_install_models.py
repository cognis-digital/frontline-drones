"""Tests for install_models.py — the NVIDIA HF model installer helpers."""

from __future__ import annotations

import importlib

import pytest

im = importlib.import_module("install_models")


def test_load_models_matches_catalog():
    models = im.load_models()
    assert len(models) == 19
    assert {"repo_id", "category", "license", "url", "description"} <= set(models[0])


def test_categories_sorted_unique():
    cats = im.categories(im.load_models())
    assert cats == sorted(set(cats))
    assert "segmentation" in cats
    assert set(im.SPECIAL) <= set(cats)


def test_install_commands_include_download():
    seg = next(m for m in im.load_models() if m["category"] == "segmentation")
    cmds = im.install_commands(seg)
    assert any("huggingface-cli download" in c for c in cmds)
    assert any(c.startswith("pip install") for c in cmds)


def test_install_commands_special_category_has_no_pip_runner():
    # world_model/robotics/speech have no generic RUNNERS entry -> only the download cmd
    models = im.load_models()
    special = next(m for m in models if m["category"] in im.SPECIAL and m["category"] not in im.RUNNERS)
    cmds = im.install_commands(special)
    assert cmds == [f"huggingface-cli download {special['repo_id']}"]


def test_snippet_for_known_category_embeds_repo():
    seg = next(m for m in im.load_models() if m["category"] == "segmentation")
    snip = im.snippet_for(seg)
    assert snip is not None
    assert seg["repo_id"] in snip


@pytest.mark.parametrize("category", list(im.SPECIAL))
def test_special_categories_never_fake_a_snippet(category):
    m = next((x for x in im.load_models() if x["category"] == category), None)
    if m is not None:
        assert im.snippet_for(m) is None


def test_snippet_device_substitution():
    seg = next(m for m in im.load_models() if m["category"] == "segmentation")
    snip = im.snippet_for(seg, device="cuda")
    assert "cuda" in snip
    assert "{device}" not in snip


def test_every_model_is_actionable():
    # Each model must yield either a runnable snippet or a documented special note.
    for m in im.load_models():
        has_snippet = im.snippet_for(m) is not None
        is_special = m["category"] in im.SPECIAL
        assert has_snippet or is_special, m["repo_id"]


def test_selftest_returns_zero(capsys):
    assert im.selftest() == 0
    assert "selftest passed" in capsys.readouterr().out
