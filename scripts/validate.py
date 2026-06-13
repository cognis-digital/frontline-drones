#!/usr/bin/env python3
"""Validate the frontline-drones catalog CSVs. Pure standard library.

Checks, per file: it parses as CSV, the required columns are present, no row has
an empty primary key, and every source/url column holds an http(s) URL. ASCII
output so it runs cleanly on any OS console. Exit non-zero on any problem.
"""
from __future__ import annotations

import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

SPEC = {
    "military_drones.csv": {"key": "id", "url": "source_url",
                            "required": {"id", "name", "manufacturer", "country", "role", "source_url"}},
    "commercial_platforms.csv": {"key": "id", "url": "docs_url",
                                 "required": {"id", "name", "vendor", "category", "license", "docs_url"}},
    "nvidia_hf_models.csv": {"key": "repo_id", "url": "url",
                             "required": {"repo_id", "description", "modality", "license", "url"}},
}


def check(fname: str, spec: dict) -> int:
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"[FAIL] {fname}: missing")
        return 1
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"[FAIL] {fname}: no rows")
        return 1
    bad = 0
    cols = set(rows[0].keys())
    missing = spec["required"] - cols
    if missing:
        print(f"[FAIL] {fname}: missing columns {sorted(missing)}")
        bad += 1
    keys = set()
    for i, row in enumerate(rows, 2):  # row 1 is the header
        k = (row.get(spec["key"]) or "").strip()
        if not k:
            print(f"[FAIL] {fname}:{i}: empty {spec['key']}")
            bad += 1
        elif k in keys:
            print(f"[FAIL] {fname}:{i}: duplicate {spec['key']} {k!r}")
            bad += 1
        keys.add(k)
        url = (row.get(spec["url"]) or "").strip()
        if not url.startswith(("http://", "https://")):
            print(f"[FAIL] {fname}:{i}: {spec['url']} is not a URL: {url!r}")
            bad += 1
    print(f"[{'OK' if not bad else 'FAIL'}] {fname}: {len(rows)} rows, {bad} problem(s)")
    return bad


def main() -> int:
    total = sum(check(f, s) for f, s in SPEC.items())
    print(f"\n[{'OK' if not total else 'FAIL'}] {len(SPEC)} files checked, {total} problem(s).")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
