"""Tests for the dynamic dataset-growth engine and harvest frontier."""
import pytest

from frontline_drones.frontier import GEO_TILES, build_frontier, source_pairs
from frontline_drones.growth import (
    Endpoint,
    GrowthEngine,
    HarvestStore,
    Record,
    SourceStats,
    expand,
    synthetic_fetcher,
)


def test_expand_base_plus_terms():
    eps = expand([("S", "http://x")], ["a", "b"], param_name="q")
    assert {"S:_all", "S:q=a", "S:q=b"} <= {e.key for e in eps}


def test_store_merge_dedup():
    store = HarvestStore()
    r1 = Record("u1", "S", "e", 0.0)
    assert store.merge([r1, Record("u2", "S", "e", 0.0)]) == 2
    assert store.merge([r1]) == 0


def test_reliability_penalizes_errors():
    good = SourceStats(fetches=10, records=100, new_records=100, errors=0)
    bad = SourceStats(fetches=10, records=100, new_records=100, errors=8)
    assert good.reliability > bad.reliability


def _engine():
    eng = GrowthEngine(fetcher=synthetic_fetcher(3))
    eng.add_endpoints(expand([("S", "http://x")], [str(i) for i in range(10)]))
    return eng


def test_engine_grows_over_cycles():
    sizes = [r.total_size for r in _engine().grow(cycles=3, start=0.0)]
    assert sizes[0] < sizes[1] < sizes[2]


def test_engine_dedup_same_timestamp():
    eng = _engine()
    a = eng.run_cycle(now=5.0)
    assert a.new_records > 0 and eng.run_cycle(now=5.0).new_records == 0


def test_engine_survives_fetcher_errors():
    def boom(endpoint, now):
        raise RuntimeError("down")
    eng = GrowthEngine(fetcher=boom)
    eng.add_endpoints([Endpoint("S", "s:0", "http://x")])
    assert eng.run_cycle(now=0.0).new_records == 0


def test_tile_grid_size():
    assert len(GEO_TILES) == 2592


def test_frontier_reaches_thousands():
    endpoints = build_frontier()
    assert len(endpoints) >= 2000
    assert len({e.key for e in endpoints}) == len(endpoints)


def test_source_pairs_from_csvs():
    pairs = source_pairs()
    assert pairs and all(u.startswith("http") for (n, u) in pairs)


def test_frontier_feeds_engine():
    eng = GrowthEngine(fetcher=synthetic_fetcher(1))
    eng.add_endpoints(build_frontier())
    assert eng.run_cycle(now=0.0, max_endpoints=40).new_records == 40
