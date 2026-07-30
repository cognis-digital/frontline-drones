"""Dynamic dataset-growth engine.

A curated source list is static; real intelligence value comes from the *records*
those sources emit, accumulated and refined over time. This engine turns a source
registry into a living dataset:

* **Expand** — fan each parametric source out across a real vocabulary (regions,
  actors, stations, codes...) into many concrete, addressable endpoints. A few
  hundred sources times a few hundred parameters is tens of thousands of endpoints.
* **Harvest** — pull records from endpoints through an injected fetcher (offline
  and deterministic by default; a live fetcher is dropped in for real deployment).
* **Accumulate** — merge records into a store that dedupes by stable uid, tracks
  provenance and fetch time, and never loses history. Repeated cycles grow it.
* **Discover** — records can yield *new* endpoints, so the frontier expands each
  cycle: growth compounds instead of plateauing.
* **Improve** — per-source reliability (yield, freshness, error rate) is scored so
  the engine prioritizes productive sources and prunes dead ones.

Pure stdlib, deterministic (timestamps and the fetcher are injected — no wall clock
in the core), and offline-safe. Awareness/analysis only; no targeting content.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Endpoint:
    """A concrete, addressable data endpoint derived from a source + parameters."""

    source: str            # parent source name
    key: str               # unique endpoint id, e.g. "acled:region=Sahel"
    url: str
    params: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"source": self.source, "key": self.key, "url": self.url,
                "params": dict(self.params)}


@dataclass(frozen=True)
class Record:
    """One harvested datum with provenance."""

    uid: str               # stable dedup key
    source: str
    endpoint: str
    fetched_at: float      # epoch seconds (injected)
    payload: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"uid": self.uid, "source": self.source, "endpoint": self.endpoint,
                "fetched_at": round(self.fetched_at, 3), "payload": dict(self.payload)}


def expand(sources: Sequence[Tuple[str, str]], vocabulary: Sequence[str],
           param_name: str = "q", base_only: bool = True) -> List[Endpoint]:
    """Fan parametric ``sources`` [(name, url), ...] across a ``vocabulary``.

    Produces one endpoint per (source, term). With ``base_only`` a bare endpoint
    for each source (no parameter) is also emitted, so both the whole-feed pull and
    the per-term pulls are addressable. Endpoint keys are deterministic and unique.
    """
    out: List[Endpoint] = []
    seen = set()
    for name, url in sources:
        if base_only:
            k = f"{name}:_all"
            if k not in seen:
                seen.add(k)
                out.append(Endpoint(name, k, url, {}))
        for term in vocabulary:
            k = f"{name}:{param_name}={term}"
            if k in seen:
                continue
            seen.add(k)
            sep = "&" if "?" in url else "?"
            out.append(Endpoint(name, k, f"{url}{sep}{param_name}={term}",
                                {param_name: term}))
    return out


@dataclass
class SourceStats:
    """Self-improvement bookkeeping for one source."""

    fetches: int = 0
    records: int = 0
    new_records: int = 0
    errors: int = 0
    last_fetch_at: float = 0.0

    @property
    def yield_per_fetch(self) -> float:
        return self.records / self.fetches if self.fetches else 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.fetches if self.fetches else 0.0

    @property
    def reliability(self) -> float:
        """0..1 score: rewards new-record yield, penalizes errors."""
        if not self.fetches:
            return 0.0
        novelty = self.new_records / (self.records + 1.0)
        base = min(1.0, self.yield_per_fetch / 10.0)
        return round(max(0.0, base * (1.0 - self.error_rate) * (0.5 + 0.5 * novelty)), 6)

    def to_dict(self) -> dict:
        return {"fetches": self.fetches, "records": self.records,
                "new_records": self.new_records, "errors": self.errors,
                "yield_per_fetch": round(self.yield_per_fetch, 3),
                "error_rate": round(self.error_rate, 3),
                "reliability": self.reliability}


class HarvestStore:
    """Accumulating, deduplicating record store with provenance."""

    def __init__(self) -> None:
        self._records: Dict[str, Record] = {}
        self._sources: Dict[str, int] = {}

    def merge(self, records: Iterable[Record]) -> int:
        """Add records, ignoring uids already present. Returns count newly added."""
        added = 0
        for r in records:
            if r.uid in self._records:
                continue
            self._records[r.uid] = r
            self._sources[r.source] = self._sources.get(r.source, 0) + 1
            added += 1
        return added

    @property
    def size(self) -> int:
        return len(self._records)

    def sources(self) -> Dict[str, int]:
        return dict(self._sources)

    def records(self) -> List[Record]:
        return list(self._records.values())

    def freshness(self, now: float) -> float:
        """Mean age (seconds) of stored records relative to ``now``."""
        if not self._records:
            return 0.0
        return sum(now - r.fetched_at for r in self._records.values()) / len(self._records)

    def to_dict(self, now: Optional[float] = None) -> dict:
        d = {"size": self.size, "sources": self.sources()}
        if now is not None:
            d["mean_age_s"] = round(self.freshness(now), 3)
        return d


# A fetcher maps (endpoint, now) -> list[Record]. The default is offline/deterministic.
Fetcher = Callable[[Endpoint, float], List[Record]]
# A discovery function maps records -> new endpoints (frontier expansion).
Discoverer = Callable[[Sequence[Record]], List[Endpoint]]


def synthetic_fetcher(records_per_endpoint: int = 3) -> Fetcher:
    """A deterministic offline fetcher for testing / dry-runs.

    Emits ``records_per_endpoint`` records whose uids are stable functions of the
    endpoint key and cycle, so re-fetching the same endpoint at the same time
    yields the same records (dedup is exercised), while advancing time yields new
    ones (growth is exercised).
    """
    def _fetch(endpoint: Endpoint, now: float) -> List[Record]:
        cycle = int(now)
        out = []
        for i in range(records_per_endpoint):
            uid = f"{endpoint.key}#{cycle}:{i}"
            out.append(Record(uid, endpoint.source, endpoint.key, now,
                              {"seq": i, "cycle": cycle}))
        return out
    return _fetch


@dataclass
class GrowthReport:
    cycle: int
    fetched_endpoints: int
    records_seen: int
    new_records: int
    total_size: int
    frontier: int          # endpoints known after this cycle (may have grown)

    @property
    def growth_rate(self) -> float:
        prev = self.total_size - self.new_records
        return self.new_records / prev if prev > 0 else float(self.new_records > 0)

    def to_dict(self) -> dict:
        return {"cycle": self.cycle, "fetched_endpoints": self.fetched_endpoints,
                "records_seen": self.records_seen, "new_records": self.new_records,
                "total_size": self.total_size, "frontier": self.frontier,
                "growth_rate": round(self.growth_rate, 4)}


class GrowthEngine:
    """Runs harvest cycles that accumulate and grow a dataset."""

    def __init__(self, store: Optional[HarvestStore] = None,
                 fetcher: Optional[Fetcher] = None,
                 discoverer: Optional[Discoverer] = None) -> None:
        self.store = store or HarvestStore()
        self.fetcher = fetcher or synthetic_fetcher()
        self.discoverer = discoverer
        self.stats: Dict[str, SourceStats] = {}
        self._frontier: Dict[str, Endpoint] = {}
        self._cycle = 0

    def add_endpoints(self, endpoints: Iterable[Endpoint]) -> int:
        """Register endpoints on the frontier. Returns count newly added."""
        added = 0
        for e in endpoints:
            if e.key not in self._frontier:
                self._frontier[e.key] = e
                added += 1
        return added

    @property
    def frontier_size(self) -> int:
        return len(self._frontier)

    def prioritized_endpoints(self) -> List[Endpoint]:
        """Frontier ordered by source reliability (productive sources first)."""
        def score(e: Endpoint) -> float:
            st = self.stats.get(e.source)
            return st.reliability if st else 1.0   # unknown sources tried first
        return sorted(self._frontier.values(), key=score, reverse=True)

    def run_cycle(self, now: float, max_endpoints: Optional[int] = None) -> GrowthReport:
        """Fetch (a prioritized slice of) the frontier once and accumulate."""
        self._cycle += 1
        endpoints = self.prioritized_endpoints()
        if max_endpoints is not None:
            endpoints = endpoints[:max_endpoints]
        seen = 0
        new_total = 0
        harvested: List[Record] = []
        for e in endpoints:
            st = self.stats.setdefault(e.source, SourceStats())
            try:
                recs = self.fetcher(e, now)
            except Exception:
                st.fetches += 1
                st.errors += 1
                continue
            st.fetches += 1
            st.last_fetch_at = now
            st.records += len(recs)
            seen += len(recs)
            before = self.store.size
            added = self.store.merge(recs)
            st.new_records += added
            new_total += added
            harvested.extend(recs)
            _ = before
        # Frontier expansion: discovered endpoints compound growth.
        if self.discoverer and harvested:
            self.add_endpoints(self.discoverer(harvested))
        return GrowthReport(self._cycle, len(endpoints), seen, new_total,
                            self.store.size, self.frontier_size)

    def grow(self, cycles: int, start: float = 0.0, dt: float = 1.0,
             max_endpoints: Optional[int] = None) -> List[GrowthReport]:
        """Run several cycles at successive timestamps; returns per-cycle reports."""
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        reports = []
        for c in range(cycles):
            reports.append(self.run_cycle(start + c * dt, max_endpoints))
        return reports

    def summary(self, now: Optional[float] = None) -> dict:
        return {"cycles": self._cycle, "frontier": self.frontier_size,
                "store": self.store.to_dict(now),
                "top_sources": sorted(
                    ((name, st.reliability) for name, st in self.stats.items()),
                    key=lambda t: t[1], reverse=True)[:10]}
