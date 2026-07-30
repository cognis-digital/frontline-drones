"""Multi-target data association and track management (awareness only).

A counter-UAS sensor delivers a fresh set of point *detections* every scan; to
build situational awareness those detections must be stitched into persistent
*tracks*. This module is a small, deterministic, stdlib-only track manager that
does the classic radar/EO bookkeeping:

* **gating** - only associate a detection to a track if it falls inside a
  velocity-predicted gate radius;
* **assignment** - greedy global-nearest-neighbour matching of detections to
  tracks each scan;
* **track life-cycle** - spawn *tentative* tracks for unassigned detections,
  promote to *confirmed* after an M-of-N hit history, *coast* tracks that miss
  updates, and *delete* them after too many consecutive misses.

Scope: pure association and bookkeeping for awareness. The constant-velocity
prediction exists only to place the association gate and estimate track
velocity; there is no intercept, guidance or engagement logic anywhere.
Coordinates are a local ENU plane in metres; time is seconds.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

# Track life-cycle states, weakest -> strongest -> terminal.
TRACK_STATES: tuple[str, ...] = ("tentative", "confirmed", "coasting", "deleted")


@dataclass(frozen=True)
class Detection:
    """One point detection from a single scan.

    Attributes:
        east / north: Position in the local ENU plane, metres.
        t: Scan timestamp, seconds.
        meta: Optional opaque metadata carried through to the associated track.
    """

    east: float
    north: float
    t: float = 0.0
    meta: dict = field(default_factory=dict)


def _as_detection(item, t: float) -> Detection:
    """Coerce a caller-supplied detection into a :class:`Detection`."""
    if isinstance(item, Detection):
        return item if item.t else Detection(item.east, item.north, t, item.meta)
    if isinstance(item, dict):
        return Detection(
            east=float(item.get("east", item.get("x", 0.0))),
            north=float(item.get("north", item.get("y", 0.0))),
            t=float(item.get("t", t)),
            meta=dict(item.get("meta", {})),
        )
    seq = list(item)
    east, north = float(seq[0]), float(seq[1])
    return Detection(east=east, north=north, t=t)


@dataclass
class Track:
    """A persistent track built from associated detections (mutable bookkeeping).

    Attributes:
        track_id: Stable integer id assigned at spawn.
        east / north: Latest associated position, ENU metres.
        vel_east / vel_north: Estimated ENU velocity, m/s (0 until 2+ fixes).
        last_update_t: Timestamp of the most recent association.
        hits: Total number of associated detections.
        misses: Consecutive scans missed since the last association.
        age: Number of scans this track has existed for.
        state: One of :data:`TRACK_STATES`.
        history: Recent (t, east, north) fixes, newest last.
    """

    track_id: int
    east: float
    north: float
    vel_east: float = 0.0
    vel_north: float = 0.0
    last_update_t: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    state: str = "tentative"
    history: list[tuple[float, float, float]] = field(default_factory=list)

    def predict(self, t: float) -> tuple[float, float]:
        """Constant-velocity predicted (east, north) at time ``t`` (gate centre)."""
        dt = t - self.last_update_t
        return (self.east + self.vel_east * dt, self.north + self.vel_north * dt)

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vel_east, self.vel_north)

    @property
    def heading_deg(self) -> float:
        """Compass heading of the velocity vector (0 = N), or 0 when stationary."""
        if self.vel_east == 0.0 and self.vel_north == 0.0:
            return 0.0
        return (math.degrees(math.atan2(self.vel_east, self.vel_north)) + 360.0) % 360.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "east": round(self.east, 3),
            "north": round(self.north, 3),
            "vel_east": round(self.vel_east, 3),
            "vel_north": round(self.vel_north, 3),
            "speed_mps": round(self.speed_mps, 3),
            "heading_deg": round(self.heading_deg, 3),
            "last_update_t": round(self.last_update_t, 3),
            "hits": self.hits,
            "misses": self.misses,
            "age": self.age,
            "state": self.state,
        }


class TrackManager:
    """Greedy nearest-neighbour multi-target track manager (awareness only).

    Call :meth:`update` once per scan with that scan's detections. The manager
    gates each detection against every live track's velocity-predicted position,
    solves a greedy global-nearest-neighbour assignment, updates matched tracks,
    spawns tentative tracks for the leftovers, and ages/coasts/deletes tracks that
    were not updated. Everything is deterministic given the same input sequence.

    Args:
        gate_radius_m: Maximum association distance from a track's predicted
            position, metres.
        confirm_hits: Hits required (within ``confirm_window`` scans) to promote a
            tentative track to ``confirmed`` (the M of an M-of-N rule).
        confirm_window: The N of the M-of-N confirmation rule (scan window).
        max_coast_misses: Consecutive missed scans before a track is deleted.
        history_len: How many recent fixes to retain per track.
    """

    def __init__(
        self,
        *,
        gate_radius_m: float = 50.0,
        confirm_hits: int = 3,
        confirm_window: int = 5,
        max_coast_misses: int = 3,
        history_len: int = 16,
    ) -> None:
        if gate_radius_m <= 0:
            raise ValueError("gate_radius_m must be positive")
        if confirm_hits < 1:
            raise ValueError("confirm_hits must be >= 1")
        if confirm_window < confirm_hits:
            raise ValueError("confirm_window must be >= confirm_hits")
        if max_coast_misses < 1:
            raise ValueError("max_coast_misses must be >= 1")
        self.gate_radius_m = float(gate_radius_m)
        self.confirm_hits = int(confirm_hits)
        self.confirm_window = int(confirm_window)
        self.max_coast_misses = int(max_coast_misses)
        self.history_len = int(history_len)
        self.tracks: list[Track] = []
        self._id_counter = itertools.count(1)

    # -- internal helpers ----------------------------------------------------

    def _spawn(self, det: Detection) -> Track:
        tid = next(self._id_counter)
        track = Track(
            track_id=tid,
            east=det.east,
            north=det.north,
            last_update_t=det.t,
            history=[(det.t, det.east, det.north)],
        )
        self.tracks.append(track)
        return track

    def _update_track(self, track: Track, det: Detection) -> None:
        dt = det.t - track.last_update_t
        if dt > 0:
            track.vel_east = (det.east - track.east) / dt
            track.vel_north = (det.north - track.north) / dt
        track.east = det.east
        track.north = det.north
        track.last_update_t = det.t
        track.hits += 1
        track.misses = 0
        track.history.append((det.t, det.east, det.north))
        if len(track.history) > self.history_len:
            track.history = track.history[-self.history_len :]
        # M-of-N confirmation: enough hits accrued within the confirmation window.
        if track.state == "tentative" and track.hits >= self.confirm_hits:
            track.state = "confirmed"
        elif track.state == "coasting":
            track.state = "confirmed" if track.hits >= self.confirm_hits else "tentative"

    def _assign(self, t: float, dets: list[Detection]) -> dict[int, int]:
        """Greedy global-nearest-neighbour: returns {det_index: track_index}."""
        live = [i for i, tr in enumerate(self.tracks) if tr.state != "deleted"]
        pairs: list[tuple[float, int, int]] = []
        for di, det in enumerate(dets):
            for ti in live:
                px, py = self.tracks[ti].predict(t)
                dist = math.hypot(det.east - px, det.north - py)
                if dist <= self.gate_radius_m:
                    pairs.append((dist, di, ti))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))
        assigned_det: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _dist, di, ti in pairs:
            if di in assigned_det or ti in used_tracks:
                continue
            assigned_det[di] = ti
            used_tracks.add(ti)
        return assigned_det

    # -- public API ----------------------------------------------------------

    def update(self, detections, t: float | None = None) -> list[Track]:
        """Ingest one scan of detections at time ``t``; return the live tracks.

        ``detections`` is any iterable of :class:`Detection`, ``(east, north)``
        tuples, or dicts. If ``t`` is None it is taken from the detections (or the
        previous scan time). Returns all non-deleted tracks after the update.
        """
        dets = [_as_detection(d, t if t is not None else 0.0) for d in detections]
        if t is None:
            t = max((d.t for d in dets), default=self._last_time())
        # Backfill scan time onto detections lacking their own.
        dets = [d if d.t else Detection(d.east, d.north, t, d.meta) for d in dets]

        assignment = self._assign(t, dets)
        updated_tracks: set[int] = set()
        for di, det in enumerate(dets):
            ti = assignment.get(di)
            if ti is None:
                self._spawn(det)
            else:
                self._update_track(self.tracks[ti], det)
                updated_tracks.add(ti)

        # Age tracks that were not updated this scan.
        for ti, track in enumerate(self.tracks):
            if track.state == "deleted":
                continue
            track.age += 1
            if ti in updated_tracks:
                continue
            if track.last_update_t == t:
                # Freshly spawned this scan; do not immediately penalise.
                continue
            track.misses += 1
            if track.state in ("tentative", "confirmed"):
                track.state = "coasting" if track.state == "confirmed" else track.state
            if track.state == "tentative" and track.misses >= 1:
                # A tentative track that misses before confirming is dropped fast.
                if track.misses >= max(1, self.max_coast_misses - 1):
                    track.state = "deleted"
            if track.misses >= self.max_coast_misses:
                track.state = "deleted"

        return self.live_tracks()

    def _last_time(self) -> float:
        return max((tr.last_update_t for tr in self.tracks), default=0.0)

    def live_tracks(self) -> list[Track]:
        """All tracks not in the ``deleted`` state, in spawn order."""
        return [tr for tr in self.tracks if tr.state != "deleted"]

    def confirmed_tracks(self) -> list[Track]:
        """Live tracks that have reached the ``confirmed`` state."""
        return [tr for tr in self.tracks if tr.state == "confirmed"]

    def prune_deleted(self) -> int:
        """Drop deleted tracks from the internal list; return how many removed."""
        before = len(self.tracks)
        self.tracks = [tr for tr in self.tracks if tr.state != "deleted"]
        return before - len(self.tracks)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of the manager state."""
        return {
            "gate_radius_m": self.gate_radius_m,
            "confirm_hits": self.confirm_hits,
            "confirm_window": self.confirm_window,
            "max_coast_misses": self.max_coast_misses,
            "num_tracks": len(self.tracks),
            "num_live": len(self.live_tracks()),
            "num_confirmed": len(self.confirmed_tracks()),
            "tracks": [tr.to_dict() for tr in self.tracks],
        }
