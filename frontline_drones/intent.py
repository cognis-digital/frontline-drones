"""Behavioral intent classification from track kinematics.

2026 counter-UAS fusion frameworks add *behavioral intent* on top of detection:
having a track is not enough — is it transiting past, loitering, orbiting a site
for reconnaissance, or closing on a protected asset? This module derives kinematic
features from a track's position history relative to a defended point and
classifies the behavior, with a confidence and human-readable reasoning.

Scope: passive awareness / classification decision support only. It consumes an
already-formed track (positions in local meters) and outputs a label; it carries
no engagement, cueing, or defeat content. Pure stdlib, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]


class IntentClass(str, Enum):
    TRANSIT = "transit"            # straight, fast, not closing on the asset
    LOITER = "loiter"             # low net travel, wandering
    RECON_ORBIT = "recon_orbit"   # steady one-directional turn at ~constant range
    APPROACH = "approach"         # closing on the protected asset
    UNKNOWN = "unknown"


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _linfit_slope(ys: Sequence[float]) -> float:
    """Least-squares slope of ys vs index (per-step trend)."""
    n = len(ys)
    if n < 2:
        return 0.0
    xm = (n - 1) / 2.0
    ym = sum(ys) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(ys))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


@dataclass(frozen=True)
class TrackFeatures:
    mean_speed: float
    straightness: float          # net displacement / path length, 0..1
    closure_rate: float          # meters/step toward asset (positive == closing)
    turn_consistency: float      # -1..1 signed fraction of same-direction turns
    range_cv: float              # coeff. of variation of range-to-asset
    mean_range: float

    def to_dict(self) -> dict:
        return {
            "mean_speed": round(self.mean_speed, 4),
            "straightness": round(self.straightness, 4),
            "closure_rate": round(self.closure_rate, 4),
            "turn_consistency": round(self.turn_consistency, 4),
            "range_cv": round(self.range_cv, 4),
            "mean_range": round(self.mean_range, 4),
        }


def track_features(track: Sequence[Point], asset: Point, dt_s: float = 1.0
                   ) -> TrackFeatures:
    if len(track) < 2:
        raise ValueError("track needs at least 2 points")
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    steps = [(_dist(track[i], track[i + 1])) for i in range(len(track) - 1)]
    path_len = sum(steps)
    net = _dist(track[0], track[-1])
    straightness = (net / path_len) if path_len > 0 else 0.0
    mean_speed = (path_len / (len(steps) * dt_s)) if steps else 0.0

    ranges = [_dist(p, asset) for p in track]
    mean_range = sum(ranges) / len(ranges)
    # Closure: positive when range shrinks over time.
    closure_rate = -_linfit_slope(ranges)
    var = sum((r - mean_range) ** 2 for r in ranges) / len(ranges)
    range_cv = (math.sqrt(var) / mean_range) if mean_range > 0 else 0.0

    # Turn consistency: sign agreement of successive heading changes.
    headings = [math.atan2(track[i + 1][1] - track[i][1], track[i + 1][0] - track[i][0])
                for i in range(len(track) - 1)]
    turns = []
    for i in range(len(headings) - 1):
        d = (headings[i + 1] - headings[i] + math.pi) % (2 * math.pi) - math.pi
        turns.append(d)
    if turns:
        pos = sum(1 for t in turns if t > 1e-6)
        neg = sum(1 for t in turns if t < -1e-6)
        turn_consistency = (pos - neg) / len(turns)
    else:
        turn_consistency = 0.0

    return TrackFeatures(mean_speed, straightness, closure_rate,
                         turn_consistency, range_cv, mean_range)


@dataclass(frozen=True)
class IntentThresholds:
    approach_closure_frac: float = 0.15   # closing >15% of mean range per step
    transit_straightness: float = 0.8
    loiter_straightness: float = 0.35
    orbit_turn_consistency: float = 0.6
    orbit_range_cv: float = 0.15          # near-constant range


@dataclass(frozen=True)
class IntentAssessment:
    intent: IntentClass
    confidence: float
    reasoning: str
    features: TrackFeatures

    def to_dict(self) -> dict:
        return {"intent": self.intent.value, "confidence": round(self.confidence, 4),
                "reasoning": self.reasoning, "features": self.features.to_dict()}


def classify_intent(track: Sequence[Point], asset: Point, dt_s: float = 1.0,
                    thresholds: Optional[IntentThresholds] = None) -> IntentAssessment:
    """Classify a track's behavior relative to a protected asset."""
    th = thresholds or IntentThresholds()
    f = track_features(track, asset, dt_s)

    # Closure normalized to mean range (per step).
    norm_closure = (f.closure_rate / f.mean_range) if f.mean_range > 0 else 0.0

    if norm_closure >= th.approach_closure_frac and f.straightness >= th.loiter_straightness:
        conf = min(1.0, norm_closure / th.approach_closure_frac * 0.6 + 0.4)
        return IntentAssessment(IntentClass.APPROACH, round(conf, 4),
                                "closing on the protected asset", f)
    if (abs(f.turn_consistency) >= th.orbit_turn_consistency
            and f.range_cv <= th.orbit_range_cv):
        return IntentAssessment(IntentClass.RECON_ORBIT, 0.7,
                                "steady one-directional turn at near-constant range", f)
    if f.straightness <= th.loiter_straightness and abs(norm_closure) < th.approach_closure_frac:
        return IntentAssessment(IntentClass.LOITER, 0.65,
                                "low net travel, wandering near the area", f)
    if f.straightness >= th.transit_straightness:
        return IntentAssessment(IntentClass.TRANSIT, 0.7,
                                "straight, steady track not closing on the asset", f)
    return IntentAssessment(IntentClass.UNKNOWN, 0.3, "ambiguous kinematics", f)


def coordinated_ingress(tracks: Sequence[Sequence[Point]], asset: Point,
                        dt_s: float = 1.0, min_closing: int = 3) -> bool:
    """True if several tracks are simultaneously closing on the asset (swarm ingress)."""
    closing = 0
    for tr in tracks:
        if len(tr) < 2:
            continue
        a = classify_intent(tr, asset, dt_s)
        if a.intent is IntentClass.APPROACH:
            closing += 1
    return closing >= min_closing
