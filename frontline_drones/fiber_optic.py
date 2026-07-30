"""Fiber-optic (RF-silent) drone detection support.

Fiber-optic drones trail a hair-thin optical cable instead of a radio link, so
they emit **no control-link RF and no Remote ID** — the reason RF/Remote-ID
sensors miss them and jamming does not affect them (NATO's 2025 counter-UAS
challenge found no single defeat mechanism). The detectable tell is a track that
is **physically present** on the airframe-sensing modalities (acoustic rotor tone,
EO/IR optics, radar micro-Doppler) yet **radio-silent** on the link modalities.

This module fuses multi-modal detections into a fiber-optic likelihood, checks a
track's range history against a tether-payout geometry, and recommends a sensor
mix for a given electronic-warfare environment.

Scope: passive detection/classification decision support only. It ingests already-
measured detection confidences and range features; it neither records, transmits,
nor emits anything, and carries no engagement or defeat content. Pure stdlib,
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence


class Modality(str, Enum):
    ACOUSTIC = "acoustic"
    OPTICAL = "optical"        # EO / visual
    THERMAL = "thermal"        # IR
    RADAR = "radar"            # micro-Doppler
    RF = "rf"                  # control-link emission
    REMOTE_ID = "remote_id"    # broadcast Remote ID


# Modalities that sense the airframe itself vs. a radio emission.
PHYSICAL_MODALITIES = frozenset(
    {Modality.ACOUSTIC, Modality.OPTICAL, Modality.THERMAL, Modality.RADAR}
)
LINK_MODALITIES = frozenset({Modality.RF, Modality.REMOTE_ID})


@dataclass(frozen=True)
class SensorHit:
    """A per-modality detection confidence for a single track."""

    modality: Modality
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def is_physical(self) -> bool:
        return self.modality in PHYSICAL_MODALITIES

    @property
    def is_link(self) -> bool:
        return self.modality in LINK_MODALITIES


def _noisy_or(confidences: Sequence[float]) -> float:
    """Combine independent detection confidences: 1 - prod(1 - c)."""
    prod = 1.0
    for c in confidences:
        prod *= (1.0 - c)
    return round(1.0 - prod, 6)


@dataclass(frozen=True)
class FiberOpticAssessment:
    likelihood: float          # 0..1 that this is a fiber-optic (RF-silent) drone
    physical_evidence: float   # fused confidence the airframe is present
    link_evidence: float       # fused confidence a radio link/Remote ID is present
    rf_silent: bool
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "likelihood": self.likelihood,
            "physical_evidence": self.physical_evidence,
            "link_evidence": self.link_evidence,
            "rf_silent": self.rf_silent,
            "reasoning": self.reasoning,
        }


def assess_fiber_optic(hits: Sequence[SensorHit],
                       rf_silent_threshold: float = 0.2) -> FiberOpticAssessment:
    """Fuse multi-modal hits into a fiber-optic likelihood.

    A fiber-optic drone shows strong physical evidence with near-zero link
    evidence, so likelihood = physical_evidence * (1 - link_evidence). A normal
    radio drone (strong RF/Remote ID) is driven toward zero.
    """
    physical = _noisy_or([h.confidence for h in hits if h.is_physical])
    link = _noisy_or([h.confidence for h in hits if h.is_link])
    likelihood = round(physical * (1.0 - link), 6)
    rf_silent = link <= rf_silent_threshold
    if physical == 0.0:
        reason = "no airframe-sensing detection; not assessable as fiber-optic"
    elif not rf_silent:
        reason = "radio link/Remote ID present; consistent with a conventional RF drone"
    else:
        reason = "airframe present but radio-silent; consistent with a fiber-optic drone"
    return FiberOpticAssessment(likelihood, physical, link, rf_silent, reason)


@dataclass(frozen=True)
class TetherGeometry:
    """Consistency of a track's range history with a fiber tether payout."""

    within_tether: bool
    payout_monotonicity: float   # 0..1 fraction of non-retracting range steps
    max_range_m: float
    consistent: bool

    def to_dict(self) -> dict:
        return {
            "within_tether": self.within_tether,
            "payout_monotonicity": self.payout_monotonicity,
            "max_range_m": self.max_range_m,
            "consistent": self.consistent,
        }


def assess_tether(ranges_m: Sequence[float], max_tether_m: float = 20_000.0,
                  monotonicity_floor: float = 0.6) -> TetherGeometry:
    """Check slant-range history against a spooled-fiber flyout.

    A tethered drone's range from its launch point cannot exceed the spool length,
    and the fiber pays out (range mostly non-decreasing) rather than retracting.
    """
    if max_tether_m <= 0:
        raise ValueError("max_tether_m must be > 0")
    if any(r < 0 for r in ranges_m):
        raise ValueError("ranges must be >= 0")
    max_range = max(ranges_m) if ranges_m else 0.0
    within = max_range <= max_tether_m
    steps = list(zip(ranges_m, list(ranges_m)[1:]))
    if steps:
        non_retract = sum(1 for a, b in steps if b >= a - 1e-9)
        monotonicity = round(non_retract / len(steps), 6)
    else:
        monotonicity = 1.0
    consistent = within and monotonicity >= monotonicity_floor
    return TetherGeometry(within, monotonicity, round(max_range, 3), consistent)


def recommend_sensors(jamming_level: float) -> List[Modality]:
    """Recommend a detection sensor mix for the current EW environment.

    Under heavy jamming, RF/Remote-ID detection of conventional drones degrades and
    fiber-optic drones are radio-silent regardless — so airframe-sensing modalities
    are prioritized. In a benign RF environment, link modalities are cheap wins and
    lead the mix.
    """
    if not 0.0 <= jamming_level <= 1.0:
        raise ValueError("jamming_level must be in [0, 1]")
    physical = [Modality.ACOUSTIC, Modality.OPTICAL, Modality.RADAR, Modality.THERMAL]
    if jamming_level >= 0.5:
        return physical
    return [Modality.RF, Modality.REMOTE_ID] + physical
