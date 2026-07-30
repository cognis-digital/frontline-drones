"""Multi-sensor fusion track-confidence scorer (situational-awareness only).

Credible counter-UAS detection fuses **at least two** sensor modalities because
each has a disqualifying blind spot (RF is blind to autonomous/fiber-optic
drones, radar struggles with small-RCS targets in clutter, acoustic is limited
to ~300-500 m and degraded by wind, EO/IR is limited by lighting/weather/FoV).
This module takes per-modality detection inputs plus a track-quality metric and
returns a normalized fused confidence and a false-alarm-aware severity tier.

Scope: this produces a *situational-awareness score only*. It contains no
tasking, cueing-to-effector, guidance or engagement logic. Everything here is a
deterministic pure function of its inputs, so it is fully unit-testable and
stdlib-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical detection modalities and their per-sensor reliability weight. Weights
# reflect how much independent trust a *single* firing sensor of that class earns
# in the fused estimate. Remote ID is deliberately the least-trusted corroborator
# because it is a spoofable identification aid, not authenticated truth.
MODALITY_WEIGHTS: dict[str, float] = {
    "rf": 0.85,
    "radar": 0.90,
    "acoustic": 0.60,
    "eo_ir": 0.80,
    "remote_id": 0.70,
}

MODALITIES: tuple[str, ...] = tuple(MODALITY_WEIGHTS)

# A modality "fires" when its normalized confidence reaches this threshold.
FIRE_THRESHOLD = 0.5

# Severity tiers, ordered weakest -> strongest.
SEVERITY_TIERS: tuple[str, ...] = ("none", "low", "medium", "high", "critical")


@dataclass(frozen=True)
class FusionResult:
    """Outcome of fusing per-modality detections for one track.

    Attributes:
        fused_confidence: Normalized [0, 1] combined confidence.
        modalities_fired: Sorted names of modalities that reached the fire
            threshold and corroborated the track.
        num_modalities: Count of ``modalities_fired``.
        meets_fusion_baseline: True when >= 2 independent modalities fired,
            the industry baseline for a *credible* detection.
        track_quality: The [0, 1] track-quality metric passed in.
        severity: One of :data:`SEVERITY_TIERS`, false-alarm aware.
        rationale: Human-readable explanation of the score and severity.
    """

    fused_confidence: float
    modalities_fired: tuple[str, ...]
    num_modalities: int
    meets_fusion_baseline: bool
    track_quality: float
    severity: str
    rationale: str = ""
    per_modality: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the result."""
        return {
            "fused_confidence": round(self.fused_confidence, 4),
            "modalities_fired": list(self.modalities_fired),
            "num_modalities": self.num_modalities,
            "meets_fusion_baseline": self.meets_fusion_baseline,
            "track_quality": round(self.track_quality, 4),
            "severity": self.severity,
            "rationale": self.rationale,
            "per_modality": {k: round(v, 4) for k, v in self.per_modality.items()},
        }


def _normalize(value: bool | float | int | None) -> float | None:
    """Coerce a modality input to a [0, 1] confidence, or ``None`` if absent.

    ``None`` means the sensor is not present/reporting (excluded from fusion).
    ``bool`` maps to 0.0/1.0. Numbers are clamped into [0, 1].
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    v = float(value)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _clamp_track_quality(track_quality: float) -> float:
    tq = float(track_quality)
    if tq < 0.0:
        return 0.0
    if tq > 1.0:
        return 1.0
    return tq


def _severity(
    fused: float,
    fired: tuple[str, ...],
    meets_baseline: bool,
) -> str:
    """Map a fused confidence to a false-alarm-aware severity tier.

    False-alarm awareness: a single-sensor detection is capped so it can never
    reach the top tiers no matter how confident the lone sensor is, and a
    Remote-ID-only detection is capped hardest because Remote ID is spoofable.
    Only a fused (>= 2 modality) track can escalate to ``high``/``critical``.
    """
    n = len(fired)
    if n == 0 or fused < 0.2:
        return "none"

    if n == 1:
        # Single sensor: cap at "medium"; Remote-ID-only caps at "low".
        if fired == ("remote_id",):
            return "low"
        return "medium" if fused >= 0.5 else "low"

    # Fused, multi-sensor track.
    if meets_baseline and fused >= 0.85:
        return "critical"
    if fused >= 0.7:
        return "high"
    if fused >= 0.45:
        return "medium"
    return "low"


def fuse(
    *,
    rf: bool | float | None = None,
    radar: bool | float | None = None,
    acoustic: bool | float | None = None,
    eo_ir: bool | float | None = None,
    remote_id: bool | float | None = None,
    track_quality: float = 1.0,
) -> FusionResult:
    """Fuse per-modality detections into one situational-awareness score.

    Each keyword is a modality input: ``None`` (sensor absent), a ``bool``, or a
    confidence in [0, 1]. ``track_quality`` in [0, 1] discounts the fused score
    for poor kinematic track quality.

    The combination is a weighted noisy-OR over the modalities that *fired*
    (reached :data:`FIRE_THRESHOLD`): ``1 - prod(1 - w_i * c_i)``. This rewards
    independent corroboration (two mediocre sensors beat one good one) while
    staying bounded in [0, 1]. The result is then scaled by a track-quality
    factor so a weak track cannot present as a high-confidence detection.

    Returns:
        A :class:`FusionResult`. Deterministic for identical inputs.
    """
    inputs = {
        "rf": rf,
        "radar": radar,
        "acoustic": acoustic,
        "eo_ir": eo_ir,
        "remote_id": remote_id,
    }
    tq = _clamp_track_quality(track_quality)

    per_modality: dict[str, float] = {}
    fired: list[str] = []
    product = 1.0
    for name in MODALITIES:
        conf = _normalize(inputs[name])
        if conf is None:
            continue
        per_modality[name] = conf
        if conf >= FIRE_THRESHOLD:
            fired.append(name)
            product *= 1.0 - MODALITY_WEIGHTS[name] * conf

    fired_t = tuple(sorted(fired))
    raw = 0.0 if not fired_t else 1.0 - product
    # Track quality discounts confidence but never below half its own weight, so
    # a perfect-quality track passes through unchanged and a zero-quality track
    # is halved rather than zeroed (the sensors still fired).
    fused = raw * (0.5 + 0.5 * tq)

    meets_baseline = len(fired_t) >= 2
    severity = _severity(fused, fired_t, meets_baseline)

    if not fired_t:
        rationale = "No modality reached the detection threshold."
    else:
        base = (
            f"{len(fired_t)} modality/modalities fired ({', '.join(fired_t)}); "
            f"fused confidence {fused:.2f}, track quality {tq:.2f}."
        )
        if meets_baseline:
            base += " Meets the >=2-modality fusion baseline for a credible detection."
        else:
            base += (
                " Single-sensor detection: below the >=2-modality fusion baseline; "
                "severity capped to limit false alarms."
            )
        rationale = base

    return FusionResult(
        fused_confidence=fused,
        modalities_fired=fired_t,
        num_modalities=len(fired_t),
        meets_fusion_baseline=meets_baseline,
        track_quality=tq,
        severity=severity,
        rationale=rationale,
        per_modality=per_modality,
    )


def severity_rank(severity: str) -> int:
    """Return the ordinal rank of a severity tier (higher = more severe)."""
    try:
        return SEVERITY_TIERS.index(severity)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown severity {severity!r}") from exc
