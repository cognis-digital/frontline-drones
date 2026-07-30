"""Canonical detection-alert schema + serializer (situational awareness only).

Defines one normalized detection-alert record so downstream dashboards / SIEMs
receive consistent warning events regardless of which sensors fired. An alert is
a *situational-awareness notification* - it reports that something was detected,
its fused confidence, classified type and severity. It never carries a command,
tasking or countermeasure.

Pure stdlib. Pairs with :mod:`frontline_drones.fusion` (which produces the
fused confidence/severity) and :mod:`frontline_drones.export` (SAPIENT export).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .fusion import SEVERITY_TIERS, FusionResult, severity_rank

SCHEMA_VERSION = "frontline-drones.detection-alert/1"


@dataclass(frozen=True)
class DetectionAlert:
    """A canonical detection-alert (warning) event.

    Attributes:
        timestamp: ISO-8601 (or any caller-supplied) time string.
        sensors_fired: Modalities that corroborated the detection.
        fused_confidence: Normalized [0, 1] fused confidence.
        classified_type: Best-guess platform/type label (e.g. ``"consumer_dji"``)
            or ``"unknown"``.
        track_quality: [0, 1] kinematic track-quality metric.
        severity: One of :data:`fusion.SEVERITY_TIERS`.
        source_node: Identifier of the reporting sensor node/site.
        meets_fusion_baseline: True when >= 2 modalities fired.
        note: Free-text rationale/advisory.
    """

    timestamp: str
    sensors_fired: tuple[str, ...]
    fused_confidence: float
    classified_type: str = "unknown"
    track_quality: float = 1.0
    severity: str = "none"
    source_node: str = "unspecified"
    meets_fusion_baseline: bool = False
    note: str = ""
    schema: str = SCHEMA_VERSION
    kind: str = "detection_alert"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_TIERS:
            raise ValueError(
                f"severity must be one of {SEVERITY_TIERS}, got {self.severity!r}"
            )

    @property
    def severity_rank(self) -> int:
        """Ordinal rank of :attr:`severity` (higher = more severe)."""
        return severity_rank(self.severity)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (stable key order)."""
        return {
            "schema": self.schema,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "source_node": self.source_node,
            "classified_type": self.classified_type,
            "sensors_fired": list(self.sensors_fired),
            "fused_confidence": round(self.fused_confidence, 4),
            "track_quality": round(self.track_quality, 4),
            "meets_fusion_baseline": self.meets_fusion_baseline,
            "severity": self.severity,
            "severity_rank": self.severity_rank,
            "note": self.note,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise the alert to a JSON object string (UTF-8 safe)."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def from_fusion(
    result: FusionResult,
    *,
    timestamp: str,
    classified_type: str = "unknown",
    source_node: str = "unspecified",
) -> DetectionAlert:
    """Build a :class:`DetectionAlert` from a :class:`fusion.FusionResult`.

    Copies the fused confidence, fired modalities, track quality, severity and
    fusion-baseline flag straight through so the alert stays consistent with the
    scoring that produced it.
    """
    return DetectionAlert(
        timestamp=timestamp,
        sensors_fired=result.modalities_fired,
        fused_confidence=result.fused_confidence,
        classified_type=classified_type,
        track_quality=result.track_quality,
        severity=result.severity,
        source_node=source_node,
        meets_fusion_baseline=result.meets_fusion_baseline,
        note=result.rationale,
    )


def serialize_batch(alerts: list[DetectionAlert], *, indent: int = 2) -> str:
    """Serialise a list of alerts to a JSON array string."""
    return json.dumps([a.to_dict() for a in alerts], indent=indent, ensure_ascii=False)
