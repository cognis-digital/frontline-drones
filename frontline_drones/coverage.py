"""RF-blind coverage-gap analyzer (advisory only).

Operationalizes the defining 2024-2026 counter-UAS lesson: **RF-only detection
is blind to autonomous and fiber-optic-controlled drones**, which spool out a
physical fiber for control/video and emit zero RF signature. Given a modeled
threat and a chosen set of sensor modalities, this module flags coverage gaps
and advises which complementary modalities (typically radar and/or acoustic)
close them.

Scope: the output is a *warning/advisory report* about detection coverage. It
contains no engagement, mitigation or defeat logic. Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .fusion import MODALITIES

# Threat models this analyzer understands, each with the physics of what a
# detector can observe. ``rf_link`` is False for threats that present no
# exploitable control-link RF (autonomous waypoint, fiber-optic control).
@dataclass(frozen=True)
class ThreatModel:
    """How detectable a threat class is, per sensor physics.

    Attributes:
        name: Canonical threat token.
        label: Human-readable description.
        rf_link: Whether the threat radiates an exploitable control-link RF
            signal (False => RF/Remote-ID detection is blind to it).
        remote_id: Whether the threat is likely to broadcast ASTM F3411
            Remote ID (compliant consumer platforms do; adversary FPV/fiber
            and autonomous munitions generally do not).
        acoustic: Whether the threat has an exploitable acoustic signature.
        radar_note: Guidance on radar detectability.
    """

    name: str
    label: str
    rf_link: bool
    remote_id: bool
    acoustic: bool
    radar_note: str


THREATS: dict[str, ThreatModel] = {
    "consumer_dji": ThreatModel(
        name="consumer_dji",
        label="Consumer DJI-class quadcopter",
        rf_link=True,
        remote_id=True,
        acoustic=True,
        radar_note="Small multirotor RCS; micro-Doppler classifiable with X/Ku-band radar.",
    ),
    "fpv": ThreatModel(
        name="fpv",
        label="FPV attack/racing quad (analog or digital)",
        rf_link=True,
        remote_id=False,
        acoustic=True,
        radar_note="Very small RCS; needs high-band (Ku/W) radar and good clutter rejection.",
    ),
    "fiber_optic": ThreatModel(
        name="fiber_optic",
        label="Fiber-optic-controlled (FOC) drone",
        rf_link=False,
        remote_id=False,
        acoustic=True,
        radar_note="Emits zero RF; radar and acoustic are the primary detection paths.",
    ),
    "autonomous_waypoint": ThreatModel(
        name="autonomous_waypoint",
        label="Autonomous waypoint / INS-GNSS munition (e.g. Shahed-class)",
        rf_link=False,
        remote_id=False,
        acoustic=True,
        radar_note="No control-link RF; radar (bulk + prop Doppler) and acoustic detect it.",
    ),
}

THREAT_ALIASES: dict[str, str] = {
    "dji": "consumer_dji",
    "consumer": "consumer_dji",
    "racing": "fpv",
    "attack_fpv": "fpv",
    "foc": "fiber_optic",
    "fiber": "fiber_optic",
    "fibre_optic": "fiber_optic",
    "autonomous": "autonomous_waypoint",
    "waypoint": "autonomous_waypoint",
    "shahed": "autonomous_waypoint",
    "loitering_munition": "autonomous_waypoint",
}

# Modalities that can never see a threat with no RF link.
_RF_DEPENDENT = ("rf", "remote_id")


def resolve_threat(name: str) -> ThreatModel:
    """Return the :class:`ThreatModel` for a threat token or alias."""
    key = name.strip().lower().replace("-", "_")
    if key in THREATS:
        return THREATS[key]
    if key in THREAT_ALIASES:
        return THREATS[THREAT_ALIASES[key]]
    valid = ", ".join(sorted(THREATS))
    raise KeyError(f"unknown threat {name!r}; choose one of: {valid}")


def normalize_sensors(sensors: list[str] | tuple[str, ...]) -> list[str]:
    """Normalize/validate a list of sensor modality tokens.

    Accepts case-insensitive tokens and a few friendly spellings
    (``eo/ir``, ``eo-ir``, ``ir`` -> ``eo_ir``; ``remoteid`` -> ``remote_id``).
    Unknown tokens raise ``ValueError`` listing the valid modalities.
    """
    fixups = {
        "eo/ir": "eo_ir",
        "eo-ir": "eo_ir",
        "eoir": "eo_ir",
        "ir": "eo_ir",
        "eo": "eo_ir",
        "remoteid": "remote_id",
        "remote-id": "remote_id",
    }
    out: list[str] = []
    for s in sensors:
        key = s.strip().lower()
        key = fixups.get(key, key)
        if key not in MODALITIES:
            valid = ", ".join(MODALITIES)
            raise ValueError(f"unknown sensor modality {s!r}; choose from: {valid}")
        if key not in out:
            out.append(key)
    return out


@dataclass(frozen=True)
class CoverageReport:
    """Advisory report on detection coverage for a threat + sensor selection.

    Attributes:
        threat: The threat token analysed.
        sensors: Normalized modalities selected.
        gaps: Hard coverage gaps (the selection cannot detect the threat).
        warnings: Serious advisories (e.g. RF-only vs a no-RF threat).
        advisories: Softer notes and recommendations.
        recommended_additions: Modalities that would close the gaps.
        covered: True when at least one selected modality can detect the threat.
        meets_fusion_baseline: True when >= 2 *effective* modalities cover it.
    """

    threat: str
    sensors: tuple[str, ...]
    gaps: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    advisories: tuple[str, ...] = field(default_factory=tuple)
    recommended_additions: tuple[str, ...] = field(default_factory=tuple)
    covered: bool = True
    meets_fusion_baseline: bool = False

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the report."""
        return {
            "threat": self.threat,
            "sensors": list(self.sensors),
            "covered": self.covered,
            "meets_fusion_baseline": self.meets_fusion_baseline,
            "gaps": list(self.gaps),
            "warnings": list(self.warnings),
            "advisories": list(self.advisories),
            "recommended_additions": list(self.recommended_additions),
        }


def _effective_modalities(threat: ThreatModel, sensors: list[str]) -> list[str]:
    """Which selected modalities can actually detect this threat."""
    effective: list[str] = []
    for s in sensors:
        if s == "rf" and not threat.rf_link:
            continue
        if s == "remote_id" and not threat.remote_id:
            continue
        # radar, acoustic, eo_ir are physics-based and always effective here
        # (range/clutter caveats are surfaced as advisories, not gaps).
        effective.append(s)
    return effective


def analyze_coverage(
    threat: str,
    sensors: list[str] | tuple[str, ...],
) -> CoverageReport:
    """Flag detection coverage gaps for ``threat`` given selected ``sensors``.

    The core rule operationalizes the fiber-optic lesson: an RF-dependent-only
    selection (``rf`` and/or ``remote_id`` with nothing else) against a threat
    that emits no control-link RF is a **hard coverage gap**, and the report
    recommends adding radar and/or acoustic.

    Returns a :class:`CoverageReport`. Advisory only.
    """
    tm = resolve_threat(threat)
    sel = normalize_sensors(list(sensors))

    warnings: list[str] = []
    advisories: list[str] = []
    gaps: list[str] = []
    additions: list[str] = []

    effective = _effective_modalities(tm, sel)

    # Hard gap: no selected modality can detect this threat at all.
    if not effective:
        if not tm.rf_link and set(sel) <= set(_RF_DEPENDENT):
            gaps.append(
                f"RF-blind gap: {tm.label} emits no exploitable control-link RF, so an "
                f"RF/Remote-ID-only selection cannot detect it. This is the fiber-optic / "
                f"autonomous-drone lesson: RF-only detection (and RF jamming) is obsolete "
                f"against such threats."
            )
        else:
            gaps.append(
                f"No selected modality can detect {tm.label} with the current sensor set."
            )
        for want in ("radar", "acoustic"):
            if want not in sel:
                additions.append(want)
        warnings.append(
            "Add radar and/or acoustic detection to obtain any coverage of this threat."
        )

    # Soft advisory: RF present but ineffective against this threat.
    if "rf" in sel and not tm.rf_link:
        warnings.append(
            f"RF DF is included but {tm.label} has no control-link RF to detect; "
            f"do not rely on RF for this threat."
        )
    if "remote_id" in sel and not tm.remote_id:
        advisories.append(
            f"Remote ID is included but {tm.label} is unlikely to broadcast ASTM F3411; "
            f"treat any Remote ID as a spoofable aid, never trusted authentication."
        )

    # Modality-specific caveats when they ARE effective.
    if "acoustic" in effective:
        advisories.append(
            "Acoustic range is ~300-500 m and degrades with wind/ambient noise."
        )
    if "radar" in effective:
        advisories.append(tm.radar_note)
    if "eo_ir" in effective:
        advisories.append(
            "EO/IR needs slew-to-cue from another sensor and is limited by lighting/weather/FoV."
        )

    # Fusion-baseline advisory: credible detection fuses >= 2 effective modalities.
    meets_baseline = len(set(effective)) >= 2
    if effective and not meets_baseline:
        advisories.append(
            "Only one effective modality: below the >=2-modality fusion baseline for a "
            "credible detection; expect a higher false-alarm rate."
        )
        # Recommend a complementary physics-based modality.
        for want in ("radar", "acoustic", "eo_ir"):
            if want not in sel and want not in additions:
                additions.append(want)
                break

    covered = bool(effective)

    return CoverageReport(
        threat=tm.name,
        sensors=tuple(sel),
        gaps=tuple(gaps),
        warnings=tuple(warnings),
        advisories=tuple(advisories),
        recommended_additions=tuple(dict.fromkeys(additions)),
        covered=covered,
        meets_fusion_baseline=meets_baseline,
    )
