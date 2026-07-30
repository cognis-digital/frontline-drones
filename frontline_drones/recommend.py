"""Sensor-selection recommender (detection only).

Encodes the ``docs/counter-uas-selection.md`` matrix as data: given a threat
model (consumer DJI / FPV / fiber-optic / autonomous-waypoint) and a deployment
scenario (fixed-site / mobile-convoy / dismounted / maritime), it emits a
recommended **fused detection** sensor stack with rationale, then runs the
coverage-gap analyzer to confirm the recommendation has no RF-blind hole.

Hard scope: this recommends **detection sensors only**. It explicitly refuses to
output any defeat/mitigation/jamming/engagement guidance. Detection is broadly
deployable; mitigation is heavily legally restricted and out of scope for this
tool. Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .coverage import CoverageReport, analyze_coverage, resolve_threat

MITIGATION_REFUSAL = (
    "Detection-only tool: no mitigation, defeat, jamming, spoofing or engagement "
    "guidance is provided. Mitigation is heavily legally restricted - obtain legal "
    "review (e.g. CISA guidance) before any action that intercepts or defeats a UAS."
)

# Deployment scenarios and the sensor-stack considerations they impose.
@dataclass(frozen=True)
class Scenario:
    """A deployment scenario and its detection-stack bias."""

    name: str
    label: str
    core: tuple[str, ...]
    optional: tuple[str, ...]
    rationale: str


SCENARIOS: dict[str, Scenario] = {
    "fixed_site": Scenario(
        name="fixed_site",
        label="Fixed critical infrastructure (airport, refinery, base)",
        core=("radar", "rf", "eo_ir"),
        optional=("acoustic", "remote_id"),
        rationale="Long-range volume coverage (radar) + type ID (RF) + forensic record (EO/IR).",
    ),
    "mobile_convoy": Scenario(
        name="mobile_convoy",
        label="Mobile / field / convoy",
        core=("radar", "rf"),
        optional=("acoustic", "eo_ir"),
        rationale="Quick-deploy compact radar + RF that still covers silent drones on the move.",
    ),
    "dismounted": Scenario(
        name="dismounted",
        label="Covert / dismounted / portable",
        core=("rf", "acoustic"),
        optional=("eo_ir",),
        rationale="Passive (no emissions), light, short-range early warning.",
    ),
    "maritime": Scenario(
        name="maritime",
        label="Maritime / port",
        core=("radar", "eo_ir"),
        optional=("rf", "acoustic"),
        rationale="Clutter-tolerant volume search over water (radar) + visual/thermal ID.",
    ),
}

SCENARIO_ALIASES: dict[str, str] = {
    "fixed": "fixed_site",
    "fixed-site": "fixed_site",
    "site": "fixed_site",
    "infrastructure": "fixed_site",
    "mobile": "mobile_convoy",
    "convoy": "mobile_convoy",
    "field": "mobile_convoy",
    "dismount": "dismounted",
    "portable": "dismounted",
    "covert": "dismounted",
    "maritime-port": "maritime",
    "port": "maritime",
    "naval": "maritime",
}


def resolve_scenario(name: str) -> Scenario:
    """Return the :class:`Scenario` for a token or alias."""
    key = name.strip().lower().replace(" ", "_")
    if key in SCENARIOS:
        return SCENARIOS[key]
    dash = key.replace("_", "-")
    if dash in SCENARIO_ALIASES:
        return SCENARIOS[SCENARIO_ALIASES[dash]]
    if key in SCENARIO_ALIASES:
        return SCENARIOS[SCENARIO_ALIASES[key]]
    valid = ", ".join(sorted(SCENARIOS))
    raise KeyError(f"unknown scenario {name!r}; choose one of: {valid}")


@dataclass(frozen=True)
class Recommendation:
    """A recommended fused detection stack for a threat + scenario.

    Attributes:
        threat / scenario: Canonical tokens analysed.
        core_sensors: The recommended fused detection core.
        optional_sensors: Useful add-ons for this scenario.
        rationale: Why this stack (scenario + threat reasoning).
        coverage: The :class:`CoverageReport` for ``core_sensors`` vs the threat.
        warnings: Coverage warnings surfaced from the analyzer.
        mitigation: Always the detection-only refusal string.
    """

    threat: str
    scenario: str
    core_sensors: tuple[str, ...]
    optional_sensors: tuple[str, ...]
    rationale: str
    coverage: CoverageReport
    warnings: tuple[str, ...] = field(default_factory=tuple)
    mitigation: str = MITIGATION_REFUSAL

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the recommendation."""
        return {
            "threat": self.threat,
            "scenario": self.scenario,
            "core_sensors": list(self.core_sensors),
            "optional_sensors": list(self.optional_sensors),
            "rationale": self.rationale,
            "warnings": list(self.warnings),
            "coverage": self.coverage.to_dict(),
            "mitigation": self.mitigation,
        }


def _threat_adjust(threat_name: str, core: list[str], optional: list[str]) -> tuple[list[str], str]:
    """Apply threat-specific mandatory additions to a scenario core.

    The fiber-optic / autonomous lesson: for zero-RF threats, radar and acoustic
    are *mandatory* and RF is demoted to optional (it cannot see the threat).
    """
    tm = resolve_threat(threat_name)
    note = ""
    core = list(core)
    optional = list(optional)

    if not tm.rf_link:
        # Mandate physics-based sensors; do not rely on RF.
        for want in ("radar", "acoustic"):
            if want not in core:
                core.append(want)
                if want in optional:
                    optional.remove(want)
        note = (
            f"{tm.label} emits no control-link RF: radar + acoustic are mandatory and RF "
            f"is demoted to optional (it cannot detect this threat)."
        )
        if "rf" in core:
            core.remove("rf")
            if "rf" not in optional:
                optional.append("rf")
    elif tm.name == "fpv":
        # Small-RCS FPV benefits from acoustic corroboration.
        if "acoustic" not in core and "acoustic" not in optional:
            optional.append("acoustic")
        note = (
            "FPV has a very small radar cross-section; keep RF for its 5.8 GHz/ELRS link and "
            "add acoustic corroboration."
        )
    else:
        note = "Consumer DJI-class: RF classifies the OcuSync link and Remote ID may aid ID."

    return core, optional, note  # type: ignore[return-value]


def recommend(threat: str, scenario: str) -> Recommendation:
    """Recommend a fused detection stack for ``threat`` in ``scenario``.

    Deterministic. Runs :func:`coverage.analyze_coverage` on the resulting core
    so the recommendation is guaranteed gap-checked (and self-corrects zero-RF
    threats toward radar/acoustic). Never emits mitigation guidance.
    """
    tm = resolve_threat(threat)
    sc = resolve_scenario(scenario)

    core, optional, threat_note = _threat_adjust(tm.name, list(sc.core), list(sc.optional))
    # De-duplicate while preserving order.
    core = list(dict.fromkeys(core))
    optional = [s for s in dict.fromkeys(optional) if s not in core]

    cov = analyze_coverage(tm.name, core)
    # If a residual gap remains, fold the analyzer's recommended additions into core.
    if cov.recommended_additions:
        for add in cov.recommended_additions:
            if add not in core:
                core.append(add)
        optional = [s for s in optional if s not in core]
        cov = analyze_coverage(tm.name, core)

    rationale = (
        f"Scenario '{sc.label}': {sc.rationale} "
        f"Threat '{tm.label}': {threat_note} "
        f"Credible C-UAS detection fuses >=2 modalities - this core provides "
        f"{cov.meets_fusion_baseline and 'that fusion baseline' or 'the available coverage'}."
    )

    return Recommendation(
        threat=tm.name,
        scenario=sc.name,
        core_sensors=tuple(core),
        optional_sensors=tuple(optional),
        rationale=rationale,
        coverage=cov,
        warnings=cov.warnings,
    )
