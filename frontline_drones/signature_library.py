"""Cross-modality signature classifier: fuse per-sensor candidate lists.

The single-modality classifiers (:mod:`frontline_drones.acoustic` BPF match,
:mod:`frontline_drones.rf_classify` control-link match) each return a ranked list
of platform candidates keyed by the same ``platform`` token used across the
signature CSVs. This module merges those lists into one **cross-modality platform
hypothesis**: a platform corroborated by two sensors outranks one seen by a lone
sensor, mirroring the fusion baseline the rest of the toolkit enforces.

Scope: classification / situational-awareness scoring only. It combines already-
produced detection candidates; it emits no tasking or engagement content. Pure
stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .acoustic import AcousticCandidate, match_bpf
from .rf_classify import RFCandidate, RFClassification, RFObservation, classify

# Per-modality trust weight when merging candidate scores. Mirrors the intent of
# fusion.MODALITY_WEIGHTS: RF classification is diagnostic but band-shared;
# acoustic is corroborative and short-range.
MODALITY_TRUST: dict[str, float] = {
    "rf": 0.85,
    "acoustic": 0.60,
}


@dataclass(frozen=True)
class PlatformHypothesis:
    """A cross-modality platform hypothesis.

    Attributes:
        platform: The shared platform token.
        name: Display name (first modality that supplies one).
        fused_score: Combined [0, 1] score across contributing modalities.
        modalities: Sorted names of modalities that nominated this platform.
        num_modalities: Count of ``modalities``.
        corroborated: True when >= 2 modalities nominated it.
        per_modality: Raw per-modality score for this platform.
    """

    platform: str
    name: str
    fused_score: float
    modalities: tuple[str, ...]
    num_modalities: int
    corroborated: bool
    per_modality: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "name": self.name,
            "fused_score": round(self.fused_score, 4),
            "modalities": list(self.modalities),
            "num_modalities": self.num_modalities,
            "corroborated": self.corroborated,
            "per_modality": {k: round(v, 4) for k, v in self.per_modality.items()},
        }


def _noisy_or(weighted: list[float]) -> float:
    """Bounded combine of independent evidences: ``1 - prod(1 - w_i)``."""
    product = 1.0
    for w in weighted:
        product *= 1.0 - max(0.0, min(1.0, w))
    return 1.0 - product


def fuse_candidates(
    *,
    rf: list[RFCandidate] | None = None,
    acoustic: list[AcousticCandidate] | None = None,
    limit: int | None = None,
) -> list[PlatformHypothesis]:
    """Merge per-modality candidate lists into ranked platform hypotheses.

    Each candidate contributes ``trust * score`` for its modality; a platform's
    modality contributions combine with a weighted noisy-OR so independent
    corroboration raises confidence without exceeding 1.0. Results sort by
    corroboration first (>=2 modalities beat 1), then fused score, then token.
    """
    names: dict[str, str] = {}
    per_mod: dict[str, dict[str, float]] = {}

    def ingest(modality: str, cands) -> None:
        trust = MODALITY_TRUST.get(modality, 0.5)
        for c in cands or []:
            plat = c.platform
            if not plat:
                continue
            per_mod.setdefault(plat, {})
            # Keep the strongest score if a platform appears twice in one list.
            contribution = trust * float(c.score)
            if contribution > per_mod[plat].get(modality, 0.0):
                per_mod[plat][modality] = contribution
            if plat not in names and getattr(c, "name", ""):
                names[plat] = c.name

    ingest("rf", rf)
    ingest("acoustic", acoustic)

    hyps: list[PlatformHypothesis] = []
    for plat, mods in per_mod.items():
        modalities = tuple(sorted(mods))
        fused = _noisy_or(list(mods.values()))
        hyps.append(
            PlatformHypothesis(
                platform=plat,
                name=names.get(plat, plat),
                fused_score=fused,
                modalities=modalities,
                num_modalities=len(modalities),
                corroborated=len(modalities) >= 2,
                per_modality=dict(mods),
            )
        )

    hyps.sort(key=lambda h: (-h.num_modalities, -h.fused_score, h.platform))
    return hyps if limit is None else hyps[:limit]


@dataclass(frozen=True)
class MultiModalObservation:
    """A bundle of what each passive sensor measured for one track.

    Any field may be ``None`` (sensor absent/not reporting).

    Attributes:
        rf: An :class:`rf_classify.RFObservation`, or None.
        acoustic_fundamental_hz: Observed acoustic BPF fundamental, or None.
    """

    rf: RFObservation | None = None
    acoustic_fundamental_hz: float | None = None


def classify_observation(
    obs: MultiModalObservation,
    *,
    limit: int | None = 5,
    data_dir: str | None = None,
) -> list[PlatformHypothesis]:
    """Run the per-modality classifiers on ``obs`` and fuse their candidates.

    Convenience wrapper: builds RF candidates via
    :func:`rf_classify.classify` and acoustic candidates via
    :func:`acoustic.match_bpf` from a single observation bundle, then calls
    :func:`fuse_candidates`. Awareness / classification only.
    """
    rf_cands: list[RFCandidate] = []
    if obs.rf is not None:
        result: RFClassification = classify(obs.rf, limit=None, data_dir=data_dir)
        rf_cands = list(result.candidates)

    ac_cands: list[AcousticCandidate] = []
    if obs.acoustic_fundamental_hz is not None:
        ac_cands = match_bpf(
            obs.acoustic_fundamental_hz, limit=None, data_dir=data_dir
        )

    return fuse_candidates(rf=rf_cands, acoustic=ac_cands, limit=limit)
