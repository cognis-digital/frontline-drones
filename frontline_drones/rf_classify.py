"""Passive RF control-link classifier against the signature library.

Given features a **passive** RF direction-finder / spectrum sensor already
measured - which ISM band(s) the control/video link occupies, whether it
frequency-hops, its channel bandwidth and any protocol hint - this module ranks
the ``rf-signatures.csv`` library rows to suggest likely platform classes, with a
confidence score and the spoofability/uncertainty caveats such a classification
carries.

Scope: passive receive-side classification only. It never tunes, transmits,
jams or spoofs; it scores already-observed features. Pure stdlib, deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import Row, load_dataset

RF_CLASSIFY_CAVEAT = (
    "Passive RF classification is a probabilistic aid: control links share ISM "
    "bands and hopping schemes, adversary FPV can run arbitrary firmware, and "
    "fiber-optic / autonomous drones emit no RF at all. Corroborate with a "
    "second modality before acting on any RF-only identification."
)

# Common band tokens normalised to a canonical spelling.
_BAND_FIXUPS: dict[str, str] = {
    "2.4ghz": "2.4 GHz",
    "2.4 ghz": "2.4 GHz",
    "2400mhz": "2.4 GHz",
    "5.8ghz": "5.8 GHz",
    "5.8 ghz": "5.8 GHz",
    "900mhz": "900 MHz",
    "915mhz": "900 MHz",
    "433mhz": "433 MHz",
    "1.2ghz": "1.2 GHz",
    "1.3ghz": "1.2 GHz",
}

# Matches one or more slash-separated numbers that share a trailing unit, e.g.
# "2.4/5.8 GHz" -> numbers "2.4","5.8" all in GHz.
_BAND_RE = re.compile(r"((?:\d+(?:\.\d+)?/)*\d+(?:\.\d+)?)\s*(ghz|mhz)", re.IGNORECASE)


def normalize_bands(text: str) -> set[str]:
    """Extract a set of canonical band tokens (``"2.4 GHz"``, ``"5.8 GHz"`` ...).

    Reads any free-text band description (``"2.4/5.8 GHz ISM"``, ``"5.8ghz"``)
    and returns the GHz/MHz centre tokens it mentions, deduplicated.
    """
    if not text:
        return set()
    found: set[str] = set()
    low = text.lower()
    for key, canon in _BAND_FIXUPS.items():
        if key in low.replace(" ", ""):
            found.add(canon)
    for nums, unit in _BAND_RE.findall(text):
        canon_unit = unit.upper().replace("GHZ", "GHz").replace("MHZ", "MHz")
        for num in nums.split("/"):
            found.add(f"{num} {canon_unit}")
    return found


def _hop_token(text: str) -> str:
    """Reduce a hopping description to ``"fhss"``, ``"fixed"`` or ``"unknown"``."""
    low = (text or "").lower()
    if any(t in low for t in ("fhss", "hop", "frequency-hopping", "frequency hopping")):
        return "fhss"
    if any(t in low for t in ("fixed", "static", "single channel", "non-hopping")):
        return "fixed"
    return "unknown"


@dataclass(frozen=True)
class RFObservation:
    """Features a passive RF sensor measured for one control link.

    Attributes:
        bands: Observed band tokens (free spellings accepted; normalised on use).
        hopping: ``"fhss"`` / ``"fixed"`` / ``"unknown"`` or any free text.
        channel_bandwidth_mhz: Observed occupied bandwidth, MHz (optional).
        protocol_hint: Free-text protocol/vendor hint (e.g. ``"ocusync"``).
    """

    bands: tuple[str, ...] = field(default_factory=tuple)
    hopping: str = "unknown"
    channel_bandwidth_mhz: float | None = None
    protocol_hint: str = ""


@dataclass(frozen=True)
class RFCandidate:
    """One ranked RF library match.

    Attributes:
        platform: Library platform token.
        name: Display name.
        vendor: Reported vendor.
        score: Match confidence in [0, 1].
        band_overlap: Count of observed bands that overlap the library row.
        matched_on: Which feature dimensions contributed to the score.
        source_url: Citation.
    """

    platform: str
    name: str
    vendor: str
    score: float
    band_overlap: int
    matched_on: tuple[str, ...]
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "name": self.name,
            "vendor": self.vendor,
            "score": round(self.score, 4),
            "band_overlap": self.band_overlap,
            "matched_on": list(self.matched_on),
            "source_url": self.source_url,
        }


def _score_row(obs: RFObservation, row: Row) -> tuple[float, int, list[str]]:
    """Score one library row against an observation. Returns (score, overlap, dims)."""
    obs_bands = set()
    for b in obs.bands:
        obs_bands |= normalize_bands(b)
    lib_bands = normalize_bands(row.get("control_band", ""))
    overlap = len(obs_bands & lib_bands) if obs_bands else 0

    dims: list[str] = []
    band_component = 0.0
    if obs_bands and lib_bands:
        # Jaccard-like overlap of band sets.
        union = len(obs_bands | lib_bands)
        band_component = overlap / union if union else 0.0
        if overlap:
            dims.append("band")

    hop_component = 0.0
    obs_hop = _hop_token(obs.hopping)
    lib_hop = _hop_token(row.get("frequency_hopping", ""))
    if obs_hop != "unknown" and lib_hop != "unknown":
        if obs_hop == lib_hop:
            hop_component = 1.0
            dims.append("hopping")
        else:
            hop_component = 0.0

    proto_component = 0.0
    if obs.protocol_hint:
        hint = obs.protocol_hint.lower()
        hay = (row.get("video_protocol", "") + " " + row.get("name", "")).lower()
        if hint and hint in hay:
            proto_component = 1.0
            dims.append("protocol")

    bw_component = 0.0
    if obs.channel_bandwidth_mhz is not None:
        rng = _parse_bw_range(row.get("channel_bandwidth_mhz", ""))
        if rng is not None:
            lo, hi = rng
            if lo <= obs.channel_bandwidth_mhz <= hi:
                bw_component = 1.0
                dims.append("bandwidth")

    # Weighted blend; band overlap dominates because it is the most diagnostic
    # passive feature, hopping/protocol/bandwidth corroborate.
    score = (
        0.55 * band_component
        + 0.20 * hop_component
        + 0.15 * proto_component
        + 0.10 * bw_component
    )
    return score, overlap, dims


_BW_RE = re.compile(r"\d+(?:\.\d+)?")


def _parse_bw_range(text: str) -> tuple[float, float] | None:
    if not text:
        return None
    nums = _BW_RE.findall(text.replace("~", ""))
    if not nums:
        return None
    if len(nums) == 1:
        v = float(nums[0])
        return (v, v)
    lo, hi = float(nums[0]), float(nums[1])
    return (lo, hi) if lo <= hi else (hi, lo)


@dataclass(frozen=True)
class RFClassification:
    """Ranked RF classification result with the mandatory spoofability caveat."""

    candidates: tuple[RFCandidate, ...]
    best: RFCandidate | None
    caveat: str = RF_CLASSIFY_CAVEAT

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "best": self.best.to_dict() if self.best else None,
            "caveat": self.caveat,
        }


def classify(
    obs: RFObservation,
    *,
    limit: int | None = 5,
    min_score: float = 0.05,
    data_dir: str | None = None,
) -> RFClassification:
    """Rank ``rf-signatures.csv`` rows against a passive RF ``obs``.

    Scores each library row on band overlap (primary), hopping behaviour,
    protocol hint and channel bandwidth. Rows scoring below ``min_score`` are
    dropped; results are sorted best-first (ties broken by platform token).
    Always attaches :data:`RF_CLASSIFY_CAVEAT`. Deterministic.
    """
    out: list[RFCandidate] = []
    for row in load_dataset("rf", data_dir=data_dir):
        score, overlap, dims = _score_row(obs, row)
        if score < min_score:
            continue
        out.append(
            RFCandidate(
                platform=row.get("platform") or row.get("id", ""),
                name=row.get("name", ""),
                vendor=row.get("vendor", ""),
                score=score,
                band_overlap=overlap,
                matched_on=tuple(dims),
                source_url=row.get("source_url", ""),
            )
        )
    out.sort(key=lambda c: (-c.score, c.platform))
    if limit is not None:
        out = out[:limit]
    return RFClassification(candidates=tuple(out), best=out[0] if out else None)
