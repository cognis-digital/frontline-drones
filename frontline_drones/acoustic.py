"""Acoustic blade-passing-frequency (BPF) detection features and library match.

A multirotor's dominant acoustic cue is its **blade-passing frequency**: the
tonal peak (plus harmonics) a microphone array hears as each blade sweeps past.
For a rotor turning at ``rpm`` with ``blade_count`` blades the fundamental is
``BPF = rpm/60 * blade_count``. This module computes that physics, and matches an
observed fundamental against the shipped ``acoustic-signatures.csv`` reference to
produce ranked platform-class candidates with a confidence score.

Scope: passive acoustic detection/classification reference only. It ingests an
already-measured frequency feature; it neither records, transmits, nor emits any
sound, and carries no engagement content. Pure stdlib, deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .catalog import load_dataset

# Speed of sound in dry air at 20 C (m/s), for wavelength context.
SPEED_OF_SOUND_MPS = 343.0

_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def blade_passing_frequency(rpm: float, blade_count: int) -> float:
    """Blade-passing fundamental (Hz) for a rotor at ``rpm`` with ``blade_count`` blades."""
    if rpm < 0 or blade_count <= 0:
        raise ValueError("rpm must be >= 0 and blade_count must be > 0")
    return rpm / 60.0 * blade_count


def rpm_from_bpf(bpf_hz: float, blade_count: int) -> float:
    """Invert :func:`blade_passing_frequency`: rotor RPM from a measured fundamental."""
    if bpf_hz < 0 or blade_count <= 0:
        raise ValueError("bpf_hz must be >= 0 and blade_count must be > 0")
    return bpf_hz * 60.0 / blade_count


def harmonics(fundamental_hz: float, count: int = 5) -> list[float]:
    """Return the first ``count`` harmonics (2f, 3f, ...) of ``fundamental_hz``."""
    if count < 0:
        raise ValueError("count must be >= 0")
    return [fundamental_hz * n for n in range(2, count + 2)]


def wavelength_m(freq_hz: float) -> float:
    """Acoustic wavelength (m) of ``freq_hz`` in air at 20 C."""
    if freq_hz <= 0:
        raise ValueError("freq_hz must be > 0")
    return SPEED_OF_SOUND_MPS / freq_hz


def parse_hz_range(text: str) -> tuple[float, float] | None:
    """Parse a messy library cell like ``"~110-170"`` into ``(low, high)`` Hz.

    Returns ``None`` if no number is present. A single number yields
    ``(n, n)``. Handles a leading ``~`` and en/em dashes.
    """
    if not text:
        return None
    cleaned = text.replace("–", "-").replace("—", "-")
    nums = _RANGE_RE.findall(cleaned)
    if not nums:
        return None
    if len(nums) == 1:
        v = float(nums[0])
        return (v, v)
    lo, hi = float(nums[0]), float(nums[1])
    return (lo, hi) if lo <= hi else (hi, lo)


@dataclass(frozen=True)
class AcousticCandidate:
    """One ranked acoustic library match.

    Attributes:
        platform: Library platform token.
        name: Display name.
        score: Match confidence in [0, 1].
        in_band: True when the observed fundamental falls inside the library range.
        library_range_hz: The reference (low, high) fundamental band matched against.
        source_url: Citation for the reference row.
    """

    platform: str
    name: str
    score: float
    in_band: bool
    library_range_hz: tuple[float, float]
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "name": self.name,
            "score": round(self.score, 4),
            "in_band": self.in_band,
            "library_range_hz": list(self.library_range_hz),
            "source_url": self.source_url,
        }


def _proximity_score(value: float, lo: float, hi: float) -> tuple[float, bool]:
    """Score how well ``value`` fits the ``[lo, hi]`` band.

    In-band -> 1.0. Out of band -> decays with the distance to the nearest edge,
    scaled by the band width (or the edge value for a degenerate band), so a
    near miss still ranks above a far miss.
    """
    if lo <= value <= hi:
        return 1.0, True
    width = max(hi - lo, 1.0)
    dist = (lo - value) if value < lo else (value - hi)
    scale = max(width, hi * 0.25, 1.0)
    return max(0.0, 1.0 - dist / scale), False


def match_bpf(
    fundamental_hz: float,
    *,
    limit: int | None = None,
    data_dir: str | None = None,
) -> list[AcousticCandidate]:
    """Rank acoustic-signature library rows by fit to an observed ``fundamental_hz``.

    Reads ``acoustic-signatures.csv`` and scores each row's
    ``bpf_fundamental_hz`` band against the measurement. In-band rows score 1.0;
    near misses decay smoothly. Returns candidates sorted best-first (ties broken
    by platform token for determinism). ``limit`` caps the list length.
    """
    if fundamental_hz < 0:
        raise ValueError("fundamental_hz must be >= 0")
    out: list[AcousticCandidate] = []
    for row in load_dataset("acoustic", data_dir=data_dir):
        rng = parse_hz_range(row.get("bpf_fundamental_hz", ""))
        if rng is None:
            continue
        lo, hi = rng
        score, in_band = _proximity_score(fundamental_hz, lo, hi)
        if score <= 0.0:
            continue
        out.append(
            AcousticCandidate(
                platform=row.get("platform") or row.get("id", ""),
                name=row.get("name", ""),
                score=score,
                in_band=in_band,
                library_range_hz=(lo, hi),
                source_url=row.get("source_url", ""),
            )
        )
    out.sort(key=lambda c: (-c.score, c.platform))
    return out if limit is None else out[:limit]


@dataclass(frozen=True)
class AcousticDetection:
    """Result of scoring an acoustic observation against the signature library.

    Attributes:
        fundamental_hz: The observed blade-passing fundamental.
        estimated_rpm: Implied rotor RPM if ``blade_count`` was supplied, else None.
        candidates: Ranked :class:`AcousticCandidate` matches.
        best: The top candidate, or None if nothing matched.
        advisory: Range/limitation note (acoustic is short-range, wind-degraded).
    """

    fundamental_hz: float
    estimated_rpm: float | None
    candidates: tuple[AcousticCandidate, ...]
    best: AcousticCandidate | None
    advisory: str = ""

    def to_dict(self) -> dict:
        return {
            "fundamental_hz": round(self.fundamental_hz, 3),
            "estimated_rpm": (
                None if self.estimated_rpm is None else round(self.estimated_rpm, 1)
            ),
            "candidates": [c.to_dict() for c in self.candidates],
            "best": self.best.to_dict() if self.best else None,
            "advisory": self.advisory,
        }


ACOUSTIC_ADVISORY = (
    "Acoustic detection range is typically ~300-500 m and degrades with wind, "
    "ambient noise and airframe size; treat as short-range early warning and "
    "corroborate with another modality."
)


def detect(
    fundamental_hz: float,
    *,
    blade_count: int | None = None,
    limit: int | None = 5,
    data_dir: str | None = None,
) -> AcousticDetection:
    """Build an :class:`AcousticDetection` for an observed BPF fundamental.

    Optionally back-solves rotor RPM when ``blade_count`` is known. Runs the
    library match and attaches the short-range acoustic advisory. Awareness /
    classification only.
    """
    est_rpm = (
        rpm_from_bpf(fundamental_hz, blade_count) if blade_count else None
    )
    candidates = tuple(match_bpf(fundamental_hz, limit=limit, data_dir=data_dir))
    return AcousticDetection(
        fundamental_hz=fundamental_hz,
        estimated_rpm=est_rpm,
        candidates=candidates,
        best=candidates[0] if candidates else None,
        advisory=ACOUSTIC_ADVISORY,
    )
