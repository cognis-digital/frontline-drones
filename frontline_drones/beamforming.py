"""Acoustic beamforming: bearing estimation with self-noise rejection.

A microphone array can localize a small drone by the *direction* its rotor tone
arrives from. The hard part on a moving platform is separating the target's sound
from the array's own motors and wind — the problem 2026 acoustic interceptors
(e.g. a 16-mic beamforming array) solve to hear an FPV over their own noise.

This module implements a narrowband delay-and-sum beamformer over an arbitrary
planar mic array: it forms an angular power spectrum from per-mic complex samples,
estimates the source bearing, and rejects a known self-noise bearing (the array's
own motors) with an angular guard band so the target is recovered even when the
self-noise is louder.

Scope: passive detection / direction-finding decision support only. It consumes
already-sampled array data; it emits nothing and carries no engagement content.
Pure stdlib (math, cmath), deterministic.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

SOUND_SPEED_MS = 343.0   # air at ~20C


@dataclass(frozen=True)
class MicArray:
    """Planar microphone array; positions are (x, y) meters in the array frame."""

    positions: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.positions) < 2:
            raise ValueError("need at least 2 microphones")

    @property
    def n(self) -> int:
        return len(self.positions)

    def _delays(self, bearing_deg: float) -> List[float]:
        """Per-mic propagation delay (s) for a plane wave from `bearing_deg`.

        Bearing is measured CCW from +x. A wave from bearing b has unit direction
        (cos b, sin b); the delay at mic p is -(p . dir)/c relative to the origin.
        """
        b = math.radians(bearing_deg)
        dx, dy = math.cos(b), math.sin(b)
        return [-(x * dx + y * dy) / SOUND_SPEED_MS for (x, y) in self.positions]

    def steering_vector(self, bearing_deg: float, freq_hz: float) -> List[complex]:
        """Narrowband steering vector for a source at bearing/freq."""
        return [cmath.exp(-2j * math.pi * freq_hz * d)
                for d in self._delays(bearing_deg)]


def simulate_samples(array: MicArray, freq_hz: float,
                     sources: Sequence[Tuple[float, float]]) -> List[complex]:
    """Synthesize per-mic complex samples from sources [(bearing_deg, amplitude)].

    Deterministic; used to model an array response for testing and analysis.
    """
    samples = [0j] * array.n
    for bearing, amp in sources:
        sv = array.steering_vector(bearing, freq_hz)
        for i in range(array.n):
            samples[i] += amp * sv[i]
    return samples


@dataclass(frozen=True)
class BearingEstimate:
    bearing_deg: float
    power: float
    rejected_self: bool

    def to_dict(self) -> dict:
        return {"bearing_deg": self.bearing_deg, "power": round(self.power, 6),
                "rejected_self": self.rejected_self}


def angular_spectrum(array: MicArray, samples: Sequence[complex], freq_hz: float,
                     scan_step_deg: float = 1.0) -> List[Tuple[float, float]]:
    """Delay-and-sum beam power over 0..360 deg. Returns [(bearing, power)]."""
    if scan_step_deg <= 0:
        raise ValueError("scan_step_deg must be > 0")
    out: List[Tuple[float, float]] = []
    steps = int(round(360.0 / scan_step_deg))
    for k in range(steps):
        phi = k * scan_step_deg
        sv = array.steering_vector(phi, freq_hz)
        # Coherent sum: conjugate-align each mic to candidate bearing.
        acc = sum(s * cmath.exp(2j * math.pi * freq_hz * 0) * v.conjugate()
                  for s, v in zip(samples, sv))
        out.append((phi, abs(acc) ** 2))
    return out


def _angular_diff(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def project_out(array: MicArray, samples: Sequence[complex], freq_hz: float,
                null_bearing_deg: float) -> List[complex]:
    """Null-steering: remove the component along a bearing's steering vector.

    Projects the sample vector onto the subspace orthogonal to the null bearing's
    steering vector, cancelling a strong interferer (the array's own motor noise)
    regardless of its amplitude — what a guard band alone cannot do.
    """
    a = array.steering_vector(null_bearing_deg, freq_hz)
    # coefficient = (a^H s) / (a^H a)
    num = sum(av.conjugate() * s for av, s in zip(a, samples))
    den = sum(abs(av) ** 2 for av in a) or 1.0
    coeff = num / den
    return [s - coeff * av for s, av in zip(samples, a)]


def estimate_bearing(array: MicArray, samples: Sequence[complex], freq_hz: float,
                     self_bearing_deg: Optional[float] = None,
                     guard_deg: float = 15.0,
                     scan_step_deg: float = 1.0) -> BearingEstimate:
    """Estimate source bearing, nulling a known self-noise bearing if given.

    When `self_bearing_deg` is provided, the self-noise steering vector is
    projected out (null-steering) so the array's own motors — even when far louder
    than the target — do not win the peak; a `guard_deg` band around the null keeps
    the residual sidelobe near the null from being selected.
    """
    work = list(samples)
    rejected = False
    if self_bearing_deg is not None:
        work = project_out(array, work, freq_hz, self_bearing_deg)
        rejected = True

    spectrum = angular_spectrum(array, work, freq_hz, scan_step_deg)
    best_phi, best_pow = None, -1.0
    for phi, p in spectrum:
        if self_bearing_deg is not None and _angular_diff(phi, self_bearing_deg) <= guard_deg:
            continue
        if p > best_pow:
            best_pow, best_phi = p, phi
    if best_phi is None:  # everything guarded out — fall back to global peak
        best_phi, best_pow = max(spectrum, key=lambda t: t[1])
    return BearingEstimate(best_phi, best_pow, rejected)


def uniform_circular_array(n: int, radius_m: float) -> MicArray:
    """Convenience: an n-element uniform circular array of given radius."""
    if n < 2:
        raise ValueError("need at least 2 microphones")
    if radius_m <= 0:
        raise ValueError("radius must be > 0")
    pos = tuple((radius_m * math.cos(2 * math.pi * i / n),
                 radius_m * math.sin(2 * math.pi * i / n)) for i in range(n))
    return MicArray(pos)
