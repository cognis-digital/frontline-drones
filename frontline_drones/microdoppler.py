"""Radar micro-Doppler feature helpers and drone-vs-bird discrimination.

A rotary-wing drone's rotors imprint a symmetric **micro-Doppler** signature on
a radar return: rotating blades produce paired sidebands (HERM lines - HElicopter
Rotor Modulation) around the bulk-body Doppler, and periodic blade flashes. Birds,
by contrast, flap at low, asymmetric rates (<=~10 Hz). This module computes the
radar-band physics behind those features and scores a return as *drone-like* vs
*bird-like* using the discriminants documented in ``radar-signatures.csv``.

Scope: detection/classification feature reference only. It processes already-
measured spectral features - it emits no waveform, does not transmit, and has no
tracking-for-intercept or engagement content. Pure stdlib, deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import load_dataset

# Speed of light (m/s).
SPEED_OF_LIGHT_MPS = 299_792_458.0

# Named radar bands and a representative centre frequency (GHz) for each. Used to
# turn a band token into a wavelength for the Doppler math.
RADAR_BANDS_GHZ: dict[str, float] = {
    "L": 1.5,
    "S": 3.0,
    "C": 5.5,
    "X": 9.5,
    "Ku": 15.0,
    "K": 22.0,
    "Ka": 35.0,
    "W": 94.0,
}

# A bird's wing-flap rate rarely exceeds this (Hz); blade-flash rates above it are
# a strong rotary-wing (drone) cue.
BIRD_MAX_FLAP_HZ = 12.0

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def wavelength_m(freq_ghz: float) -> float:
    """Radar wavelength (m) for a centre frequency in GHz."""
    if freq_ghz <= 0:
        raise ValueError("freq_ghz must be > 0")
    return SPEED_OF_LIGHT_MPS / (freq_ghz * 1e9)


def band_wavelength_m(band: str) -> float:
    """Wavelength (m) for a named radar band token (case-insensitive, e.g. ``"X"``)."""
    key = band.strip()
    for name, ghz in RADAR_BANDS_GHZ.items():
        if key.lower() == name.lower():
            return wavelength_m(ghz)
    valid = ", ".join(RADAR_BANDS_GHZ)
    raise KeyError(f"unknown radar band {band!r}; choose one of: {valid}")


def doppler_shift_hz(radial_velocity_mps: float, freq_ghz: float) -> float:
    """Two-way Doppler shift (Hz) for a target closing at ``radial_velocity_mps``."""
    return 2.0 * radial_velocity_mps / wavelength_m(freq_ghz)


def blade_tip_velocity_mps(rpm: float, rotor_radius_m: float) -> float:
    """Linear blade-tip speed (m/s) for a rotor at ``rpm`` of radius ``rotor_radius_m``."""
    if rpm < 0 or rotor_radius_m <= 0:
        raise ValueError("rpm must be >= 0 and rotor_radius_m must be > 0")
    return 2.0 * 3.141592653589793 * rotor_radius_m * rpm / 60.0


def blade_flash_rate_hz(rpm: float, blade_count: int) -> float:
    """Blade-flash rate (Hz): how often a blade presents a flat flash to the radar.

    Equal to the blade-passing frequency, ``rpm/60 * blade_count``. A rate far
    above :data:`BIRD_MAX_FLAP_HZ` is a strong rotary-wing (drone) indicator.
    """
    if rpm < 0 or blade_count <= 0:
        raise ValueError("rpm must be >= 0 and blade_count must be > 0")
    return rpm / 60.0 * blade_count


def micro_doppler_bandwidth_hz(tip_velocity_mps: float, freq_ghz: float) -> float:
    """Full micro-Doppler spread (Hz): +/- sidebands from advancing/retreating blades."""
    # Advancing and retreating tips give +/- the tip Doppler => 2x the one-way span.
    return 2.0 * doppler_shift_hz(tip_velocity_mps, freq_ghz)


def _first_number(text: str) -> float | None:
    m = _NUM_RE.search(text or "")
    return float(m.group()) if m else None


@dataclass(frozen=True)
class MicroDopplerAssessment:
    """Drone-vs-bird discrimination result for a micro-Doppler return.

    Attributes:
        drone_likelihood: [0, 1] score that the return is a rotary-wing drone.
        classification: ``"drone_like"``, ``"bird_like"`` or ``"ambiguous"``.
        symmetric: Whether the sidebands were reported symmetric (a drone cue).
        flash_rate_hz: The blade-flash / modulation rate assessed.
        cues: Human-readable discriminant cues that drove the score.
    """

    drone_likelihood: float
    classification: str
    symmetric: bool
    flash_rate_hz: float
    cues: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "drone_likelihood": round(self.drone_likelihood, 4),
            "classification": self.classification,
            "symmetric": self.symmetric,
            "flash_rate_hz": round(self.flash_rate_hz, 3),
            "cues": list(self.cues),
        }


def discriminate_bird(
    *,
    flash_rate_hz: float,
    symmetric_sidebands: bool,
    spectrum_symmetry: float = 1.0,
) -> MicroDopplerAssessment:
    """Score a micro-Doppler return as drone-like vs bird-like.

    Args:
        flash_rate_hz: Observed blade-flash / modulation rate. High rates
            (>> :data:`BIRD_MAX_FLAP_HZ`) are rotary-wing cues.
        symmetric_sidebands: Whether paired symmetric HERM sidebands were seen
            (rigid rotors -> symmetric; birds -> asymmetric flap).
        spectrum_symmetry: [0, 1] measured symmetry of the sidebands about the
            bulk Doppler (1 = perfectly symmetric).

    Returns:
        A :class:`MicroDopplerAssessment`. Deterministic.
    """
    if flash_rate_hz < 0:
        raise ValueError("flash_rate_hz must be >= 0")
    sym = max(0.0, min(1.0, spectrum_symmetry))
    cues: list[str] = []

    # Rate cue: sigmoid-ish ramp centred on the bird flap ceiling.
    if flash_rate_hz >= BIRD_MAX_FLAP_HZ:
        rate_score = min(1.0, 0.5 + (flash_rate_hz - BIRD_MAX_FLAP_HZ) / 40.0)
        cues.append(
            f"blade-flash rate {flash_rate_hz:.0f} Hz exceeds the ~{BIRD_MAX_FLAP_HZ:.0f} Hz "
            f"bird wing-flap ceiling (rotary-wing cue)."
        )
    else:
        rate_score = max(0.0, flash_rate_hz / (BIRD_MAX_FLAP_HZ * 2.0))
        cues.append(
            f"blade-flash rate {flash_rate_hz:.0f} Hz within the bird flap band "
            f"(<= ~{BIRD_MAX_FLAP_HZ:.0f} Hz)."
        )

    # Symmetry cue.
    sym_flag = bool(symmetric_sidebands)
    sym_score = sym if sym_flag else 0.5 * sym
    if sym_flag:
        cues.append("symmetric HERM sidebands present (rigid-rotor / drone cue).")
    else:
        cues.append("asymmetric spectrum (bird-flap-like).")

    likelihood = max(0.0, min(1.0, 0.6 * rate_score + 0.4 * sym_score))
    if likelihood >= 0.6:
        classification = "drone_like"
    elif likelihood <= 0.35:
        classification = "bird_like"
    else:
        classification = "ambiguous"

    return MicroDopplerAssessment(
        drone_likelihood=likelihood,
        classification=classification,
        symmetric=sym_flag,
        flash_rate_hz=flash_rate_hz,
        cues=tuple(cues),
    )


@dataclass(frozen=True)
class RadarBandAdvice:
    """Recommended radar band(s) for a platform class, from the library."""

    platform: str
    name: str
    recommended_band: str
    bird_discriminant: str
    source_url: str = ""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "name": self.name,
            "recommended_band": self.recommended_band,
            "bird_discriminant": self.bird_discriminant,
            "source_url": self.source_url,
        }


def recommended_bands(*, data_dir: str | None = None) -> list[RadarBandAdvice]:
    """Return the per-platform recommended radar band table from the library."""
    out: list[RadarBandAdvice] = []
    for row in load_dataset("radar", data_dir=data_dir):
        out.append(
            RadarBandAdvice(
                platform=row.get("platform") or row.get("id", ""),
                name=row.get("name", ""),
                recommended_band=row.get("recommended_band", ""),
                bird_discriminant=row.get("bird_discriminant", ""),
                source_url=row.get("source_url", ""),
            )
        )
    return out
