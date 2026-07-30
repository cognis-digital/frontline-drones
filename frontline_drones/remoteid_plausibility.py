"""Plausibility / spoof-flag checks for decoded Remote ID (awareness only).

ASTM F3411 Remote ID is a **spoofable identification aid, not authenticated
truth** (see :mod:`frontline_drones.remoteid`). This module operationalises that
caveat: it runs consistency checks over decoded Remote ID location fields - and
over a *sequence* of fixes - to flag values that are physically implausible or
internally inconsistent, the kind of thing a naive spoofer gets wrong (null-island
coordinates, teleporting positions, speeds beyond the airframe's envelope,
direction inconsistent with actual displacement).

It **never** asserts a message is genuine - a clean plausibility score only means
"nothing here contradicts itself", which a competent spoofer can also achieve. It
raises confidence in a *suspicious* message; it can never confer trust.

Scope: passive analysis of already-decoded fields. No transmission, no
engagement. Pure stdlib, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geo import haversine_m

# Rough top-speed envelope (m/s) by OpenDroneID UA type. Conservative upper
# bounds used only to flag *implausible* reported/implied speeds, not to identify.
UA_TYPE_MAX_SPEED_MPS: dict[str, float] = {
    "multirotor": 35.0,
    "aeroplane": 120.0,
    "hybrid_lift": 90.0,
    "gyroplane": 60.0,
    "ornithopter": 30.0,
    "glider": 80.0,
    "airship": 40.0,
    "free_balloon": 30.0,
    "captive_balloon": 10.0,
    "tethered_powered": 5.0,
    "rocket": 600.0,
}
# Fallback envelope for unknown/other UA types.
DEFAULT_MAX_SPEED_MPS = 120.0

# Plausible geodetic-altitude window (metres) for civil small-UAS Remote ID.
MIN_PLAUSIBLE_ALT_M = -430.0   # ~Dead Sea shore, the lowest dry land
MAX_PLAUSIBLE_ALT_M = 12_000.0


@dataclass(frozen=True)
class PlausibilityFlag:
    """One raised plausibility concern.

    Attributes:
        code: Stable machine token (e.g. ``"null_island"``).
        severity: ``"info"`` / ``"warn"`` / ``"suspect"``.
        detail: Human-readable explanation.
    """

    code: str
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass(frozen=True)
class PlausibilityReport:
    """Outcome of plausibility-checking Remote ID field(s).

    Attributes:
        plausible: True when no ``suspect``-severity flag was raised.
        score: [0, 1] plausibility score (1 = nothing inconsistent found).
        flags: All raised :class:`PlausibilityFlag` items.
        caveat: The permanent "clean != trusted" reminder.
    """

    plausible: bool
    score: float
    flags: tuple[PlausibilityFlag, ...] = field(default_factory=tuple)
    caveat: str = (
        "A clean plausibility result does NOT authenticate the message; Remote ID "
        "remains spoofable and unverified. This only flags self-contradiction."
    )

    def to_dict(self) -> dict:
        return {
            "plausible": self.plausible,
            "score": round(self.score, 4),
            "flags": [f.to_dict() for f in self.flags],
            "caveat": self.caveat,
        }


_SEVERITY_PENALTY = {"info": 0.05, "warn": 0.2, "suspect": 0.5}


def _finalize(flags: list[PlausibilityFlag]) -> PlausibilityReport:
    score = 1.0
    for f in flags:
        score -= _SEVERITY_PENALTY.get(f.severity, 0.2)
    score = max(0.0, min(1.0, score))
    plausible = not any(f.severity == "suspect" for f in flags)
    return PlausibilityReport(plausible=plausible, score=score, flags=tuple(flags))


def check_location_fields(fields: dict) -> PlausibilityReport:
    """Plausibility-check a single decoded Location message's ``fields`` dict.

    Looks at latitude/longitude bounds, the null-island signature, altitude
    window, direction range and reported speed vs the UA-type envelope. Returns
    a :class:`PlausibilityReport`. The ``ua_type`` key (from a paired Basic ID
    message) tightens the speed envelope when present.
    """
    flags: list[PlausibilityFlag] = []

    lat = _as_float(fields.get("latitude"))
    lon = _as_float(fields.get("longitude"))
    if lat is not None and lon is not None:
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            flags.append(
                PlausibilityFlag(
                    "coords_out_of_range", "suspect",
                    f"latitude/longitude out of valid range ({lat}, {lon}).",
                )
            )
        elif abs(lat) < 1e-6 and abs(lon) < 1e-6:
            flags.append(
                PlausibilityFlag(
                    "null_island", "suspect",
                    "coordinates at (0, 0) 'null island' - a classic default/spoof value.",
                )
            )

    alt = _as_float(fields.get("geodetic_altitude_m"))
    if alt is not None and not (MIN_PLAUSIBLE_ALT_M <= alt <= MAX_PLAUSIBLE_ALT_M):
        flags.append(
            PlausibilityFlag(
                "altitude_implausible", "warn",
                f"geodetic altitude {alt} m outside the plausible small-UAS window "
                f"[{MIN_PLAUSIBLE_ALT_M}, {MAX_PLAUSIBLE_ALT_M}] m.",
            )
        )

    direction = _as_float(fields.get("direction_deg"))
    if direction is not None and not (0.0 <= direction <= 359.0):
        flags.append(
            PlausibilityFlag(
                "direction_out_of_range", "warn",
                f"track direction {direction} deg outside 0-359.",
            )
        )

    speed = _as_float(fields.get("speed_mps"))
    if speed is not None:
        if speed < 0:
            flags.append(
                PlausibilityFlag("negative_speed", "suspect", f"negative speed {speed} m/s.")
            )
        else:
            envelope = _speed_envelope(fields.get("ua_type"))
            if speed > envelope:
                flags.append(
                    PlausibilityFlag(
                        "speed_exceeds_envelope", "warn",
                        f"reported speed {speed} m/s exceeds the ~{envelope:.0f} m/s "
                        f"envelope for UA type {fields.get('ua_type', 'unknown')!r}.",
                    )
                )

    status = fields.get("operational_status")
    if status == "airborne" and speed == 0.0 and fields.get("height_m") in (0.0, None):
        flags.append(
            PlausibilityFlag(
                "airborne_but_static", "info",
                "status 'airborne' with zero speed and zero height - possibly stale/default.",
            )
        )

    return _finalize(flags)


def check_track(fixes: list[dict], *, ua_type: str | None = None) -> PlausibilityReport:
    """Plausibility-check a time-ordered sequence of Location ``fixes``.

    Each fix is a dict with ``latitude``, ``longitude`` and a ``t`` timestamp in
    seconds. Beyond per-fix checks, this flags **teleports**: an implied speed
    between consecutive fixes that exceeds the UA-type envelope, the tell-tale of
    a coordinate-injecting spoofer. Returns a combined :class:`PlausibilityReport`.
    """
    flags: list[PlausibilityFlag] = []
    envelope = _speed_envelope(ua_type)

    # Per-fix checks first (reuse the single-message logic, merge flags).
    for fix in fixes:
        merged = dict(fix)
        if ua_type is not None and "ua_type" not in merged:
            merged["ua_type"] = ua_type
        for f in check_location_fields(merged).flags:
            flags.append(f)

    ordered = sorted(
        (f for f in fixes if f.get("t") is not None), key=lambda f: f["t"]
    )
    for a, b in zip(ordered, ordered[1:], strict=False):
        dt = float(b["t"]) - float(a["t"])
        if dt <= 0:
            flags.append(
                PlausibilityFlag(
                    "nonmonotonic_time", "warn",
                    f"non-increasing timestamps ({a['t']} -> {b['t']}).",
                )
            )
            continue
        la, lo_a = _as_float(a.get("latitude")), _as_float(a.get("longitude"))
        lb, lo_b = _as_float(b.get("latitude")), _as_float(b.get("longitude"))
        if None in (la, lo_a, lb, lo_b):
            continue
        dist = haversine_m(la, lo_a, lb, lo_b)
        implied = dist / dt
        if implied > envelope * 1.5:
            flags.append(
                PlausibilityFlag(
                    "teleport", "suspect",
                    f"implied speed {implied:.0f} m/s between fixes far exceeds the "
                    f"~{envelope:.0f} m/s envelope - position teleport / injection.",
                )
            )

    return _finalize(flags)


def _speed_envelope(ua_type: object) -> float:
    if isinstance(ua_type, str):
        return UA_TYPE_MAX_SPEED_MPS.get(ua_type, DEFAULT_MAX_SPEED_MPS)
    return DEFAULT_MAX_SPEED_MPS


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
