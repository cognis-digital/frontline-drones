"""Sensor cueing and cross-sensor track handoff (detection/awareness only).

A layered counter-UAS site typically pairs **wide-area detectors** (surveillance
radar, RF direction-finding, an acoustic array) that watch a large volume and
produce coarse *tracks*, with **narrow-field-of-view trackers** (an EO/IR
gimbal, a dish-fed tracking radar) that must be *slewed onto* a track to observe
it closely. This module does the deterministic geometry and scheduling for two
purely defensive jobs:

* **Cueing** - given a track reported by a wide-area detector, compute the
  pointing command (bearing, elevation, range, slew time) needed to bring a
  narrow-FOV sensor onto it, optionally *leading* the target so the sensor
  points where the track will be once the slew finishes.

* **Handoff** - as a track flies through overlapping sensor footprints, decide
  which sensor should hold custody, predict when the current sensor will lose
  the track, and choose the successor that keeps the track covered with the
  smallest coverage gap. A whole-path *custody schedule* stitches the sensors
  into a hand-off chain that minimises both gaps and the number of handoffs.

Scope: this is observation bookkeeping - where to point a *sensor* and when to
pass a *track* between sensors so awareness is not lost. Sensors here are
detection/tracking instruments, never effectors. There is no engagement,
weapon, jamming, fire-control, targeting or intercept content of any kind: a
"coverage gap" is a gap in *our own observation* to be closed with more sensor
dwell, not an opening to exploit. The constant-velocity path model exists only
to predict where a track will be so a sensor can keep watching it.

Coordinates are a local ENU plane in metres (east, north) with an optional
``up`` altitude in metres; angles are degrees (compass bearing, 0 = North,
90 = East); time is seconds, measured *relative to now* (``t = 0`` is the
track's current position). Pure standard library, deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Advisory role tokens. A sensor's role never changes any geometry here; it only
# documents intent (a wide-area detector that cues vs. a narrow tracker cued).
SENSOR_ROLES: tuple[str, ...] = ("detector", "tracker")

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Angle / bearing helpers
# --------------------------------------------------------------------------- #

def bearing_deg(delta_east: float, delta_north: float) -> float:
    """Compass bearing of an ENU offset, degrees in ``[0, 360)`` (0 = N, 90 = E).

    Returns 0.0 for a zero-length offset (no defined direction).
    """
    if delta_east == 0.0 and delta_north == 0.0:
        return 0.0
    return (math.degrees(math.atan2(delta_east, delta_north)) + 360.0) % 360.0


def angular_sep_deg(a_deg: float, b_deg: float) -> float:
    """Smallest unsigned angle between two bearings, degrees in ``[0, 180]``."""
    d = (a_deg - b_deg) % 360.0
    return d if d <= 180.0 else 360.0 - d


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Sensor:
    """A detection/tracking sensor with a modeled coverage footprint.

    The footprint is a horizontal annulus centred on the sensor: a track is
    within coverage when its ground range lies in ``[min_range_m, max_range_m]``.
    The ``min_range_m`` inner radius models a near dead-zone (e.g. minimum radar
    range or a gimbal that cannot look straight down).

    Attributes:
        sensor_id: Stable identifier (int or string token).
        east / north: Sensor position in the local ENU plane, metres.
        max_range_m: Outer detection radius, metres (> 0).
        min_range_m: Inner dead-zone radius, metres (>= 0, < ``max_range_m``).
        height_m: Sensor height above the ENU ground plane, metres.
        slew_rate_dps: Azimuth slew rate, degrees per second (> 0). Used to size
            the time to point a narrow sensor onto a cued bearing.
        settle_s: Fixed settle/acquire time added after a slew, seconds (>= 0).
        fov_deg: Advisory field-of-view width for a narrow tracker, degrees.
        role: Advisory role, one of :data:`SENSOR_ROLES`.
    """

    sensor_id: object
    east: float
    north: float
    max_range_m: float
    min_range_m: float = 0.0
    height_m: float = 0.0
    slew_rate_dps: float = 30.0
    settle_s: float = 0.0
    fov_deg: float = 0.0
    role: str = "tracker"

    def __post_init__(self) -> None:
        if self.max_range_m <= 0:
            raise ValueError("max_range_m must be positive")
        if self.min_range_m < 0:
            raise ValueError("min_range_m must be non-negative")
        if self.min_range_m >= self.max_range_m:
            raise ValueError("min_range_m must be < max_range_m")
        if self.slew_rate_dps <= 0:
            raise ValueError("slew_rate_dps must be positive")
        if self.settle_s < 0:
            raise ValueError("settle_s must be non-negative")
        if self.role not in SENSOR_ROLES:
            valid = ", ".join(SENSOR_ROLES)
            raise ValueError(f"role must be one of: {valid}")

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "east": self.east,
            "north": self.north,
            "max_range_m": self.max_range_m,
            "min_range_m": self.min_range_m,
            "height_m": self.height_m,
            "slew_rate_dps": self.slew_rate_dps,
            "settle_s": self.settle_s,
            "fov_deg": self.fov_deg,
            "role": self.role,
        }


@dataclass(frozen=True)
class Target:
    """A track's current state on a constant-velocity path (awareness only).

    ``t = 0`` corresponds to the position ``(east, north, up)``; the predicted
    position at relative time ``t`` seconds is ``(east + vel_east * t,
    north + vel_north * t, up)`` (altitude held constant).

    Attributes:
        track_id: Identifier carried through to cue/handoff outputs.
        east / north: Current ENU position, metres.
        vel_east / vel_north: Ground velocity, m/s (0 => stationary).
        up: Altitude above the ENU ground plane, metres.
    """

    track_id: object = 0
    east: float = 0.0
    north: float = 0.0
    vel_east: float = 0.0
    vel_north: float = 0.0
    up: float = 0.0

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vel_east, self.vel_north)

    def position_at(self, t: float) -> tuple[float, float]:
        """Predicted ground position ``(east, north)`` at relative time ``t``."""
        return (self.east + self.vel_east * t, self.north + self.vel_north * t)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "east": self.east,
            "north": self.north,
            "vel_east": self.vel_east,
            "vel_north": self.vel_north,
            "up": self.up,
            "speed_mps": round(self.speed_mps, 6),
        }


def as_target(item, *, track_id: object = 0) -> Target:
    """Coerce a caller-supplied value into a :class:`Target`.

    Accepts a :class:`Target`, a mapping with ``east``/``north`` (and optional
    ``vel_east``/``vel_north``/``up``/``track_id``; ``x``/``y`` are accepted as
    aliases for east/north), any object exposing those attributes (e.g. a
    :class:`frontline_drones.track_association.Track`), or an ``(east, north)``
    / ``(east, north, vel_east, vel_north)`` sequence.
    """
    if isinstance(item, Target):
        return item
    if isinstance(item, dict):
        return Target(
            track_id=item.get("track_id", track_id),
            east=float(item.get("east", item.get("x", 0.0))),
            north=float(item.get("north", item.get("y", 0.0))),
            vel_east=float(item.get("vel_east", 0.0)),
            vel_north=float(item.get("vel_north", 0.0)),
            up=float(item.get("up", 0.0)),
        )
    if hasattr(item, "east") and hasattr(item, "north"):
        return Target(
            track_id=getattr(item, "track_id", track_id),
            east=float(item.east),
            north=float(item.north),
            vel_east=float(getattr(item, "vel_east", 0.0)),
            vel_north=float(getattr(item, "vel_north", 0.0)),
            up=float(getattr(item, "up", 0.0)),
        )
    seq = list(item)
    east = float(seq[0])
    north = float(seq[1])
    ve = float(seq[2]) if len(seq) > 2 else 0.0
    vn = float(seq[3]) if len(seq) > 3 else 0.0
    return Target(track_id=track_id, east=east, north=north, vel_east=ve, vel_north=vn)


# --------------------------------------------------------------------------- #
# Instantaneous geometry
# --------------------------------------------------------------------------- #

def ground_range_m(sensor: Sensor, target: Target, t: float = 0.0) -> float:
    """Horizontal range from ``sensor`` to the track's position at time ``t``."""
    pe, pn = target.position_at(t)
    return math.hypot(pe - sensor.east, pn - sensor.north)


def slant_range_m(sensor: Sensor, target: Target, t: float = 0.0) -> float:
    """3-D range from ``sensor`` to the track's position at time ``t``."""
    gr = ground_range_m(sensor, target, t)
    return math.hypot(gr, target.up - sensor.height_m)


def elevation_deg(sensor: Sensor, target: Target, t: float = 0.0) -> float:
    """Elevation angle to the track at time ``t``, degrees (negative = below)."""
    gr = ground_range_m(sensor, target, t)
    dz = target.up - sensor.height_m
    if gr == 0.0 and dz == 0.0:
        return 0.0
    return math.degrees(math.atan2(dz, gr))


def bearing_to(sensor: Sensor, target: Target, t: float = 0.0) -> float:
    """Compass bearing from ``sensor`` to the track's position at time ``t``."""
    pe, pn = target.position_at(t)
    return bearing_deg(pe - sensor.east, pn - sensor.north)


def in_coverage(sensor: Sensor, target: Target, t: float = 0.0) -> bool:
    """True when the track at time ``t`` is inside the sensor's coverage annulus."""
    gr = ground_range_m(sensor, target, t)
    return sensor.min_range_m - _EPS <= gr <= sensor.max_range_m + _EPS


def slew_time_s(sensor: Sensor, from_bearing_deg: float, to_bearing_deg: float) -> float:
    """Time for ``sensor`` to slew from one bearing to another, seconds.

    Uses the shortest angular travel divided by the slew rate, plus the fixed
    settle time. Deterministic; never negative.
    """
    sep = angular_sep_deg(from_bearing_deg, to_bearing_deg)
    return sep / sensor.slew_rate_dps + sensor.settle_s


# --------------------------------------------------------------------------- #
# Cueing
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CueCommand:
    """A pointing command to bring a sensor onto a track (observation only).

    Attributes:
        sensor_id: The sensor being cued.
        track_id: The track it is cued to.
        bearing_deg: Commanded compass bearing, degrees.
        elevation_deg: Commanded elevation, degrees.
        ground_range_m: Horizontal range to the commanded aim point, metres.
        slant_range_m: 3-D range to the commanded aim point, metres.
        slew_time_s: Time to slew from ``from_bearing`` to ``bearing_deg`` and
            settle, seconds.
        lead_time_s: The look-ahead used to place the aim point (0 when not
            leading). Equals ``slew_time_s`` for a lead cue.
        aim_east / aim_north: The predicted ground aim point, metres.
        in_coverage: Whether the aim point lies in the sensor's coverage annulus.
        feasible: Whether the cue is usable (in coverage).
        advisories: Plain-language notes (out of coverage, dead-zone, etc.).
    """

    sensor_id: object
    track_id: object
    bearing_deg: float
    elevation_deg: float
    ground_range_m: float
    slant_range_m: float
    slew_time_s: float
    lead_time_s: float
    aim_east: float
    aim_north: float
    in_coverage: bool
    feasible: bool
    advisories: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "track_id": self.track_id,
            "bearing_deg": round(self.bearing_deg, 4),
            "elevation_deg": round(self.elevation_deg, 4),
            "ground_range_m": round(self.ground_range_m, 4),
            "slant_range_m": round(self.slant_range_m, 4),
            "slew_time_s": round(self.slew_time_s, 6),
            "lead_time_s": round(self.lead_time_s, 6),
            "aim_east": round(self.aim_east, 4),
            "aim_north": round(self.aim_north, 4),
            "in_coverage": self.in_coverage,
            "feasible": self.feasible,
            "advisories": list(self.advisories),
        }


def cue_sensor(
    sensor: Sensor,
    target,
    *,
    from_bearing_deg: float = 0.0,
    lead: bool = True,
    iterations: int = 8,
) -> CueCommand:
    """Compute a pointing command to bring ``sensor`` onto ``target``.

    When ``lead`` is true and the track is moving, the aim point is *led*: the
    slew time is estimated, the track is predicted forward by that time, and the
    process is repeated for ``iterations`` fixed-point steps so the sensor points
    where the track will be once the slew completes (not where it was). With
    ``lead`` false the sensor is aimed at the track's current position.

    Deterministic. Cueing/observation only - the command points a sensor, never
    an effector.
    """
    tgt = as_target(target)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if lead and tgt.speed_mps > 0.0:
        t_lead = 0.0
        for _ in range(iterations):
            pe, pn = tgt.position_at(t_lead)
            brg = bearing_deg(pe - sensor.east, pn - sensor.north)
            t_lead = slew_time_s(sensor, from_bearing_deg, brg)
    else:
        brg0 = bearing_to(sensor, tgt, 0.0)
        t_lead = 0.0 if not lead else slew_time_s(sensor, from_bearing_deg, brg0)

    aim_e, aim_n = tgt.position_at(t_lead)
    aim = Target(track_id=tgt.track_id, east=aim_e, north=aim_n, up=tgt.up)

    brg = bearing_to(sensor, aim, 0.0)
    gr = ground_range_m(sensor, aim, 0.0)
    sr = slant_range_m(sensor, aim, 0.0)
    elev = elevation_deg(sensor, aim, 0.0)
    slew = slew_time_s(sensor, from_bearing_deg, brg)
    covered = in_coverage(sensor, aim, 0.0)

    advisories: list[str] = []
    if not covered:
        if gr > sensor.max_range_m:
            advisories.append(
                f"Aim point is {gr - sensor.max_range_m:.1f} m beyond the sensor's "
                f"{sensor.max_range_m:.0f} m outer range; cue is out of coverage."
            )
        elif gr < sensor.min_range_m:
            advisories.append(
                f"Aim point is inside the {sensor.min_range_m:.0f} m near dead-zone; "
                f"sensor cannot observe it."
            )
    if lead and tgt.speed_mps > 0.0 and covered and not in_coverage(sensor, tgt, 0.0):
        advisories.append(
            "Track is not yet in coverage but is predicted to be within it once the "
            "slew completes; pre-position the sensor now."
        )
    if lead and tgt.speed_mps > 0.0 and in_coverage(sensor, tgt, 0.0) and not covered:
        advisories.append(
            "Track is in coverage now but predicted to leave it during the slew; "
            "expect a short observation gap."
        )

    return CueCommand(
        sensor_id=sensor.sensor_id,
        track_id=tgt.track_id,
        bearing_deg=brg,
        elevation_deg=elev,
        ground_range_m=gr,
        slant_range_m=sr,
        slew_time_s=slew,
        lead_time_s=t_lead if (lead and tgt.speed_mps > 0.0) else 0.0,
        aim_east=aim_e,
        aim_north=aim_n,
        in_coverage=covered,
        feasible=covered,
        advisories=tuple(advisories),
    )


def select_tracker(
    sensors: list[Sensor],
    target,
    *,
    from_bearings: dict | None = None,
    lead: bool = True,
) -> CueCommand | None:
    """Pick the sensor that can observe ``target`` now with the least slew.

    Considers only sensors whose current coverage annulus contains the track,
    cues each (optionally from its current bearing in ``from_bearings`` keyed by
    ``sensor_id``), and returns the :class:`CueCommand` with the smallest slew
    time. Ties break toward the earliest sensor in input order. Returns ``None``
    when no sensor currently covers the track.
    """
    tgt = as_target(target)
    best: CueCommand | None = None
    for s in sensors:
        if not in_coverage(s, tgt, 0.0):
            continue
        fb = 0.0 if from_bearings is None else float(from_bearings.get(s.sensor_id, 0.0))
        cmd = cue_sensor(s, tgt, from_bearing_deg=fb, lead=lead)
        if best is None or cmd.slew_time_s < best.slew_time_s - _EPS:
            best = cmd
    return best


# --------------------------------------------------------------------------- #
# Coverage windows along the track's path
# --------------------------------------------------------------------------- #

def _disk_interval(
    d_east: float, d_north: float, ve: float, vn: float, radius: float
) -> tuple[float, float] | None:
    """Real interval ``[s_lo, s_hi]`` of times a moving point is within ``radius``.

    Solves ``|d + v*s| <= radius``. Assumes ``|v| > 0`` (caller handles the
    stationary case). Returns ``None`` when the path never enters the disk.
    """
    a = ve * ve + vn * vn
    b = 2.0 * (d_east * ve + d_north * vn)
    c = d_east * d_east + d_north * d_north - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    s1 = (-b - sq) / (2.0 * a)
    s2 = (-b + sq) / (2.0 * a)
    return (s1, s2) if s1 <= s2 else (s2, s1)


def coverage_windows(sensor: Sensor, target) -> list[tuple[float, float]]:
    """Time windows (relative seconds, ``t >= 0``) the track is in coverage.

    Returns a list of ``(t_enter, t_exit)`` intervals during which the track's
    constant-velocity path lies inside the sensor's coverage annulus, clipped to
    ``t >= 0`` (now onward). A stationary track inside coverage yields a single
    open-ended window ``(0.0, inf)``. A track that crosses the near dead-zone
    produces two windows (approach, then recede). Windows are returned in
    ascending time order.
    """
    tgt = as_target(target)
    d_e = tgt.east - sensor.east
    d_n = tgt.north - sensor.north
    ve, vn = tgt.vel_east, tgt.vel_north

    if ve == 0.0 and vn == 0.0:
        gr = math.hypot(d_e, d_n)
        inside_outer = gr <= sensor.max_range_m + _EPS
        inside_deadzone = gr < sensor.min_range_m - _EPS
        return [(0.0, math.inf)] if (inside_outer and not inside_deadzone) else []

    outer = _disk_interval(d_e, d_n, ve, vn, sensor.max_range_m)
    if outer is None:
        return []
    o_lo = max(outer[0], 0.0)
    o_hi = outer[1]
    if o_hi <= o_lo + _EPS:
        return []

    windows: list[tuple[float, float]] = [(o_lo, o_hi)]

    if sensor.min_range_m > 0.0:
        inner = _disk_interval(d_e, d_n, ve, vn, sensor.min_range_m)
        if inner is not None:
            i_lo = max(inner[0], 0.0)
            i_hi = max(inner[1], 0.0)
            if i_hi > i_lo + _EPS:
                windows = _subtract_interval(o_lo, o_hi, i_lo, i_hi)

    return [(a, b) for (a, b) in windows if b > a + _EPS]


def _subtract_interval(
    lo: float, hi: float, a: float, b: float
) -> list[tuple[float, float]]:
    """Return ``[lo, hi]`` with the open sub-interval ``(a, b)`` removed."""
    parts: list[tuple[float, float]] = []
    if a > lo:
        parts.append((lo, min(a, hi)))
    if b < hi:
        parts.append((max(b, lo), hi))
    return parts


def current_window(sensor: Sensor, target) -> tuple[float, float] | None:
    """The coverage window active *now* (its ``t_enter`` is 0), or ``None``.

    Returns the window the track currently sits inside (enter time clamped to 0),
    or ``None`` when the track is not presently in this sensor's coverage.
    """
    for enter, exit_ in coverage_windows(sensor, target):
        if enter <= _EPS:
            return (enter, exit_)
    return None


def time_to_exit(sensor: Sensor, target) -> float | None:
    """Seconds until the track leaves the sensor's coverage, or ``None``.

    ``None`` means the track is not currently in coverage; ``inf`` means a
    stationary track that never leaves.
    """
    win = current_window(sensor, target)
    return None if win is None else win[1]


# --------------------------------------------------------------------------- #
# Pairwise handoff
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HandoffPlan:
    """A recommended track handoff from one sensor to another (awareness only).

    Attributes:
        track_id: The track being handed off.
        from_sensor: Sensor currently holding custody.
        to_sensor: Recommended successor (``None`` if none can continue custody).
        current_exit_t: When the current sensor loses the track (relative s).
        next_enter_t: When the successor first sees the track (relative s).
        next_exit_t: When the successor would in turn lose the track.
        overlap_s: Seconds both sensors can see the track (0 if none).
        gap_s: Seconds with no coverage between the two (0 if seamless).
        seamless: True when the successor covers the track before the current
            sensor loses it (``overlap_s >= min_overlap_s``).
        recommended_handoff_t: Suggested time to pass custody (relative s).
        successor_ready: Whether the successor can slew onto the track before it
            needs to observe (True when no bearings supplied).
        advisories: Plain-language notes.
    """

    track_id: object
    from_sensor: object
    to_sensor: object
    current_exit_t: float
    next_enter_t: float
    next_exit_t: float
    overlap_s: float
    gap_s: float
    seamless: bool
    recommended_handoff_t: float
    successor_ready: bool = True
    advisories: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "from_sensor": self.from_sensor,
            "to_sensor": self.to_sensor,
            "current_exit_t": _round_or_none(self.current_exit_t),
            "next_enter_t": _round_or_none(self.next_enter_t),
            "next_exit_t": _round_or_none(self.next_exit_t),
            "overlap_s": round(self.overlap_s, 6),
            "gap_s": _round_or_none(self.gap_s),
            "seamless": self.seamless,
            "recommended_handoff_t": _round_or_none(self.recommended_handoff_t),
            "successor_ready": self.successor_ready,
            "advisories": list(self.advisories),
        }


def _round_or_none(x: float, ndigits: int = 6):
    """Round a float, mapping non-finite sentinels to ``None`` for JSON."""
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return round(x, ndigits)


def plan_handoff(
    from_sensor: Sensor,
    candidates: list[Sensor],
    target,
    *,
    min_overlap_s: float = 0.0,
    from_bearings: dict | None = None,
) -> HandoffPlan:
    """Choose the successor sensor that best continues custody of ``target``.

    Predicts when ``from_sensor`` loses the track, then among ``candidates``
    (any sensor other than ``from_sensor``) finds the one whose coverage keeps
    the track observed with the smallest gap, preferring a seamless overlap and,
    among equals, the successor that holds custody longest. Ties break toward the
    earliest candidate in input order.

    If ``from_bearings`` is supplied (keyed by ``sensor_id``), the successor's
    slew readiness is checked: it must be able to slew onto the track before the
    track enters its coverage. Purely observational handoff.
    """
    tgt = as_target(target)
    cur = current_window(from_sensor, tgt)
    advisories: list[str] = []

    if cur is None:
        current_exit = 0.0
        advisories.append(
            "Track is not currently in the from-sensor's coverage; handoff is "
            "planned from now (t = 0)."
        )
    else:
        current_exit = cur[1]

    best: tuple | None = None  # (gap, -next_exit, order, sensor, enter, exit)
    for order, cand in enumerate(candidates):
        if cand.sensor_id == from_sensor.sensor_id:
            continue
        for enter, exit_ in coverage_windows(cand, tgt):
            if exit_ <= current_exit + _EPS:
                continue  # does not extend custody beyond the current sensor
            gap = max(0.0, enter - current_exit)
            key = (gap, -exit_, order)
            if best is None or key < best[0]:
                best = (key, cand, enter, exit_)
            break  # earliest qualifying window for this candidate

    if best is None:
        advisories.append(
            "No candidate sensor can continue custody of this track; it will be "
            "lost when the current sensor's coverage ends. Add sensor dwell to "
            "close the gap."
        )
        return HandoffPlan(
            track_id=tgt.track_id,
            from_sensor=from_sensor.sensor_id,
            to_sensor=None,
            current_exit_t=current_exit,
            next_enter_t=math.inf,
            next_exit_t=math.inf,
            overlap_s=0.0,
            gap_s=math.inf,
            seamless=False,
            recommended_handoff_t=current_exit,
            successor_ready=False,
            advisories=tuple(advisories),
        )

    _key, succ, next_enter, next_exit = best
    overlap = max(0.0, current_exit - next_enter)
    gap = max(0.0, next_enter - current_exit)
    seamless = overlap >= min_overlap_s and gap <= _EPS

    if gap > _EPS:
        recommended = current_exit
        advisories.append(
            f"Coverage gap of {gap:.2f} s between losing the track at t={current_exit:.2f} s "
            f"and the successor acquiring it at t={next_enter:.2f} s."
        )
    else:
        # Hand over inside the overlap, biased toward the middle for margin.
        lo = max(next_enter, 0.0)
        recommended = min(current_exit, (lo + current_exit) / 2.0)
        recommended = max(recommended, lo)

    successor_ready = True
    if from_bearings is not None:
        fb = float(from_bearings.get(succ.sensor_id, 0.0))
        acq_t = max(next_enter, 0.0)
        req_bearing = bearing_to(succ, tgt, acq_t)
        slew = slew_time_s(succ, fb, req_bearing)
        successor_ready = slew <= acq_t + _EPS
        if not successor_ready:
            advisories.append(
                f"Successor needs {slew:.2f} s to slew but the track enters its coverage in "
                f"{acq_t:.2f} s; pre-cue it earlier or accept a brief acquisition delay."
            )

    if seamless and not advisories:
        advisories.append(
            f"Seamless handoff: {overlap:.2f} s of overlapping coverage available."
        )

    return HandoffPlan(
        track_id=tgt.track_id,
        from_sensor=from_sensor.sensor_id,
        to_sensor=succ.sensor_id,
        current_exit_t=current_exit,
        next_enter_t=next_enter,
        next_exit_t=next_exit,
        overlap_s=overlap,
        gap_s=gap,
        seamless=seamless,
        recommended_handoff_t=recommended,
        successor_ready=successor_ready,
        advisories=tuple(advisories),
    )


# --------------------------------------------------------------------------- #
# Whole-path custody schedule
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CustodySegment:
    """One leg of a custody chain: a sensor holds the track over ``[start, end]``."""

    sensor_id: object
    start_t: float
    end_t: float

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "start_t": _round_or_none(self.start_t),
            "end_t": _round_or_none(self.end_t),
            "duration_s": _round_or_none(self.duration_s),
        }


@dataclass(frozen=True)
class CustodySchedule:
    """A hand-off chain keeping a track observed across many sensors.

    Attributes:
        track_id: The scheduled track.
        segments: Ordered custody legs (each a :class:`CustodySegment`).
        gaps: ``(gap_start_t, gap_end_t)`` intervals with no coverage between
            legs, if any.
        covered_s: Total time the track is under custody.
        gap_s: Total uncovered time within the observed span.
        num_handoffs: Number of sensor-to-sensor handoffs (``len(segments) - 1``,
            floored at 0).
        seamless: True when the track is covered continuously from first
            acquisition to final loss (no interior gaps).
        first_acquire_t / final_loss_t: Span endpoints (relative s).
        advisories: Plain-language notes.
    """

    track_id: object
    segments: tuple[CustodySegment, ...]
    gaps: tuple[tuple[float, float], ...]
    covered_s: float
    gap_s: float
    num_handoffs: int
    seamless: bool
    first_acquire_t: float
    final_loss_t: float
    advisories: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "segments": [s.to_dict() for s in self.segments],
            "gaps": [[_round_or_none(a), _round_or_none(b)] for a, b in self.gaps],
            "covered_s": _round_or_none(self.covered_s),
            "gap_s": round(self.gap_s, 6),
            "num_handoffs": self.num_handoffs,
            "seamless": self.seamless,
            "first_acquire_t": _round_or_none(self.first_acquire_t),
            "final_loss_t": _round_or_none(self.final_loss_t),
            "advisories": list(self.advisories),
        }


def schedule_custody(
    sensors: list[Sensor],
    target,
    *,
    horizon_s: float | None = None,
) -> CustodySchedule:
    """Stitch ``sensors`` into a custody chain that keeps ``target`` observed.

    Collects every sensor's coverage windows along the track's path, then applies
    the classic greedy interval-covering rule: starting at first acquisition,
    keep the current sensor until its coverage ends, then hand to the sensor
    whose coverage reaches furthest into the future. This minimises the number of
    handoffs while maximising covered time; genuine gaps (no sensor covers an
    interval) are reported, never hidden.

    ``horizon_s`` caps open-ended (stationary-track) windows and bounds the
    schedule; it is required when any window is unbounded, otherwise the schedule
    naturally ends when the track leaves all coverage.

    Deterministic; ties break toward the earliest sensor in input order.
    """
    tgt = as_target(target)

    # (enter, exit, order, sensor_id) for every window, capped by horizon.
    windows: list[tuple[float, float, int, object]] = []
    for order, s in enumerate(sensors):
        for enter, exit_ in coverage_windows(s, tgt):
            e = exit_
            if math.isinf(e):
                if horizon_s is None:
                    raise ValueError(
                        "an unbounded coverage window was found (stationary track "
                        "inside coverage); pass horizon_s to bound the schedule"
                    )
                e = horizon_s
            if horizon_s is not None:
                e = min(e, horizon_s)
                if enter > horizon_s:
                    continue
            if e > enter + _EPS:
                windows.append((enter, e, order, s.sensor_id))

    advisories: list[str] = []
    if not windows:
        advisories.append("No sensor covers this track's path; it cannot be observed.")
        return CustodySchedule(
            track_id=tgt.track_id,
            segments=(),
            gaps=(),
            covered_s=0.0,
            gap_s=0.0,
            num_handoffs=0,
            seamless=False,
            first_acquire_t=math.inf,
            final_loss_t=math.inf,
            advisories=tuple(advisories),
        )

    windows.sort(key=lambda w: (w[0], w[2]))
    first_acquire = windows[0][0]

    segments: list[CustodySegment] = []
    gaps: list[tuple[float, float]] = []
    covered = 0.0

    cursor = first_acquire
    while True:
        # Windows that cover the cursor instant: extend custody as far as possible.
        active = [w for w in windows if w[0] <= cursor + _EPS and w[1] > cursor + _EPS]
        if active:
            # Farthest-reaching window; earliest input order breaks ties.
            best = min(active, key=lambda w: (-w[1], w[2]))
            seg_end = best[1]
            if segments and segments[-1].sensor_id == best[3]:
                # Same sensor still best: extend its segment rather than re-log it.
                prev = segments[-1]
                segments[-1] = CustodySegment(prev.sensor_id, prev.start_t, seg_end)
            else:
                segments.append(CustodySegment(best[3], cursor, seg_end))
            covered += seg_end - cursor
            cursor = seg_end
        else:
            # Gap: jump to the next window that starts after the cursor.
            upcoming = [w for w in windows if w[0] > cursor + _EPS]
            if not upcoming:
                break
            nxt = min(upcoming, key=lambda w: (w[0], w[2]))
            gaps.append((cursor, nxt[0]))
            cursor = nxt[0]

    final_loss = segments[-1].end_t if segments else math.inf
    gap_total = math.fsum(b - a for a, b in gaps)
    seamless = not gaps
    num_handoffs = max(0, len(segments) - 1)

    if seamless:
        advisories.append(
            f"Continuous custody from t={first_acquire:.2f} s to t={final_loss:.2f} s "
            f"across {len(segments)} sensor(s) with {num_handoffs} handoff(s)."
        )
    else:
        advisories.append(
            f"{len(gaps)} coverage gap(s) totalling {gap_total:.2f} s; add sensor dwell "
            f"or reposition to close them."
        )

    return CustodySchedule(
        track_id=tgt.track_id,
        segments=tuple(segments),
        gaps=tuple(gaps),
        covered_s=covered,
        gap_s=gap_total,
        num_handoffs=num_handoffs,
        seamless=seamless,
        first_acquire_t=first_acquire,
        final_loss_t=final_loss,
        advisories=tuple(advisories),
    )
