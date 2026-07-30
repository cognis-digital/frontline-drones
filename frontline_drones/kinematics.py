"""Track kinematics and track-quality scoring (situational-awareness only).

Turns a sequence of timestamped ENU position fixes (from a radar, EO/IR tracker
or fused tracker) into the kinematic features counter-UAS analysts reason about:
speed, heading, climb rate, turn rate, path straightness and an overall
**track-quality** metric in [0, 1]. Track quality is the number
:mod:`frontline_drones.fusion` multiplies into its fused confidence so a jittery,
sparse or physically implausible track cannot present as a high-confidence
detection.

It also produces a coarse *motion pattern* label (hover / transit / loiter /
erratic) that aids classification - a hovering small multirotor and a fast
straight-line transit are very different detection pictures.

Scope: descriptive kinematics only. There is no prediction-for-intercept,
guidance or engagement logic - the constant-velocity projection here exists only
to score track smoothness and support awareness CPA math. Pure stdlib,
deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A "hover" is any track whose mean speed stays under this (m/s): a station-
# keeping small multirotor. Above the transit threshold it is treated as a
# purposeful transit.
HOVER_SPEED_MPS = 1.5
TRANSIT_SPEED_MPS = 8.0

# Motion-pattern labels, coarsest to most manoeuvring.
MOTION_PATTERNS: tuple[str, ...] = ("hover", "loiter", "transit", "erratic")


@dataclass(frozen=True)
class Fix:
    """One timestamped ENU position fix.

    Attributes:
        t: Timestamp in seconds (monotonic within a track).
        east / north / up: Local ENU position, metres.
    """

    t: float
    east: float
    north: float
    up: float = 0.0


def _as_fix(sample: Fix | tuple | list | dict) -> Fix:
    """Coerce a caller-supplied sample into a :class:`Fix`."""
    if isinstance(sample, Fix):
        return sample
    if isinstance(sample, dict):
        return Fix(
            t=float(sample["t"]),
            east=float(sample.get("east", sample.get("x", 0.0))),
            north=float(sample.get("north", sample.get("y", 0.0))),
            up=float(sample.get("up", sample.get("z", 0.0))),
        )
    seq = list(sample)
    t, east, north = float(seq[0]), float(seq[1]), float(seq[2])
    up = float(seq[3]) if len(seq) > 3 else 0.0
    return Fix(t=t, east=east, north=north, up=up)


def _heading_deg(de: float, dn: float) -> float:
    """Compass heading (deg, 0=N, clockwise) of an East/North displacement."""
    return (math.degrees(math.atan2(de, dn)) + 360.0) % 360.0


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two headings, degrees in [0, 180]."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


@dataclass(frozen=True)
class TrackFeatures:
    """Kinematic features and quality score for a track.

    Attributes:
        num_fixes: Number of position fixes used.
        duration_s: Elapsed time across the track.
        path_length_m: Total distance travelled along the path.
        displacement_m: Straight-line start-to-end distance.
        straightness: displacement / path_length in [0, 1] (1 = perfectly straight).
        mean_speed_mps / max_speed_mps: Speed statistics.
        mean_climb_mps: Mean vertical rate (positive = climbing).
        mean_turn_rate_dps: Mean heading change per second (deg/s).
        heading_deg: Overall start-to-end heading.
        motion_pattern: One of :data:`MOTION_PATTERNS`.
        track_quality: Overall [0, 1] quality metric for fusion.
    """

    num_fixes: int
    duration_s: float
    path_length_m: float
    displacement_m: float
    straightness: float
    mean_speed_mps: float
    max_speed_mps: float
    mean_climb_mps: float
    mean_turn_rate_dps: float
    heading_deg: float
    motion_pattern: str
    track_quality: float

    def to_dict(self) -> dict:
        return {
            "num_fixes": self.num_fixes,
            "duration_s": round(self.duration_s, 3),
            "path_length_m": round(self.path_length_m, 3),
            "displacement_m": round(self.displacement_m, 3),
            "straightness": round(self.straightness, 4),
            "mean_speed_mps": round(self.mean_speed_mps, 3),
            "max_speed_mps": round(self.max_speed_mps, 3),
            "mean_climb_mps": round(self.mean_climb_mps, 3),
            "mean_turn_rate_dps": round(self.mean_turn_rate_dps, 3),
            "heading_deg": round(self.heading_deg, 3),
            "motion_pattern": self.motion_pattern,
            "track_quality": round(self.track_quality, 4),
        }


def _classify_motion(mean_speed: float, straightness: float, turn_rate: float) -> str:
    """Label the coarse motion pattern from summary kinematics."""
    if mean_speed < HOVER_SPEED_MPS:
        return "hover"
    if turn_rate > 25.0 and straightness < 0.5:
        return "erratic"
    if mean_speed >= TRANSIT_SPEED_MPS and straightness >= 0.7:
        return "transit"
    if straightness < 0.55:
        return "loiter"
    return "transit"


def _quality(
    num_fixes: int,
    duration_s: float,
    straightness: float,
    max_speed: float,
    turn_rate: float,
    gap_ratio: float,
) -> float:
    """Combine track evidence into a [0, 1] quality score.

    Rewards more fixes, longer coverage and smooth motion; penalises large
    sampling gaps, physically implausible speeds (> ~90 m/s for a small UAS) and
    very high turn rates that usually indicate association noise rather than a
    real manoeuvre.
    """
    if num_fixes < 2 or duration_s <= 0.0:
        return 0.0
    # Support: saturates around 8 fixes.
    support = min(1.0, (num_fixes - 1) / 7.0)
    # Duration: saturates around 6 s of continuous track.
    dur = min(1.0, duration_s / 6.0)
    # Smoothness: penalise heavy turning.
    smooth = max(0.0, 1.0 - turn_rate / 90.0)
    # Regular sampling: penalise big gaps relative to the mean interval.
    regular = max(0.0, 1.0 - gap_ratio)
    # Plausibility: a small UAS over ~90 m/s is suspect.
    plausible = 1.0 if max_speed <= 90.0 else max(0.0, 1.0 - (max_speed - 90.0) / 90.0)
    # Straightness contributes mildly (a clean hover is fine, so floor it).
    straight = 0.5 + 0.5 * straightness
    score = support * dur * (0.25 + 0.75 * smooth) * (0.4 + 0.6 * regular)
    score *= plausible * straight
    return max(0.0, min(1.0, score))


def track_features(samples) -> TrackFeatures:
    """Compute :class:`TrackFeatures` from an ordered sequence of position fixes.

    ``samples`` is any iterable of :class:`Fix`, ``(t, east, north[, up])``
    tuples, or ``{"t","east","north","up"}`` dicts. Fixes are sorted by time.
    A track with fewer than two usable fixes yields a zero-quality result.
    """
    fixes = sorted((_as_fix(s) for s in samples), key=lambda f: f.t)
    n = len(fixes)
    if n < 2:
        return TrackFeatures(
            num_fixes=n,
            duration_s=0.0,
            path_length_m=0.0,
            displacement_m=0.0,
            straightness=0.0,
            mean_speed_mps=0.0,
            max_speed_mps=0.0,
            mean_climb_mps=0.0,
            mean_turn_rate_dps=0.0,
            heading_deg=0.0,
            motion_pattern="hover",
            track_quality=0.0,
        )

    path_len = 0.0
    max_speed = 0.0
    speeds: list[float] = []
    headings: list[float] = []
    intervals: list[float] = []
    total_climb = 0.0

    for a, b in zip(fixes, fixes[1:], strict=False):
        dt = b.t - a.t
        de, dn, du = b.east - a.east, b.north - a.north, b.up - a.up
        seg = math.sqrt(de * de + dn * dn + du * du)
        path_len += seg
        total_climb += du
        if dt > 0:
            intervals.append(dt)
            speed = seg / dt
            speeds.append(speed)
            max_speed = max(max_speed, speed)
            if math.hypot(de, dn) > 1e-9:
                headings.append(_heading_deg(de, dn))

    duration = fixes[-1].t - fixes[0].t
    disp = math.sqrt(
        (fixes[-1].east - fixes[0].east) ** 2
        + (fixes[-1].north - fixes[0].north) ** 2
        + (fixes[-1].up - fixes[0].up) ** 2
    )
    straightness = disp / path_len if path_len > 0 else 0.0
    mean_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
    mean_climb = (total_climb / duration) if duration > 0 else 0.0

    # Mean absolute turn rate across successive headings.
    turn_rates: list[float] = []
    for h1, h2 in zip(headings, headings[1:], strict=False):
        turn_rates.append(_angle_diff(h1, h2))
    # Normalise turn by the mean interval to get deg/s.
    mean_interval = (sum(intervals) / len(intervals)) if intervals else 0.0
    mean_turn_rate = (
        (sum(turn_rates) / len(turn_rates)) / mean_interval
        if turn_rates and mean_interval > 0
        else 0.0
    )

    overall_heading = _heading_deg(
        fixes[-1].east - fixes[0].east, fixes[-1].north - fixes[0].north
    )

    # Sampling-gap ratio: how uneven the intervals are (0 = perfectly regular).
    gap_ratio = 0.0
    if len(intervals) >= 2 and mean_interval > 0:
        max_gap = max(intervals)
        gap_ratio = min(1.0, (max_gap - mean_interval) / (mean_interval * 3.0))

    pattern = _classify_motion(mean_speed, straightness, mean_turn_rate)
    quality = _quality(n, duration, straightness, max_speed, mean_turn_rate, gap_ratio)

    return TrackFeatures(
        num_fixes=n,
        duration_s=duration,
        path_length_m=path_len,
        displacement_m=disp,
        straightness=straightness,
        mean_speed_mps=mean_speed,
        max_speed_mps=max_speed,
        mean_climb_mps=mean_climb,
        mean_turn_rate_dps=mean_turn_rate,
        heading_deg=overall_heading,
        motion_pattern=pattern,
        track_quality=quality,
    )


def velocity_enu(samples) -> tuple[float, float, float]:
    """Estimate a current ENU velocity vector (m/s) from the last two fixes.

    Useful for feeding :func:`frontline_drones.geo.closest_point_of_approach`.
    Returns ``(0, 0, 0)`` when fewer than two fixes or a zero time step.
    """
    fixes = sorted((_as_fix(s) for s in samples), key=lambda f: f.t)
    if len(fixes) < 2:
        return (0.0, 0.0, 0.0)
    a, b = fixes[-2], fixes[-1]
    dt = b.t - a.t
    if dt <= 0:
        return (0.0, 0.0, 0.0)
    return ((b.east - a.east) / dt, (b.north - a.north) / dt, (b.up - a.up) / dt)
