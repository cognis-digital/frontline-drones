"""Passive RF spectrum occupancy and FHSS burst analysis (awareness only).

A passive spectrum sensor watching the ISM bands sees a stream of short RF
*bursts*: each has a centre frequency, a start time, a dwell (duration) and an
occupied bandwidth. This module turns a window of such bursts into the summary
features a counter-UAS analyst reasons about - which bands are occupied, how many
distinct channels are in use, how fast the emitter hops, its mean dwell and duty
cycle, and a frequency-hopping-vs-fixed heuristic - and hands the band tokens
straight to :mod:`frontline_drones.rf_classify`.

Scope: passive, receive-side spectrum description only. Nothing here tunes,
transmits, jams or spoofs; it summarises already-observed energy. Pure stdlib,
deterministic. Frequencies are MHz, times seconds, dwell milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Canonical drone control/video ISM sub-bands, as (low_mhz, high_mhz, token).
# A burst is attributed to the first band whose range contains its centre freq.
BAND_PLAN: tuple[tuple[float, float, str], ...] = (
    (420.0, 450.0, "433 MHz"),
    (860.0, 930.0, "900 MHz"),
    (1150.0, 1350.0, "1.2 GHz"),
    (2400.0, 2500.0, "2.4 GHz"),
    (5150.0, 5350.0, "5.2 GHz"),
    (5650.0, 5925.0, "5.8 GHz"),
)


def band_of(center_freq_mhz: float) -> str | None:
    """Return the ISM band token containing ``center_freq_mhz``, or ``None``."""
    for lo, hi, token in BAND_PLAN:
        if lo <= center_freq_mhz <= hi:
            return token
    return None


@dataclass(frozen=True)
class Burst:
    """One observed RF burst.

    Attributes:
        center_freq_mhz: Burst centre frequency, MHz.
        start_s: Burst start time within the window, seconds.
        duration_ms: Dwell time on this frequency, milliseconds.
        bandwidth_mhz: Occupied bandwidth, MHz (optional).
        power_dbm: Received power, dBm (optional).
    """

    center_freq_mhz: float
    start_s: float
    duration_ms: float
    bandwidth_mhz: float | None = None
    power_dbm: float | None = None

    @property
    def band(self) -> str | None:
        return band_of(self.center_freq_mhz)


def _as_burst(item) -> Burst:
    """Coerce a caller-supplied burst into a :class:`Burst`."""
    if isinstance(item, Burst):
        return item
    if isinstance(item, dict):
        return Burst(
            center_freq_mhz=float(item.get("center_freq_mhz", item.get("freq_mhz"))),
            start_s=float(item.get("start_s", 0.0)),
            duration_ms=float(item.get("duration_ms", 0.0)),
            bandwidth_mhz=(
                None if item.get("bandwidth_mhz") is None
                else float(item["bandwidth_mhz"])
            ),
            power_dbm=(None if item.get("power_dbm") is None else float(item["power_dbm"])),
        )
    seq = list(item)
    freq, start, dur = float(seq[0]), float(seq[1]), float(seq[2])
    bw = float(seq[3]) if len(seq) > 3 and seq[3] is not None else None
    pw = float(seq[4]) if len(seq) > 4 and seq[4] is not None else None
    return Burst(center_freq_mhz=freq, start_s=start, duration_ms=dur,
                 bandwidth_mhz=bw, power_dbm=pw)


def _channelize(freq_mhz: float, channel_khz: float) -> int:
    """Quantise a frequency into an integer channel index of width ``channel_khz``."""
    return int(round(freq_mhz * 1000.0 / channel_khz))


@dataclass(frozen=True)
class SpectrumSummary:
    """Occupancy and hopping summary over a window of RF bursts.

    Attributes:
        num_bursts: Bursts analysed.
        window_s: Time span from first burst start to last burst end.
        bands: Sorted ISM band tokens the bursts occupied.
        distinct_channels: Number of distinct channels used (at the analysis
            channel granularity).
        freq_span_mhz: Max minus min centre frequency across bursts.
        mean_dwell_ms / min_dwell_ms / max_dwell_ms: Dwell-time statistics.
        hop_rate_hz: Distinct-channel changes per second across the window.
        duty_cycle: Fraction of the window occupied by burst energy, [0, 1].
        hopping: ``"fhss"``, ``"fixed"`` or ``"unknown"`` heuristic label.
        hopping_confidence: Confidence in the ``hopping`` label, [0, 1].
        classify_bands: Band tokens ready to feed rf_classify's ``bands``.
    """

    num_bursts: int
    window_s: float
    bands: tuple[str, ...]
    distinct_channels: int
    freq_span_mhz: float
    mean_dwell_ms: float
    min_dwell_ms: float
    max_dwell_ms: float
    hop_rate_hz: float
    duty_cycle: float
    hopping: str
    hopping_confidence: float
    classify_bands: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "num_bursts": self.num_bursts,
            "window_s": round(self.window_s, 4),
            "bands": list(self.bands),
            "distinct_channels": self.distinct_channels,
            "freq_span_mhz": round(self.freq_span_mhz, 4),
            "mean_dwell_ms": round(self.mean_dwell_ms, 4),
            "min_dwell_ms": round(self.min_dwell_ms, 4),
            "max_dwell_ms": round(self.max_dwell_ms, 4),
            "hop_rate_hz": round(self.hop_rate_hz, 4),
            "duty_cycle": round(self.duty_cycle, 4),
            "hopping": self.hopping,
            "hopping_confidence": round(self.hopping_confidence, 4),
            "classify_bands": list(self.classify_bands),
        }


def _classify_hopping(
    distinct_channels: int,
    freq_span_mhz: float,
    mean_dwell_ms: float,
    hop_rate_hz: float,
) -> tuple[str, float]:
    """Heuristic FHSS-vs-fixed label with a confidence in [0, 1].

    FHSS control links visit many channels across a wide span with short dwell and
    a high hop rate; a fixed link parks on one channel. The confidence blends how
    strongly each of those independent cues points the same way.
    """
    if distinct_channels <= 1:
        # One channel only: a fixed link (confidence grows with observed dwell).
        conf = 0.6 if mean_dwell_ms > 0 else 0.4
        return "fixed", conf

    # Independent FHSS cues, each in [0, 1].
    chan_cue = min(1.0, (distinct_channels - 1) / 9.0)   # saturates ~10 channels
    span_cue = min(1.0, freq_span_mhz / 60.0)            # saturates ~60 MHz spread
    dwell_cue = 1.0 if mean_dwell_ms <= 0 else min(1.0, 20.0 / mean_dwell_ms)
    hop_cue = min(1.0, hop_rate_hz / 50.0)               # saturates ~50 hops/s
    fhss_score = (chan_cue + span_cue + dwell_cue + hop_cue) / 4.0

    if fhss_score >= 0.5:
        return "fhss", fhss_score
    if fhss_score <= 0.25:
        return "fixed", 1.0 - fhss_score
    return "unknown", 1.0 - abs(fhss_score - 0.375) * 2.0


def analyze_spectrum(bursts, *, channel_khz: float = 1000.0) -> SpectrumSummary:
    """Summarise a window of passive RF bursts into occupancy + hopping features.

    ``bursts`` is any iterable of :class:`Burst`, ``(freq_mhz, start_s,
    duration_ms[, bandwidth_mhz[, power_dbm]])`` tuples, or dicts. ``channel_khz``
    sets the channelisation granularity used to count distinct channels and hops.
    Returns a deterministic :class:`SpectrumSummary`; an empty input yields a
    zeroed summary labelled ``"unknown"``.
    """
    items = sorted((_as_burst(b) for b in bursts), key=lambda b: (b.start_s, b.center_freq_mhz))
    if not items:
        return SpectrumSummary(
            num_bursts=0, window_s=0.0, bands=(), distinct_channels=0,
            freq_span_mhz=0.0, mean_dwell_ms=0.0, min_dwell_ms=0.0, max_dwell_ms=0.0,
            hop_rate_hz=0.0, duty_cycle=0.0, hopping="unknown", hopping_confidence=0.0,
            classify_bands=(),
        )
    if channel_khz <= 0:
        raise ValueError("channel_khz must be positive")

    freqs = [b.center_freq_mhz for b in items]
    dwells = [b.duration_ms for b in items]
    channels = [_channelize(f, channel_khz) for f in freqs]
    distinct_channels = len(set(channels))
    freq_span = max(freqs) - min(freqs)

    start = items[0].start_s
    end = max(b.start_s + b.duration_ms / 1000.0 for b in items)
    window_s = max(0.0, end - start)

    # Hop rate: count channel *changes* in time order, per second of window.
    changes = sum(1 for a, b in zip(channels, channels[1:], strict=False) if a != b)
    hop_rate = changes / window_s if window_s > 0 else 0.0

    total_on = sum(d / 1000.0 for d in dwells)
    duty = min(1.0, total_on / window_s) if window_s > 0 else 0.0

    mean_dwell = sum(dwells) / len(dwells)
    bands = tuple(sorted({b.band for b in items if b.band is not None}))

    hopping, hop_conf = _classify_hopping(distinct_channels, freq_span, mean_dwell, hop_rate)

    return SpectrumSummary(
        num_bursts=len(items),
        window_s=window_s,
        bands=bands,
        distinct_channels=distinct_channels,
        freq_span_mhz=freq_span,
        mean_dwell_ms=mean_dwell,
        min_dwell_ms=min(dwells),
        max_dwell_ms=max(dwells),
        hop_rate_hz=hop_rate,
        duty_cycle=duty,
        hopping=hopping,
        hopping_confidence=hop_conf,
        classify_bands=bands,
    )


def channel_histogram(bursts, *, channel_khz: float = 1000.0) -> dict[str, int]:
    """Return a {``"<freq_mhz> MHz"``: burst_count} histogram over channels.

    Handy for spotting which channels an emitter dwells on. Keys are the channel
    centre frequency rendered in MHz; deterministic ordering is the caller's to
    impose (dict preserves first-seen insertion order here, sorted by frequency).
    """
    items = [_as_burst(b) for b in bursts]
    counts: dict[int, int] = {}
    for b in items:
        ch = _channelize(b.center_freq_mhz, channel_khz)
        counts[ch] = counts.get(ch, 0) + 1
    out: dict[str, int] = {}
    for ch in sorted(counts):
        mhz = ch * channel_khz / 1000.0
        out[f"{mhz:g} MHz"] = counts[ch]
    return out
