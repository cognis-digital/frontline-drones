"""Cross-modality signature lookup over the detection reference datasets.

Joins the three passive-detection signature tables - RF control-link
(``rf``), acoustic BPF (``acoustic``) and micro-Doppler radar (``radar``) - by
their shared ``platform`` token into a single analyst quick-reference card:
"what would each sensor see for this airframe?" Read-only lookup; reuses the
existing catalog loader, so it stays faithful to the shipped CSV data.

Detection-side reference only - no transmit, jam or engagement content.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Row, load_dataset

# The signature datasets that share a ``platform`` join column.
SIGNATURE_DATASETS: tuple[str, ...] = ("rf", "acoustic", "radar")


def _index_by_platform(dataset: str, *, data_dir: str | None = None) -> dict[str, Row]:
    """Index one signature dataset by its ``platform`` column (last row wins)."""
    idx: dict[str, Row] = {}
    for row in load_dataset(dataset, data_dir=data_dir):
        key = (row.get("platform") or row.get("id") or "").strip().lower()
        if key:
            idx[key] = row
    return idx


def known_platforms(*, data_dir: str | None = None) -> list[str]:
    """Return the sorted union of platform tokens across the signature tables."""
    platforms: set[str] = set()
    for ds in SIGNATURE_DATASETS:
        platforms.update(_index_by_platform(ds, data_dir=data_dir))
    return sorted(platforms)


def resolve_platform(platform: str, *, data_dir: str | None = None) -> str:
    """Resolve a platform token by exact or unique-substring match.

    Raises:
        KeyError: if nothing matches, or a substring is ambiguous.
    """
    key = platform.strip().lower()
    known = known_platforms(data_dir=data_dir)
    if key in known:
        return key
    matches = [p for p in known if key in p]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(
            f"unknown platform {platform!r}; known platforms: {', '.join(known)}"
        )
    raise KeyError(
        f"ambiguous platform {platform!r}; matches: {', '.join(matches)}"
    )


@dataclass(frozen=True)
class SignatureCard:
    """What each passive sensor would observe for one airframe.

    Attributes:
        platform: Resolved platform token.
        name: Display name (from whichever table supplies it).
        rf / acoustic / radar: The matching row from each dataset, or ``None``.
        citations: Sorted unique source URLs across the joined rows.
    """

    platform: str
    name: str
    rf: Row | None = None
    acoustic: Row | None = None
    radar: Row | None = None
    citations: tuple[str, ...] = field(default_factory=tuple)

    @property
    def modalities_present(self) -> tuple[str, ...]:
        """Which signature modalities have data for this platform."""
        present = []
        for mod in ("rf", "acoustic", "radar"):
            if getattr(self, mod) is not None:
                present.append(mod)
        return tuple(present)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the card."""
        return {
            "platform": self.platform,
            "name": self.name,
            "modalities_present": list(self.modalities_present),
            "rf": self.rf,
            "acoustic": self.acoustic,
            "radar": self.radar,
            "citations": list(self.citations),
        }


def signature_card(platform: str, *, data_dir: str | None = None) -> SignatureCard:
    """Build a cross-modality :class:`SignatureCard` for ``platform``.

    Joins the RF, acoustic and radar signature rows sharing the resolved
    platform token. Missing modalities are ``None`` (e.g. an autonomous or
    fiber-optic airframe has no RF control-link row) - itself a useful
    detection cue.
    """
    key = resolve_platform(platform, data_dir=data_dir)
    rows: dict[str, Row | None] = {}
    for ds in SIGNATURE_DATASETS:
        rows[ds] = _index_by_platform(ds, data_dir=data_dir).get(key)

    name = ""
    citations: set[str] = set()
    for ds in SIGNATURE_DATASETS:
        row = rows[ds]
        if row is None:
            continue
        if not name and row.get("name"):
            name = row["name"]
        url = row.get("source_url") or row.get("url")
        if url:
            citations.add(url)

    return SignatureCard(
        platform=key,
        name=name or key,
        rf=rows["rf"],
        acoustic=rows["acoustic"],
        radar=rows["radar"],
        citations=tuple(sorted(citations)),
    )


def render_card_text(card: SignatureCard) -> str:
    """Render a :class:`SignatureCard` as a human-readable quick-reference block."""
    lines = [f"Signature card: {card.name}  [{card.platform}]", "=" * 60]

    if card.rf is not None:
        rf = card.rf
        lines += [
            "RF control-link:",
            f"  band:      {rf.get('control_band', '')}",
            f"  protocol:  {rf.get('video_protocol', '')}",
            f"  hopping:   {rf.get('frequency_hopping', '')}",
            f"  note:      {rf.get('detection_notes', '')}",
        ]
    else:
        lines.append("RF control-link:  (none - no exploitable control-link RF)")

    if card.acoustic is not None:
        ac = card.acoustic
        lines += [
            "Acoustic (BPF):",
            f"  fundamental: {ac.get('bpf_fundamental_hz', '')} Hz",
            f"  harmonics:   {ac.get('harmonic_range_hz', '')} Hz",
            f"  range:       {ac.get('detect_range_m', '')} m",
        ]
    else:
        lines.append("Acoustic (BPF):  (no reference row)")

    if card.radar is not None:
        rd = card.radar
        lines += [
            "Radar (micro-Doppler):",
            f"  band:         {rd.get('recommended_band', '')}",
            f"  spread:       {rd.get('micro_doppler_spread', '')}",
            f"  bird cue:     {rd.get('bird_discriminant', '')}",
        ]
    else:
        lines.append("Radar (micro-Doppler):  (no reference row)")

    if card.citations:
        lines.append("Citations:")
        lines += [f"  - {c}" for c in card.citations]
    return "\n".join(lines)
