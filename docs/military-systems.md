# Frontline military systems

Descriptive OSINT reference. Figures are *publicly reported* and vary across
sources; structured data with a source per row lives in
[`../data/military_drones.csv`](../data/military_drones.csv). No operational,
assembly, or targeting content.

## By role

- **One-way attack / loitering munitions.** Shahed-136/Geran-2 (the defining
  long-range OWA drone of the Russo-Ukrainian war, ~2,500 km class), its jet
  evolution Geran-3/Shahed-238 (higher speed compresses defender reaction time),
  ZALA Lancet (Russia's principal tactical loitering munition vs artillery/armor),
  and the US AeroVironment Switchblade 300 (anti-personnel) / 600 (anti-armor).
  Ukraine's indigenous deep-strike line: Liutyi (~2,000 km), the low-cost plywood
  AQ-400 Scythe, the turbojet Palianytsia, and Bober/UJ-26.
- **ISR & ISR-strike (MALE/HALE).** Orlan-10 (ubiquitous Russian tactical ISR/EW
  relay), Mohajer-6 (Iranian MALE), Bayraktar TB2 and the heavier Akinci (Turkey,
  wide export), and the Western/Chinese benchmarks MQ-9 Reaper, RQ-4 Global Hawk,
  IAI Heron, Wing Loong II, and CASC CH-4/CH-5.
- **Counter-UAS interceptors.** Wild Hornets STING — a civil-society-funded
  Ukrainian interceptor quadcopter produced at volume at low unit cost.
- **Uncrewed surface vessels (USVs).** Magura V5 (HUR) and Sea Baby (SBU) — the
  two USVs that reshaped Black Sea naval operations.

## Notes

- **Variant spread matters.** Reported range/warhead figures differ between
  manufacturers, defense trade press, and intelligence assessments — especially
  for OWA drones. The CSV gives a representative published value plus a source.
- **The "AI in the loop" trend.** Open teardowns (GUR War & Sanctions) document an
  upgraded Shahed-136 fitted with a camera and an NVIDIA Jetson module — the
  foreign-component/sanctions-evasion story tracked in the sibling
  `drone-warfare-osint` dataset.
- **Cost asymmetry over hit rate.** The loitering-munition model trades a low
  per-shot success rate for mass and cheap magazine depth (see the cited figures
  in `drone-warfare-osint/docs/STATISTICS.md`).

Full sources: [../SOURCES.md](../SOURCES.md).
