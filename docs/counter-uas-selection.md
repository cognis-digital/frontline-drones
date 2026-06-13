# Counter-UAS detection: a selection guide (defensive)

> Descriptive buyer's guide for choosing **drone-detection** sensors by scenario.
> Detection/situational-awareness only — no mitigation/defeat/engagement procedures.
> Pairs with the cited analysis in
> [`awesome-drone-warfare-osint/docs/counter-uas-detection.md`](https://github.com/cognis-digital/awesome-drone-warfare-osint/blob/main/docs/counter-uas-detection.md).

## First principle: no single sensor is enough → layer for fusion

Each modality has a blind spot, so credible detection fuses **≥2**:

| Modality | Best at | Blind spot |
|---|---|---|
| **RF / spectrum** (passive) | Cheap, classifies type by protocol; detects controller too | **Blind to autonomous & fiber-optic drones** (no RF link) |
| **Radar** (micro-Doppler) | Physics-based; sees silent/fiber drones; longest range | Small-RCS targets in urban/ground clutter |
| **Acoustic** (passive) | Fully passive; senses fiber/autonomous drones short-range | ~300–500 m; degraded by wind/noise |
| **EO/IR** | Visual/thermal ID + forensic record | Lighting/weather; narrow FoV (needs cueing) |

> **The 2024–26 lesson:** RF-only systems are increasingly defeated by **fiber-optic-
> controlled drones** (zero RF). If your threat includes those, you need **radar +/or
> acoustic**, not RF alone.

## Choose by scenario

| Scenario | Recommended core | Why |
|---|---|---|
| **Fixed critical infrastructure** (airport, refinery, base) | Radar (micro-Doppler) + RF + EO/IR cueing | Long-range volume coverage + ID + forensic record |
| **Mobile / field / convoy** | Compact radar + RF, optional acoustic | Quick-deploy, covers silent drones on the move |
| **Covert / dismounted / portable** | RF detector + acoustic | Passive (no emissions), light, short-range early warning |
| **Maritime / port** | Radar + EO/IR (+ RF) | Clutter-tolerant volume search over water |
| **Fiber-drone threat present** | Radar + acoustic **mandatory** | RF cannot see fiber/autonomous drones |

## Selection checklist

1. **Threat model first** — consumer DJI? FPV? fiber-optic? autonomous waypoint? That
   determines whether RF-only is viable (usually it isn't anymore).
2. **Detection ≠ mitigation** — detection is broadly deployable; *mitigation* is heavily
   legally restricted. Get legal review before anything that intercepts comms
   ([CISA guidance](https://www.cisa.gov/topics/physical-security/be-air-aware/protect-critical-infrastructure-and-public-gatherings)).
3. **Demand fusion + a track-quality metric**, not single-sensor alarms (false-alarm rate
   matters as much as detection range).
4. **Identification path** — can it read Remote ID / DroneID, and does it treat Remote ID
   as an *aid* (spoofable), not trusted authentication?
5. **Standards** — FAA Remote ID / ASTM F3411 awareness; NIST/ASTM C-UAS test methods for
   apples-to-apples vendor comparison.

## Representative detection vendors (publicly documented)

Radar: **Robin Radar (IRIS)**, **Echodyne (EchoShield)**. Multi-sensor fusion:
**Dedrone**, **DroneShield**. (Listed descriptively as market reference; not an
endorsement. Verify current specs with the vendor.)

*Sources: see the cited detection reference in the OSINT repo (CSIS, RUSI, EU JRC, FAA,
NDSS DroneID, vendor docs).*
