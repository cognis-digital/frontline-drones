# frontline-drones

> A **descriptive, citation-grade catalog** of the drone landscape: frontline
> military systems, popular commercial platforms and the open autonomy
> ecosystem, plus a reference index of NVIDIA's openly-published robotics /
> perception models on Hugging Face.

[![Data License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Code License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)

This is a **reference**. It documents *what publicly exists*, with sources. It
contains **no** operational, assembly, flight-control, guidance, or targeting
instructions, and nothing about modifying any platform for weapons. See
[DISCLAIMER.md](DISCLAIMER.md).

## What's inside

| File | Rows | Contents |
|---|---|---|
| [`data/military_drones.csv`](data/military_drones.csv) | 24 | Frontline military UAVs, loitering munitions, USVs — published specs + operators + source |
| [`data/commercial_platforms.csv`](data/commercial_platforms.csv) | 14 | Commercial drones (DJI, Skydio, Autel, Parrot, senseFly) + open stacks (PX4, ArduPilot, MAVLink, ROS 2, QGroundControl, Pixhawk) |
| [`data/nvidia_hf_models.csv`](data/nvidia_hf_models.csv) | 19 | NVIDIA's open Hugging Face models (Isaac GR00T, Cosmos, RADIO, SegFormer, Nemotron, Canary/Parakeet, NV-Embed) |

Human-readable write-ups:
- [`docs/military-systems.md`](docs/military-systems.md)
- [`docs/commercial-and-open.md`](docs/commercial-and-open.md)
- [`docs/nvidia-models.md`](docs/nvidia-models.md)
- [`docs/counter-uas-selection.md`](docs/counter-uas-selection.md) — defensive drone-detection sensor selection guide
- [`SOURCES.md`](SOURCES.md) · [`CHANGELOG.md`](CHANGELOG.md)

## Validate

```bash
python scripts/validate.py     # stdlib only; checks schema, keys, and source URLs
```

## Install & run the NVIDIA models

The NVIDIA HF index isn't just a list — `install_models.py` is a customizable,
multi-step menu that downloads and runs a catalogued model **for its documented
purpose** (segmentation, ASR, vision-language, embeddings, LLM, vision backbone):

```bash
python install_models.py        # menu: pick category -> model -> install / download / write a run-script
```

It generates real, runnable `transformers` snippets where the model runs there; for
models that need a special runtime (NeMo for ASR, the Isaac/Cosmos toolkits for
GR00T/Cosmos) it downloads the weights and points you to the model card rather than
faking a snippet.

> Scope: this is general-purpose ML tooling. It does **not** wire any model into
> drone control, navigation, or targeting — see [DISCLAIMER.md](DISCLAIMER.md).

## Sourcing & honesty

Specs are *publicly reported* figures and vary between sources (especially
range/warhead for one-way-attack drones); representative published values are
given with a primary source per row. The NVIDIA index lists model ids, purpose,
modality, license, and the Hugging Face URL — confirm license/gating on each
model card, as NVIDIA revises terms frequently. This catalog was compiled from
the sources in [SOURCES.md](SOURCES.md); verify against the primary pages before
citing.

## Related Cognis tools — defense / drone / maritime OSINT cluster

- [`awesome-drone-warfare-osint`](https://github.com/cognis-digital/awesome-drone-warfare-osint) — citation-grade dataset (8,300+ components / 195+ platforms) + compliance query CLI
- [`uaslog`](https://github.com/cognis-digital/uaslog) — counter-UAS telemetry/log analyzer (detection events, RF bands, tracks)
- [`maritimeint`](https://github.com/cognis-digital/maritimeint) — AIS grey/dark-fleet detection: watchlist + sanctions cross-reference
- [`geoaoi-pro`](https://github.com/cognis-digital/geoaoi-pro) — MIL-STD-2525 symbology + AOI helpers · [`geolens`](https://github.com/cognis-digital/geolens) — image geolocation
- [`stixgen`](https://github.com/cognis-digital/stixgen) · [`attackmap`](https://github.com/cognis-digital/attackmap) · [`ttphunt`](https://github.com/cognis-digital/ttphunt) — threat-intel tooling
- [`edgemesh`](https://github.com/cognis-digital/edgemesh) — run any model privately across your own hardware

**300+ open security & OSINT tools →** [github.com/cognis-digital](https://github.com/cognis-digital)

## License

Code: MIT ([LICENSE](LICENSE)). Data & docs: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)).

---
📡 **[Interop map](INTEROP.md)** — how this repo composes with the rest of the Cognis suite (private-AI backbone, agent language + cognition, domain intelligence).
