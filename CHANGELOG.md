# Changelog

## [0.3.0] — 2026-06-13
### Added
- `docs/counter-uas-selection.md` — a defensive **drone-detection sensor selection
  guide**: modality trade-offs (RF/radar/acoustic/EO-IR), the fiber-optic-drone blind
  spot for RF-only systems, scenario-based recommendations (fixed-site/mobile/covert/
  maritime), a selection checklist, and representative detection vendors. Detection/
  situational-awareness only — no mitigation/defeat procedures.

## [0.2.0] — 2026-06-13
### Added
- `install_models.py` — a customizable, stdlib multi-step menu that turns the NVIDIA
  HF index into working tooling: pick category → model → install / download / write a
  ready-to-run script. Generates real `transformers` run-snippets for runnable
  categories (segmentation, vision-language, LLM, embeddings, vision backbone) and
  downloads + points to the model card for special-runtime ones (NeMo ASR,
  Isaac/Cosmos). General ML tooling only — **no** drone control/targeting (DISCLAIMER).
- CI self-test (`install_models.py --selftest`) — stdlib, no downloads.

## [0.1.0] — 2026-06-13
Initial release.

### Added
- `data/military_drones.csv` — 24 frontline military UAVs / loitering munitions /
  USVs with published specs, operators, and a source per row.
- `data/commercial_platforms.csv` — 14 commercial drones + open autonomy stacks
  (DJI, Skydio, Autel, Parrot, senseFly; PX4, ArduPilot, MAVLink, QGroundControl,
  ROS 2, Gazebo, Pixhawk) with SDKs, platforms, and licenses.
- `data/nvidia_hf_models.csv` — 19 NVIDIA open Hugging Face models (GR00T, Cosmos,
  RADIO, SegFormer, Nemotron, Canary/Parakeet, NV-Embed) with modality + license + URL.
- Narrative docs (`docs/military-systems.md`, `docs/commercial-and-open.md`,
  `docs/nvidia-models.md`), `SOURCES.md`, `DISCLAIMER.md`.
- `scripts/validate.py` — stdlib-only catalog validator (schema, unique keys,
  source URLs); cross-OS CI on Linux + Windows × py3.10/3.13.
- MIT (code) + CC BY 4.0 (data) licensing for citability.

Descriptive reference only — no operational, guidance, or targeting content
(see DISCLAIMER.md).
