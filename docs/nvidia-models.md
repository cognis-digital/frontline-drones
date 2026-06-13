# NVIDIA open models on Hugging Face — reference index

A **descriptive** index of what NVIDIA publishes openly on Hugging Face, for
awareness/analysis. Structured data:
[`../data/nvidia_hf_models.csv`](../data/nvidia_hf_models.csv). This is a "what
exists" catalog — it is **not** a deployment, fine-tuning, or integration guide,
and nothing here concerns putting these models on drones or into any control,
navigation, or targeting pipeline.

## Categories

- **Robotics foundation (Isaac GR00T)** — cross-embodiment vision-language-action
  models for humanoids/robots. N1.5 is non-commercial; the N1.7 family is
  described as commercially licensed.
- **World foundation models (Cosmos)** — text/video world generation, prediction,
  and physical reasoning; mostly under the NVIDIA Open Model License and **gated**
  (accept terms on the model card before download).
- **Vision backbones (RADIO / C-RADIO)** — agglomerative ViT backbones distilling
  CLIP/DINOv2/SAM; "C-" variants are the commercially-licensed line.
- **Segmentation (SegFormer / MiT)** — first-party research checkpoints.
- **Speech (Canary, Parakeet)** — multilingual ASR / translation under CC-BY-4.0
  (open commercial); Audio-Flamingo is non-commercial.
- **LLMs & VLMs (Nemotron)** — base/reasoning LLMs, a vision-language 8B, OCR;
  NVIDIA Open Model License (Llama-derived ones also carry the Llama 3.1 license).
- **Embeddings (NV-Embed, multimodal embed)** — retrieval/rerank; NV-Embed is
  CC-BY-NC (non-commercial).

## License caution

License families seen: *NVIDIA Open Model License* (commercial w/ attribution,
not OSI-open), *Apache-2.0/commercial* (GR00T N1.7), *CC-BY-4.0* (open commercial;
Canary/Parakeet), and *CC-BY-NC / NVIDIA Source Code / OneWay Noncommercial*
(research-only; NV-Embed, Audio-Flamingo, FoundationPose). Cosmos repos are gated.
**Always re-check the `license:` tag and gating banner on the model card** — these
change. FoundationPose has no canonical first-party `nvidia/` HF repo (it ships via
NVlabs GitHub / NGC, with community weight mirrors on HF).

Full sources (Hugging Face URLs): [../SOURCES.md](../SOURCES.md).
