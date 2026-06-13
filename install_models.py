#!/usr/bin/env python3
"""Install & run the catalogued NVIDIA Hugging Face models — a customizable,
multi-step menu.

Scope: this turns the descriptive `data/nvidia_hf_models.csv` index into working
tooling that downloads a model and runs it **for its documented purpose**
(segmentation, ASR, vision-language, embeddings, LLM, vision backbone). It is
general-purpose ML tooling. It does **not** integrate any model into drone
control, navigation, or targeting — see DISCLAIMER.md.

The menu (stdlib only) walks: pick a category -> pick a model -> choose an action
(show install steps / download / write a ready-to-run script). The generated
run-scripts use `transformers` where the model runs there; models needing a
special runtime (NeMo for ASR, Isaac/Cosmos toolkits for GR00T/Cosmos) get the
download + a pointer to the model card's instructions, rather than a fake snippet.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "data", "nvidia_hf_models.csv")

# pip deps + a runnable snippet per modality/category, for the model's real purpose.
RUNNERS: dict[str, dict] = {
    "segmentation": {
        "pip": ["transformers", "torch", "pillow"],
        "snippet": '''import sys
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch

REPO = "{repo}"
img = Image.open(sys.argv[1]).convert("RGB")   # pass an image path
proc = AutoImageProcessor.from_pretrained(REPO)
model = SegformerForSemanticSegmentation.from_pretrained(REPO).to("{device}")
inputs = proc(images=img, return_tensors="pt").to("{device}")
with torch.no_grad():
    logits = model(**inputs).logits          # [1, num_labels, h/4, w/4]
seg = logits.argmax(1)[0]
print("segmentation map:", tuple(seg.shape), "labels:", sorted(set(seg.flatten().tolist()))[:10])
''',
    },
    "vision_language": {
        "pip": ["transformers", "torch", "pillow"],
        "snippet": '''import sys
from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch

REPO = "{repo}"
proc = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
model = AutoModel.from_pretrained(REPO, trust_remote_code=True, torch_dtype=torch.float16).to("{device}")
img = Image.open(sys.argv[1]).convert("RGB")
print("loaded VLM", REPO, "- see the model card for the exact chat/generate call.")
''',
    },
    "llm": {
        "pip": ["transformers", "torch", "accelerate"],
        "snippet": '''from transformers import pipeline
pipe = pipeline("text-generation", model="{repo}", device_map="auto", trust_remote_code=True)
print(pipe("Explain edge AI in one sentence.", max_new_tokens=64)[0]["generated_text"])
''',
    },
    "embeddings": {
        "pip": ["sentence-transformers", "torch"],
        "snippet": '''from sentence_transformers import SentenceTransformer
m = SentenceTransformer("{repo}", trust_remote_code=True, device="{device}")
vecs = m.encode(["edge compute", "distributed inference"])
print("embedding dim:", len(vecs[0]))
''',
    },
    "vision_backbone": {
        "pip": ["transformers", "torch", "pillow"],
        "snippet": '''from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("{repo}", trust_remote_code=True).to("{device}").eval()
x = torch.randn(1, 3, 224, 224, device="{device}")
with torch.no_grad():
    out = model(x)
print("backbone output type:", type(out).__name__)
''',
    },
}

# Categories that need a special runtime: download + point to the card (no fake snippet).
SPECIAL = {
    "speech": "ASR models (Canary/Parakeet) run on NVIDIA NeMo. Install per the model card: pip install -U nemo_toolkit['asr']",
    "world_model": "Cosmos world models need the NVIDIA Cosmos toolkit (gated). Follow the model card's setup.",
    "robotics_foundation": "GR00T runs on the Isaac GR00T stack. Follow the model card / NVIDIA Isaac instructions.",
}


def load_models() -> list[dict]:
    with open(CSV, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def categories(models: list[dict]) -> list[str]:
    return sorted({m["category"] for m in models})


def install_commands(model: dict) -> list[str]:
    cat = model["category"]
    cmds = []
    runner = RUNNERS.get(cat)
    if runner:
        cmds.append("pip install -U " + " ".join(runner["pip"]))
    cmds.append(f"huggingface-cli download {model['repo_id']}")
    return cmds


def snippet_for(model: dict, *, device: str = "cpu") -> str | None:
    runner = RUNNERS.get(model["category"])
    if not runner:
        return None
    return runner["snippet"].format(repo=model["repo_id"], device=device)


def _pick(prompt: str, options: list[str]) -> int | None:
    for i, o in enumerate(options, 1):
        print(f"  {i:2d}. {o}")
    raw = input(f"{prompt} (number, blank=back): ").strip()
    if not raw or not raw.isdigit() or not (1 <= int(raw) <= len(options)):
        return None
    return int(raw) - 1


def run_menu() -> int:  # pragma: no cover - interactive
    models = load_models()
    while True:
        print("\n=== NVIDIA HF model installer ===")
        cats = categories(models)
        ci = _pick("category", cats)
        if ci is None:
            return 0
        cat = cats[ci]
        subset = [m for m in models if m["category"] == cat]
        mi = _pick(f"model in '{cat}'", [f"{m['repo_id']}  ({m['license']})" for m in subset])
        if mi is None:
            continue
        model = subset[mi]
        print(f"\n{model['repo_id']} — {model['description']}\n  {model['url']}")
        if cat in SPECIAL:
            print("  NOTE:", SPECIAL[cat])
        action = _pick("action", ["show install steps", "download now (huggingface-cli)",
                                   "write a ready-to-run script", "choose device (cpu/cuda)"])
        device = "cpu"
        if action == 0:
            print("\n".join("  $ " + c for c in install_commands(model)))
        elif action == 1:
            for c in install_commands(model):
                print("  $", c)
            if input("run the download now? [y/N] ").strip().lower().startswith("y"):
                subprocess.run(["huggingface-cli", "download", model["repo_id"]])
        elif action == 2:
            snip = snippet_for(model, device=device)
            if not snip:
                print("  no generic runner for this category;", SPECIAL.get(cat, "see the model card."))
            else:
                out = os.path.join(ROOT, f"run_{model['repo_id'].split('/')[-1]}.py")
                with open(out, "w", encoding="utf-8") as fh:
                    fh.write(snip)
                print(f"  wrote {out}\n  install:", install_commands(model)[0])
        elif action == 3:
            device = input("  device [cpu/cuda]: ").strip() or "cpu"
            print(f"  device set to {device} (used when writing a run script)")


def selftest() -> int:
    """Non-interactive checks for CI (no model downloads)."""
    models = load_models()
    assert models and {"repo_id", "category"} <= set(models[0]), "csv shape"
    seg = next(m for m in models if m["category"] == "segmentation")
    assert any("huggingface-cli download" in c for c in install_commands(seg))
    assert seg["repo_id"] in (snippet_for(seg) or "")
    for cat in SPECIAL:
        m = next((x for x in models if x["category"] == cat), None)
        if m:
            assert snippet_for(m) is None, f"{cat} must not fake a runnable snippet"
    print("[OK] install_models selftest passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(selftest() if "--selftest" in sys.argv else run_menu())
