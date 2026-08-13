"""Build the offline Kaggle vision-SFT notebook from checked local scripts."""

import json
from pathlib import Path


HERE = Path(__file__).parent
OUTPUT = HERE / "arc_vlm_grid_sft.ipynb"


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def writefile(name):
    return f"%%writefile {name}\n" + (HERE / name).read_text()


dashboard = (HERE / "templates/dashboard.html").read_text()
cells = [
    cell("markdown", """# ARC vision-to-grid SFT

This notebook trains one Qwen3-VL 4B LoRA adapter on four L4 GPUs. Each episode presents the full ARC task as one image plus exact numeric grids, and the model learns to return only `{"output":[[...]]}`.

The 4,308 training episodes come from real ARC training tasks: leave-one-demonstration-out episodes plus official training queries. The 172 official evaluation queries are untouched by training and scored by exact grid equality. There is no rotation or color-permutation multiplication.

Run with Kaggle's **4 x L4** accelerator. Internet remains disabled.
"""),
    cell("code", '''import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SMOKE_TEST = False
os.environ["ARC_VLM_SMOKE_TEST"] = "1" if SMOKE_TEST else "0"

WHEELHOUSE = Path("/kaggle/input/datasets/aishikai/offline-unsloth-trl-wheelhouse-py312-cu128")
requirements = WHEELHOUSE / "requirements.in"
assert requirements.exists(), "Attach aishikai/offline-unsloth-trl-wheelhouse-py312-cu128"
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index",
    "--find-links", str(WHEELHOUSE), "-r", str(requirements),
], check=True)
missing = [name for name in ("unsloth", "trl", "datasets", "tensorboard") if importlib.util.find_spec(name) is None]
assert not missing, f"Offline dependency installation failed: {missing}"
print("Offline dependencies installed.")
'''),
    cell("code", '''from pathlib import Path


def find_parent(filename, sibling=None, preferred=()):
    matches = []
    for path in Path("/kaggle/input").glob(f"**/{filename}"):
        if sibling and not (path.parent / sibling).exists():
            continue
        matches.append((-sum(part in str(path).lower() for part in preferred), len(path.parts), path.parent))
    return sorted(matches)[0][2] if matches else None


COMPETITION_DIR = find_parent(
    "arc-agi_training_challenges.json",
    sibling="arc-agi_evaluation_solutions.json",
    preferred=("arc-prize-2026",),
)
MODEL_PATH = find_parent(
    "config.json",
    sibling="model.safetensors.index.json",
    preferred=("qwen3-vl-4b", "bnb-4bit"),
)
if MODEL_PATH is None:
    MODEL_PATH = find_parent("config.json", preferred=("qwen3-vl-4b", "bnb-4bit"))
assert COMPETITION_DIR, "Attach the ARC Prize 2026 competition data"
assert MODEL_PATH, "Attach aishikai/qwen3-vl-4b-instruct-unsloth-bnb-4bit"
Path("data").mkdir(exist_ok=True)
Path("data/model_path.txt").write_text(str(MODEL_PATH))
print({"competition": str(COMPETITION_DIR), "model": str(MODEL_PATH)})
'''),
    cell("code", writefile("arc_vlm_data.py")),
    cell("code", '''from arc_vlm_data import build_datasets

train_count, eval_count = build_datasets(COMPETITION_DIR, "data")
assert (train_count, eval_count) == (4308, 172), (train_count, eval_count)
print({"train_episodes": train_count, "held_out_queries": eval_count})
'''),
    cell("code", writefile("gaslamp_callback.py")),
    cell("code", '''from pathlib import Path

Path("templates").mkdir(exist_ok=True)
Path("templates/dashboard.html").write_text(''' + repr(dashboard) + ''')
print("Training dashboard asset ready.")
'''),
    cell("code", writefile("train_ddp.py")),
    cell("code", '''import torch

assert torch.cuda.device_count() == 4, f"Select 4 x L4; found {torch.cuda.device_count()} GPU(s)"
!nvidia-smi -L
subprocess.run(["torchrun", "--standalone", "--nproc_per_node=4", "train_ddp.py"], check=True)
'''),
    cell("markdown", """## Exact-grid validation

This loads the saved adapter and evaluates all 172 held-out queries greedily. It saves the raw generations, parsed predictions, and correctness flags to `outputs/eval_records.json`.
"""),
    cell("code", writefile("evaluate.py")),
    cell("code", '''!python evaluate.py

from pathlib import Path

assert Path("outputs/adapters/adapter_model.safetensors").exists()
assert Path("outputs/eval_records.json").exists()
print("Adapter and held-out predictions saved under outputs/.")
'''),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
