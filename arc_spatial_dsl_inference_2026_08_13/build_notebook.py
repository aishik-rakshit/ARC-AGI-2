"""Build the offline Kaggle inference notebook from the shared DSL cell."""

import json
from pathlib import Path


HERE = Path(__file__).parent
ROOT = HERE.parent
TRAIN_NOTEBOOK = ROOT / "arc_spatial_dsl_sft_2026_08_13" / "arc_spatial_program_sft.ipynb"
OUTPUT = HERE / "arc_spatial_program_inference.ipynb"


def cell(kind, source, tags=()):
    return {
        "cell_type": kind,
        "execution_count": None if kind == "code" else None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [] if kind == "code" else None,
        "source": source,
    }


training = json.loads(TRAIN_NOTEBOOK.read_text())
dsl_source = next(
    "".join(item["source"])
    for item in training["cells"]
    if "dsl-core" in item.get("metadata", {}).get("tags", [])
)

cells = [
    cell(
        "markdown",
        """# ARC spatial-program offline inference

This notebook loads the fixed Qwen3 LoRA adapter trained by `arc-spatial-program-sft`, infers two executable spatial programs for every ARC-AGI-2 test input, checks each program against all demonstrations, and writes `/kaggle/working/submission.json`.

It runs fully offline. Attach the ARC Prize 2026 ARC-AGI-2 competition, the Qwen base model, the Python wheelhouse, and the completed training notebook output. Use Kaggle's L4 accelerator.
""",
    ),
    cell(
        "code",
        '''import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WHEELHOUSE = Path("/kaggle/input/datasets/aishikai/offline-unsloth-trl-wheelhouse-py312-cu128")
assert (WHEELHOUSE / "requirements.in").exists(), "Attach aishikai/offline-unsloth-trl-wheelhouse-py312-cu128"
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index",
    "--find-links", str(WHEELHOUSE), "-r", str(WHEELHOUSE / "requirements.in"),
], check=True)
print("Offline dependencies installed.")
''',
        ("offline-install",),
    ),
    cell(
        "code",
        '''import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

SEED = 3407
MAX_SEQ_LENGTH = 12_288
MAX_NEW_TOKENS = 160


def find_parent(filename, required_sibling=None, preferred=()):
    matches = []
    for path in Path("/kaggle/input").glob(f"**/{filename}"):
        if required_sibling and not (path.parent / required_sibling).exists():
            continue
        score = sum(part in str(path) for part in preferred)
        matches.append((-score, len(path.parts), path.parent))
    return str(sorted(matches)[0][2]) if matches else None


BASE_MODEL = find_parent(
    "config.json", preferred=("qwen3-4b-instruct-2507-unsloth-4bit", "bnb-4bit")
)
ADAPTER = find_parent(
    "adapter_model.safetensors", required_sibling="adapter_config.json", preferred=("arc-spatial-program-sft", "outputs/adapters")
)
TEST_FILE = next(Path("/kaggle/input").glob("**/arc-agi_test_challenges.json"), None)

assert BASE_MODEL, "Attach aishikai/qwen3-4b-instruct-2507-unsloth-4bit"
assert ADAPTER, "Attach the output of aishikai/arc-spatial-program-sft"
assert TEST_FILE, "Attach the ARC Prize 2026 ARC-AGI-2 competition"
assert Path(ADAPTER, "adapter_model.safetensors").stat().st_size > 100_000_000

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print({"base_model": BASE_MODEL, "adapter": ADAPTER, "test_file": str(TEST_FILE)})
''',
        ("paths",),
    ),
    cell("code", dsl_source, ("dsl-core",)),
    cell(
        "code",
        '''DSL_SPEC = """Programs are JSON arrays executed left to right.
Allowed operations:
{"op":"rotate","k":1|2|3}
{"op":"flip","axis":"horizontal"|"vertical"}
{"op":"recolor_object","selector":SELECTOR,"color":0..9}
{"op":"recolor_foreground","color":0..9}
{"op":"move_object","selector":SELECTOR,"direction":"up"|"down"|"left"|"right","steps":1|2}
{"op":"extract_object","selector":SELECTOR}
{"op":"crop_foreground"}
{"op":"complete_symmetry","axis":"horizontal"|"vertical"}
{"op":"connect_markers"}
{"op":"trace_path","orientation":"row"|"column"}
{"op":"copy_marker_color","selector":SELECTOR}
{"op":"upscale","factor":2|3}
SELECTOR is largest, smallest, topmost, bottommost, leftmost, or rightmost.
Colors are integers 0..9. Return only the JSON array."""

SYSTEM_PROMPT = (
    "Infer the shortest valid spatial program that explains every demonstration. "
    "Return only one JSON array. Do not return prose or the query grid."
)


def render_grid(grid):
    return "\\n".join("".join(str(int(x)) for x in row) for row in grid)


def build_user_prompt(demos, query):
    parts = [DSL_SPEC, "--- DEMONSTRATIONS ---"]
    for index, example in enumerate(demos, 1):
        inp, out = example["input"], example["output"]
        parts.append(f"Demo {index} input ({len(inp)}x{len(inp[0])}):\\n{render_grid(inp)}")
        parts.append(f"Demo {index} output ({len(out)}x{len(out[0])}):\\n{render_grid(out)}")
    parts.append(f"--- QUERY ---\\nInput ({len(query)}x{len(query[0])}):\\n{render_grid(query)}")
    parts.append("Return the program only.")
    return "\\n\\n".join(parts)


def extract_program(text):
    start = text.find("[")
    if start < 0:
        raise ValueError("no JSON array")
    program, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(program, list):
        raise ValueError("program is not a list")
    return program


def solves_demos(program, demos):
    try:
        return all(apply_program(row["input"], program) == row["output"] for row in demos)
    except (KeyError, TypeError, ValueError):
        return False
''',
        ("prompt",),
    ),
    cell(
        "code",
        '''from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    local_files_only=True,
)
FastLanguageModel.for_inference(model)
tokenizer.truncation_side = "left"
print(f"Loaded adapter on {model.device}")
''',
        ("model",),
    ),
    cell(
        "code",
        '''@torch.inference_mode()
def generate_program(demos, query, sample, seed):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(demos, query)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
    ).to(model.device)
    torch.manual_seed(seed)
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": sample,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if sample:
        kwargs.update(temperature=0.6, top_p=0.9)
    output = model.generate(**inputs, **kwargs)
    text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_program(text)


def solve_test(task_id, demos, query, test_index):
    fallback = as_grid(query).tolist()
    predictions = []
    for sample in (False, True):
        try:
            seed = SEED + int(task_id, 16) + test_index * 2 + int(sample)
            program = generate_program(demos, query, sample, seed)
            if solves_demos(program, demos):
                grid = apply_program(query, program)
                if grid not in predictions:
                    predictions.append(grid)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    while len(predictions) < 2:
        predictions.append(predictions[0] if predictions else fallback)
    return {"attempt_1": predictions[0], "attempt_2": predictions[1]}


challenges = json.loads(TEST_FILE.read_text())
submission = {}
total = sum(len(task["test"]) for task in challenges.values())
done = 0
for task_id, task in challenges.items():
    submission[task_id] = []
    for test_index, test in enumerate(task["test"]):
        submission[task_id].append(solve_test(task_id, task["train"], test["input"], test_index))
        done += 1
        if done % 10 == 0:
            print(f"{done}/{total}")
''',
        ("inference",),
    ),
    cell(
        "code",
        '''assert set(submission) == set(challenges)
for task_id, task in challenges.items():
    assert len(submission[task_id]) == len(task["test"])
    for prediction in submission[task_id]:
        assert set(prediction) == {"attempt_1", "attempt_2"}
        as_grid(prediction["attempt_1"])
        as_grid(prediction["attempt_2"])

SUBMISSION_FILE = Path("/kaggle/working/submission.json")
SUBMISSION_FILE.write_text(json.dumps(submission, separators=(",", ":")))
print({"tasks": len(submission), "test_outputs": total, "file": str(SUBMISSION_FILE), "bytes": SUBMISSION_FILE.stat().st_size})
''',
        ("submission",),
    ),
]

for item in cells:
    if item["cell_type"] == "markdown":
        item.pop("execution_count")
        item.pop("outputs")

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
