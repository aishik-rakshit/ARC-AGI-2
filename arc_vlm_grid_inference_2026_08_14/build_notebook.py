"""Build the offline four-L4 ARC VLM inference notebook."""

import json
from pathlib import Path


HERE = Path(__file__).parent
TRAIN_DIR = HERE.parent / "arc_vlm_grid_sft_2026_08_13"
OUTPUT = HERE / "arc_vlm_grid_inference.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def write_file_cell(path):
    source = path.read_text()
    return cell(
        "code",
        "from pathlib import Path\n"
        f"Path({path.name!r}).write_text({source!r})\n"
        f"print('Wrote {path.name}')\n",
    )


infer_source = r'''"""Generate two ARC output-grid attempts on one Kaggle GPU shard."""

import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch


RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
SEED = 3407
MAX_SEQ_LENGTH = 8192
MAX_NEW_TOKENS = 2048
if WORLD_SIZE != 4:
    raise RuntimeError(f"Expected four GPU processes, got {WORLD_SIZE}")
torch.cuda.set_device(RANK)

from unsloth import FastVisionModel

from arc_vlm_data import render_task, task_prompt, validate_grid
from evaluate import parse_output


config = json.loads(Path("inference_paths.json").read_text())
challenges = json.loads(Path(config["test_file"]).read_text())
model, processor = FastVisionModel.from_pretrained(
    model_name=config["adapter"],
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    local_files_only=True,
)
FastVisionModel.for_inference(model)
processor.tokenizer.eos_token = "<|im_end|>"


@torch.inference_mode()
def generate(demos, query, sample, seed):
    image = render_task(demos, query)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": task_prompt(demos, query)},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": sample,
        "use_cache": True,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if sample:
        kwargs.update(temperature=0.35, top_p=0.95)
    tokens = model.generate(**inputs, **kwargs)
    raw = processor.tokenizer.decode(
        tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return parse_output(raw), raw


records = {}
task_items = list(challenges.items())[RANK::WORLD_SIZE]
for task_number, (task_id, task) in enumerate(task_items, 1):
    task_records = []
    for test_index, test in enumerate(task["test"]):
        query = validate_grid(test["input"])
        attempts, raw, errors = [], [], []
        for sample in (False, True):
            try:
                grid, text = generate(
                    task["train"], query, sample,
                    SEED + int(task_id, 16) + test_index * 2 + int(sample),
                )
                attempts.append(grid)
                raw.append(text)
                errors.append(None)
            except (json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
                attempts.append(query)
                raw.append("")
                errors.append(str(exc))
        task_records.append({
            "attempt_1": attempts[0],
            "attempt_2": attempts[1],
            "raw": raw,
            "errors": errors,
        })
    records[task_id] = task_records
    print(f"rank={RANK} tasks={task_number}/{len(task_items)}", flush=True)

Path("outputs").mkdir(exist_ok=True)
Path(f"outputs/shard_{RANK}.json").write_text(json.dumps(records))
'''


cells = [
    cell(
        "markdown",
        """# ARC VLM grid inference

Loads the Qwen3-VL adapter produced by `arc-vlm-grid-sft`, runs two direct output-grid generations per ARC-AGI-2 test query across four L4 GPUs, validates the official schema, and writes `/kaggle/working/submission.json`.

The notebook is fully offline.
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

wheelhouse = Path("/kaggle/input/datasets/aishikai/offline-unsloth-trl-wheelhouse-py312-cu128")
assert (wheelhouse / "requirements.in").exists(), "Offline wheelhouse is not attached"
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-index",
    "--find-links", str(wheelhouse), "-r", str(wheelhouse / "requirements.in"),
], check=True)
print("Offline dependencies installed.")
''',
    ),
    write_file_cell(TRAIN_DIR / "arc_vlm_data.py"),
    write_file_cell(TRAIN_DIR / "evaluate.py"),
    cell("code", f"from pathlib import Path\nPath('infer_ddp.py').write_text({infer_source!r})\nprint('Wrote infer_ddp.py')\n"),
    cell(
        "code",
        '''import json
from pathlib import Path


def find_file(name, preferred=()):
    matches = list(Path("/kaggle/input").glob(f"**/{name}"))
    matches.sort(key=lambda path: (-sum(part in str(path) for part in preferred), len(path.parts)))
    return matches[0] if matches else None


adapter_file = find_file(
    "adapter_model.safetensors", preferred=("arc-vlm-grid-sft", "outputs/adapters")
)
test_file = find_file("arc-agi_test_challenges.json", preferred=("arc-prize-2026",))
sample_file = find_file("sample_submission.json", preferred=("arc-prize-2026",))
assert adapter_file and (adapter_file.parent / "adapter_config.json").exists(), "Training adapter is not attached"
assert adapter_file.stat().st_size > 100_000_000, "Adapter file is unexpectedly small"
assert test_file and sample_file, "ARC Prize 2026 competition files are not attached"

paths = {
    "adapter": str(adapter_file.parent),
    "test_file": str(test_file),
    "sample_file": str(sample_file),
}
Path("inference_paths.json").write_text(json.dumps(paths))
print(paths, {"adapter_bytes": adapter_file.stat().st_size})
''',
    ),
    cell(
        "code",
        '''import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "torch.distributed.run", "--standalone",
    "--nproc_per_node=4", "infer_ddp.py",
], check=True)
print("All inference shards completed.")
''',
    ),
    cell(
        "code",
        '''import json
from pathlib import Path

paths = json.loads(Path("inference_paths.json").read_text())
sample = json.loads(Path(paths["sample_file"]).read_text())
records = {}
for rank in range(4):
    records.update(json.loads(Path(f"outputs/shard_{rank}.json").read_text()))

assert set(records) == set(sample)
submission = {}
diagnostics = {}
for task_id, expected_rows in sample.items():
    assert len(records[task_id]) == len(expected_rows)
    submission[task_id] = []
    diagnostics[task_id] = []
    for row in records[task_id]:
        prediction = {"attempt_1": row["attempt_1"], "attempt_2": row["attempt_2"]}
        assert set(prediction) == {"attempt_1", "attempt_2"}
        for grid in prediction.values():
            assert isinstance(grid, list) and grid and isinstance(grid[0], list) and grid[0]
            width = len(grid[0])
            assert len(grid) <= 30 and width <= 30 and all(len(line) == width for line in grid)
            assert all(type(value) is int and 0 <= value <= 9 for line in grid for value in line)
        submission[task_id].append(prediction)
        diagnostics[task_id].append({"raw": row["raw"], "errors": row["errors"]})

submission_file = Path("/kaggle/working/submission.json")
submission_file.write_text(json.dumps(submission, separators=(",", ":")))
Path("/kaggle/working/inference_diagnostics.json").write_text(json.dumps(diagnostics))
print({
    "tasks": len(submission),
    "outputs": sum(len(rows) for rows in submission.values()),
    "bytes": submission_file.stat().st_size,
    "file": str(submission_file),
})
''',
    ),
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
