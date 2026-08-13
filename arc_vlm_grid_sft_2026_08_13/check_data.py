"""CPU check for rendered ARC vision episodes."""

import json
import tempfile
from pathlib import Path

from arc_vlm_data import build_datasets, render_task, task_prompt, validate_grid


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    competition = root / "competition"
    competition.mkdir()
    task = {
        "a": {
            "train": [
                {"input": [[1, 0]], "output": [[0], [1]]},
                {"input": [[2, 0]], "output": [[0], [2]]},
            ],
            "test": [{"input": [[3, 0]]}],
        }
    }
    for split in ("training", "evaluation"):
        (competition / f"arc-agi_{split}_challenges.json").write_text(json.dumps(task))
        (competition / f"arc-agi_{split}_solutions.json").write_text(json.dumps({"a": [[[0], [3]]]}))
    train_count, eval_count = build_datasets(competition, root / "data")
    assert (train_count, eval_count) == (3, 1)

    from datasets import load_from_disk

    row = load_from_disk(root / "data/train")[0]
    assert row["images"][0].mode == "RGB"
    assert row["prompt"][0]["content"][0]["type"] == "image"
    assert json.loads(row["completion"][0]["content"][0]["text"])["output"] == row["output"]
    assert "QUERY INPUT" in row["prompt"][0]["content"][1]["text"]
    validate_grid(row["output"])

print("ARC vision data check passed")
