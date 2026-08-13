"""Small CPU check for inference parsing, execution, and submission shape."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


notebook = json.loads(Path(__file__).with_name("arc_spatial_program_inference.ipynb").read_text())
scope = {"json": json, "Counter": Counter, "np": np}
for tag in ("dsl-core", "prompt"):
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    )
    exec(compile(source, f"<notebook {tag}>", "exec"), scope)

demos = [
    {"input": [[1, 0], [2, 0]], "output": [[2, 1], [0, 0]]},
    {"input": [[3, 0, 0]], "output": [[3], [0], [0]]},
]
program = [{"op": "rotate", "k": 1}]
assert scope["solves_demos"](program, demos)
assert scope["apply_program"]([[4, 5]], program) == [[4], [5]]
assert scope["extract_program"]('<think></think> [{"op":"rotate","k":1}]') == program

metadata = json.loads(Path(__file__).with_name("kernel-metadata.json").read_text())
assert metadata["enable_internet"] is False
assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
assert metadata["kernel_sources"] == ["aishikai/arc-spatial-program-sft"]
print("offline inference notebook check passed")
