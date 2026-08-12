"""CPU check for the notebook's procedural generator and DSL executor."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


notebook = json.loads(Path(__file__).with_name("arc_spatial_program_sft.ipynb").read_text())
scope = {"json": json, "random": random, "Counter": Counter, "np": np}

for tag in ("dsl-core", "data-generator"):
    cell = next(cell for cell in notebook["cells"] if tag in cell["metadata"].get("tags", []))
    source = "".join(cell["source"])
    exec(compile(source, f"<notebook cell {tag}>", "exec"), scope)

for family_index, family in enumerate(scope["EVAL_FAMILIES"]):
    for example_index in range(20):
        seed = 20_000_000 + family_index * 100 + example_index
        row = scope["make_episode"](seed, [family])
        program = json.loads(row["program"])
        assert scope["apply_program"](row["query_input"], program) == row["query_output"]
        assert "QUERY OUTPUT" not in row["messages"][1]["content"]

print("notebook generator/executor check passed")
