"""Small CPU checks for notebook structure and output parsing."""

import ast
import json
from pathlib import Path

from evaluate import parse_output


HERE = Path(__file__).parent
notebook = json.loads((HERE / "arc_vlm_grid_sft.ipynb").read_text())
sources = ["".join(cell["source"]) for cell in notebook["cells"]]
metadata = json.loads((HERE / "kernel-metadata.json").read_text())

assert len(notebook["cells"]) == 12
assert any("torchrun --standalone --nproc_per_node=4" in source for source in sources)
assert any("(train_count, eval_count) == (4308, 172)" in source for source in sources)
assert any("completion_only_loss=True" in source for source in sources)
assert metadata["enable_internet"] is False
assert metadata["machine_shape"] == "NvidiaL4"
assert parse_output('text {"output":[[1,0],[2,3]]}') == [[1, 0], [2, 3]]
for name in ("arc_vlm_data.py", "train_ddp.py", "evaluate.py", "build_notebook.py"):
    ast.parse((HERE / name).read_text())

print("offline ARC VLM notebook check passed")
