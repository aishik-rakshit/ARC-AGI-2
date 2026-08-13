"""Small CPU checks for notebook structure and output parsing."""

import ast
import json
from pathlib import Path

from evaluate import clean_messages, parse_output


HERE = Path(__file__).parent
notebook = json.loads((HERE / "arc_vlm_grid_sft.ipynb").read_text())
sources = ["".join(cell["source"]) for cell in notebook["cells"]]
metadata = json.loads((HERE / "kernel-metadata.json").read_text())
smoke = json.loads((HERE / "smoke/arc_vlm_grid_sft_smoke.ipynb").read_text())

assert len(notebook["cells"]) == 12
assert any('subprocess.run(["torchrun", "--standalone", "--nproc_per_node=4"' in source for source in sources)
assert any("(train_count, eval_count) == (4308, 172)" in source for source in sources)
assert any("completion_only_loss=True" in source for source in sources)
assert metadata["enable_internet"] is False
assert metadata["machine_shape"] == "NvidiaL4"
assert sum("SMOKE_TEST = True" in cell["source"] for cell in smoke["cells"]) == 1
assert any("max_steps=2 if SMOKE_TEST else -1" in source for source in sources)
assert any('QWEN_EOS_TOKEN = "<|im_end|>"' in source and "training_args.eos_token = None" in source for source in sources)
assert parse_output('text {"output":[[1,0],[2,3]]}') == [[1, 0], [2, 3]]
assert clean_messages([{"role": "user", "content": [{"type": "text", "text": "x", "image": None}]}]) == [
    {"role": "user", "content": [{"type": "text", "text": "x"}]}
]
for name in ("arc_vlm_data.py", "train_ddp.py", "evaluate.py", "build_notebook.py"):
    ast.parse((HERE / name).read_text())

print("offline ARC VLM notebook check passed")
