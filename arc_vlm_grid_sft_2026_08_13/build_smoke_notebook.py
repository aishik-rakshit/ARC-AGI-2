"""Build a two-step, two-query Kaggle smoke notebook."""

import json
from pathlib import Path

import build_notebook


here = Path(__file__).parent
notebook = json.loads((here / "arc_vlm_grid_sft.ipynb").read_text())
changed = 0
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "SMOKE_TEST = False" in cell["source"]:
        cell["source"] = cell["source"].replace("SMOKE_TEST = False", "SMOKE_TEST = True")
        changed += 1
assert changed == 1
output = here / "smoke" / "arc_vlm_grid_sft_smoke.ipynb"
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(notebook, indent=1) + "\n")
print(output)
