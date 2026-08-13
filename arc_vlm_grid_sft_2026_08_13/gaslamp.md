# ARC VLM grid SFT roadbook

## Goal

Infer exact ARC-AGI-2 query output grids from demonstrations. Success is parsed JSON whose `output` grid exactly equals each of the 172 untouched official evaluation outputs.

## Method

Use vision SFT because labeled output grids exist. The input is deliberately redundant: one composite task image supplies whole-scene spatial structure, while text contains the exact grid digits, dimensions, and colors. The target is only `{"output":[[...]]}`. No unobserved program labels are invented.

## Model

- Base: `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`
- Kaggle mirror: `aishikai/qwen3-vl-4b-instruct-unsloth-bnb-4bit/Transformers/bnb-4bit/1`
- QLoRA: rank 16, alpha 16, dropout 0, all vision/language linear layers
- Export: adapter only at `outputs/adapters`

The 4-bit 4B checkpoint leaves room for multimodal activations on each 24 GB L4. LoRA updates both modalities without duplicating a full trainable model.

## Data

- Source: official ARC Prize 2026 competition JSON
- Train: 4,308 episodes from 1,000 training tasks
- Composition: 3,232 leave-one-demonstration-out episodes and 1,076 official train queries
- Validation: 172 query outputs from 120 official evaluation tasks
- Format: TRL `prompt`/`completion`, a top-level image list, and exact output grids
- Augmentation: none; no rotation, reflection, recoloring, or demo-order multiplication

Leave-one-out turns each known demonstration into a query while preserving the other demonstrations as context. The evaluation split is serialized separately and never passed to the trainer.

## Environment and hyperparameters

| Parameter | Value |
|---|---|
| Hardware | Kaggle 4 × NVIDIA L4 |
| Python | Kaggle 3.12 runtime |
| Stack | Unsloth 2026.8.13, TRL 0.24.0, Transformers 4.57.6 |
| Per-device batch | 1 |
| Gradient accumulation | 4 |
| Effective batch | 16 |
| Epochs | 1 (about 270 optimizer steps) |
| Learning rate | 1e-4 |
| Scheduler / warmup | cosine / 3% |
| Optimizer | 8-bit AdamW |
| Context | 8,192 tokens |
| Image maximum side | 1,024 pixels |
| Loss | completion only |

## Evaluation

`evaluate.py` performs greedy generation, extracts the first JSON object, validates a rectangular 1–30 cell grid with colors 0–9, and compares it exactly. It saves every raw generation, parse error, prediction, expected output, and correctness flag to `outputs/eval_records.json`.

## Files

| File | Role |
|---|---|
| `arc_vlm_data.py` | Render and serialize real ARC episodes |
| `train_ddp.py` | Four-process Unsloth vision QLoRA trainer |
| `evaluate.py` | Exact-grid held-out evaluator |
| `build_notebook.py` | Rebuild the Kaggle notebook from checked scripts |
| `arc_vlm_grid_sft.ipynb` | Offline Kaggle training notebook |
| `gaslamp_callback.py` | Training metrics callback copied from the Unsloth skill |
| `templates/dashboard.html` | Dashboard UI copied from the Unsloth skill |
| `kernel-metadata.json` | Private offline Kaggle notebook metadata |

Rebuild with `python3 build_notebook.py`; execute on Kaggle only after attaching the competition data, wheelhouse, and model mirror.
