# memory.md — Technical context for `arc_spatial_dsl_sft`

## Model
- Base model: `aishikai/qwen3-4b-instruct-2507-unsloth-4bit`
- Quantization: bitsandbytes 4-bit
- LoRA rank / alpha: 32 / 32
- Max seq length: 4096

## Dataset
- Source: procedural ARC-style episodes
- Format: TRL messages with a canonical JSON program label
- Size (train / val): 12,000 / 512
- Prompt style: system instruction, DSL, three demonstrations, hidden query

## Hyperparameters
- Learning rate: 2e-4
- Batch size / grad accum: 2 / 2 per GPU, four GPUs
- Steps / epochs: 1 epoch
- Scheduler: cosine

## Discoveries & Notes
<!-- Record debugging findings, unexpected behaviour, tuning decisions -->
# Working notes

- The local machine is macOS; the deliverable targets Kaggle CUDA rather than local training.
- Evaluation must execute predicted programs, not rely only on exact label-string matching.
- Keep synthetic transformations diverse; cosmetic permutations are a regularizer, not the dataset.
- Offline packages come from `aishikai/offline-unsloth-trl-wheelhouse-py312-cu128`; it supplies xFormers 0.0.34, torchao 0.16.0, and bitsandbytes 0.50.0 for Kaggle's PyTorch 2.10/CUDA 12.8 image.
- Pin TRL 0.24.0 for Unsloth 2026.8.13; use `processing_class` and put dataset options in `SFTConfig`.
