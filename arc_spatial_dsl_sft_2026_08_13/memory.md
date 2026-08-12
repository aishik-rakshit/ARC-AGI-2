# memory.md — Technical context for `arc_spatial_dsl_sft`

## Model
- Base model:
- Quantization:
- LoRA rank / alpha:
- Max seq length:

## Dataset
- Source:
- Format:
- Size (train / val):
- Prompt style:

## Hyperparameters
- Learning rate:
- Batch size / grad accum:
- Steps / epochs:
- Scheduler:

## Discoveries & Notes
<!-- Record debugging findings, unexpected behaviour, tuning decisions -->
# Working notes

- The local machine is macOS; the deliverable targets Kaggle CUDA rather than local training.
- Evaluation must execute predicted programs, not rely only on exact label-string matching.
- Keep synthetic transformations diverse; cosmetic permutations are a regularizer, not the dataset.
