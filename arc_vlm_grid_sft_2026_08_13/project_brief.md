# Project brief

- Problem: fine-tune one fixed vision-language model to infer exact ARC-AGI-2 output grids from demonstrations and a query.
- Audience: ARC-AGI research and Kaggle experimentation.
- Method: vision supervised fine-tuning (SFT), followed by exact-grid evaluation.
- Base model: `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` mirrored as a private Kaggle Model for offline execution.
- Hardware: Kaggle 4 × NVIDIA L4, one distributed training process per GPU.
- Inputs: a composite image of the whole task plus the same grids serialized as exact digits.
- Target: `{"output":[[...], ...]}` with integer colors 0–9.
- Deployment: adapter-only export loaded with the offline base model.

Success is measured on all 172 query outputs from the untouched 120-task official evaluation split. Primary metric: exact output-grid accuracy. The notebook must report this score before the adapter is used for a competition submission.
