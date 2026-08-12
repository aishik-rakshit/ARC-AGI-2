# Project brief

- Problem: train one fixed language model to infer an executable spatial transformation program from ARC-style demonstrations.
- Audience: ARC-AGI research and Kaggle experimentation.
- Method: supervised fine-tuning (SFT) on procedurally generated episodes with exact program labels.
- Base model: private Kaggle Model `aishikai/qwen3-4b-instruct-2507-unsloth-4bit`, mirrored unchanged from `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`.
- Offline packages: private Kaggle Dataset `aishikai/offline-unsloth-trl-wheelhouse-py311`.
- Training hardware: Kaggle 4 × NVIDIA L4 GPUs using single-node DDP.
- Output: a lightweight LoRA adapter plus executable-program accuracy on held-out synthetic episodes.

The model is not fine-tuned separately for each ARC puzzle. One global adapter learns the DSL and rule-induction task; weights remain fixed when solving a puzzle.
