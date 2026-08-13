# Progress log

- 2026-08-13: Replaced synthetic DSL labels with direct real-ARC grid supervision.
- 2026-08-13: Fixed the training contract: Qwen3-VL 4B QLoRA, vision SFT, Kaggle 4 × L4, offline, adapter-only export.
- 2026-08-13: Built and reloaded all 4,308 training episodes and 172 held-out evaluation episodes.
- 2026-08-13: Added four-GPU completion-only training, exact-grid parsing/evaluation, notebook generation, and CPU checks.
- 2026-08-13: Mirrored the verified 4-bit base checkpoint from Hugging Face to SSD and a private Kaggle Model.
- 2026-08-13: Kaggle smoke version 6 completed two DDP optimizer steps, saved a 157,387,928-byte adapter, and generated two held-out records.
- 2026-08-13: Smoke metrics were loss 0.23247 and exact accuracy 0/2; two steps validate execution, not task quality.
- Full training and evaluation remain intentionally unexecuted until the user starts the production run.
