# Working memory

- The prior 12-operation synthetic DSL expressed only 1/120 official evaluation tasks, so this version predicts real output grids directly.
- Base model: `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`, 4-bit QLoRA, rank/alpha 16, 8,192-token context.
- Dataset: 4,308 real training episodes and 172 untouched validation queries in multimodal prompt/completion format.
- Training: per-device batch 1, accumulation 4, four L4 GPUs, one epoch, 1e-4 cosine schedule.
- The wheelhouse versions were checked against TRL prompt/completion and Unsloth vision-collator source before writing the trainer.
- The full official dataset was built and reloaded locally with exact counts and decoded PIL images.
- The base checkpoint was verified on `/Volumes/Aishik_SSD3` before its private Kaggle Model upload.
- Unsloth must be imported before TRL; its generated SFT config placeholder otherwise leaks as the invalid token `<EOS_TOKEN>`.
- Arrow multimodal content contains null sibling fields; inference must remove them before `apply_chat_template` or Qwen3-VL counts a text block as another image.
- Kaggle smoke version 6 passed training, adapter reload, and two-query generation; the two-step adapter was not expected to solve either query.
