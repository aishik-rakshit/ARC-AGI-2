"""Train one Qwen3-VL LoRA adapter on four Kaggle L4 GPUs."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch


SEED = 3407
MAX_LENGTH = 8192
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
if WORLD_SIZE != 4:
    raise RuntimeError(f"Expected four DDP processes, got {WORLD_SIZE}")
torch.cuda.set_device(LOCAL_RANK)

from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

from gaslamp_callback import GaslampDashboardCallback


model_path = Path("data/model_path.txt").read_text().strip()
model, processor = FastVisionModel.from_pretrained(
    model_name=model_path,
    max_seq_length=MAX_LENGTH,
    dtype=None,
    load_in_4bit=True,
    local_files_only=True,
    use_gradient_checkpointing="unsloth",
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
    target_modules="all-linear",
)

trainer = SFTTrainer(
    model=model,
    processing_class=processor,
    train_dataset=load_from_disk("data/train"),
    data_collator=UnslothVisionDataCollator(
        model,
        processor,
        max_seq_length=MAX_LENGTH,
        resize=1024,
        resize_dimension="max",
        snap_to_patch_size=True,
        completion_only_loss=True,
    ),
    callbacks=[GaslampDashboardCallback(task_type="vision")],
    args=SFTConfig(
        output_dir="outputs/checkpoints",
        logging_dir="outputs/logs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=5,
        save_strategy="epoch",
        report_to="tensorboard",
        seed=SEED,
        data_seed=SEED,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        packing=False,
        padding_free=False,
        ddp_find_unused_parameters=False,
    ),
)
trainer.train()
if trainer.is_world_process_zero():
    trainer.model.save_pretrained("outputs/adapters")
    processor.save_pretrained("outputs/adapters")
