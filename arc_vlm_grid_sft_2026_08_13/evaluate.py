"""Run exact-grid validation for the trained vision adapter."""

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_output(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    grid = value.get("output") if isinstance(value, dict) else None
    if not isinstance(grid, list) or not grid or any(not isinstance(row, list) for row in grid):
        raise ValueError("missing output grid")
    width = len(grid[0])
    if not 1 <= len(grid) <= 30 or not 1 <= width <= 30:
        raise ValueError("invalid grid size")
    if any(len(row) != width for row in grid):
        raise ValueError("non-rectangular grid")
    if any(type(value) is not int or not 0 <= value <= 9 for row in grid for value in row):
        raise ValueError("invalid color")
    return grid


def clean_messages(messages):
    return [{
        "role": message["role"],
        "content": [{key: value for key, value in item.items() if value is not None}
                    for item in message["content"]],
    } for message in messages]


def main():
    import torch
    from datasets import load_from_disk
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name="outputs/adapters",
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    FastVisionModel.for_inference(model)

    @torch.inference_mode()
    def predict(row):
        messages = clean_messages(row["prompt"])
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=row["images"], return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=128 if os.environ.get("ARC_VLM_SMOKE_TEST") == "1" else 2304,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        return processor.tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    dataset = load_from_disk("data/eval")
    if os.environ.get("ARC_VLM_SMOKE_TEST") == "1":
        dataset = dataset.select(range(2))
    records = []
    for index, row in enumerate(dataset):
        text = predict(row)
        try:
            prediction = parse_output(text)
            error = None
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            prediction, error = None, str(exc)
        records.append({
            "task_id": row["task_id"],
            "episode_id": row["episode_id"],
            "correct": prediction == row["output"],
            "prediction": prediction,
            "expected": row["output"],
            "raw": text,
            "error": error,
        })
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(dataset)}")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/eval_records.json").write_text(json.dumps(records))
    correct = sum(record["correct"] for record in records)
    print({"exact": correct, "total": len(records), "accuracy": correct / len(records)})


if __name__ == "__main__":
    main()
