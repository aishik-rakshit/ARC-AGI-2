"""Build rendered, exact-grid ARC vision SFT episodes."""

import json
from pathlib import Path

from datasets import Dataset, Image, Sequence
from PIL import Image as PILImage, ImageDraw, ImageFont


PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37),
]


def validate_grid(grid):
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
        raise ValueError("grid must be a non-empty list of rows")
    width = len(grid[0])
    if len(grid) > 30 or width > 30 or any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular and at most 30x30")
    if any(not isinstance(value, int) or not 0 <= value <= 9 for row in grid for value in row):
        raise ValueError("grid colors must be integers 0..9")
    return grid


def grid_text(grid):
    validate_grid(grid)
    return "\n".join("".join(str(value) for value in row) for row in grid)


def task_prompt(demos, query):
    parts = [
        "Infer the exact output for QUERY. Use the image for whole-scene spatial structure and the exact grids below for coordinates and colors.",
    ]
    for index, example in enumerate(demos, 1):
        inp, out = validate_grid(example["input"]), validate_grid(example["output"])
        parts.append(f"DEMO {index} INPUT ({len(inp)}x{len(inp[0])}):\n{grid_text(inp)}")
        parts.append(f"DEMO {index} OUTPUT ({len(out)}x{len(out[0])}):\n{grid_text(out)}")
    query = validate_grid(query)
    parts.append(f"QUERY INPUT ({len(query)}x{len(query[0])}):\n{grid_text(query)}")
    parts.append('Return only JSON in the form {"output":[[...]]}.')
    return "\n\n".join(parts)


def _grid_image(grid, size=210):
    grid = validate_grid(grid)
    height, width = len(grid), len(grid[0])
    cell = max(3, min(size // height, size // width))
    image = PILImage.new("RGB", (width * cell + 1, height * cell + 1), (64, 64, 64))
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            x, y = col * cell, row * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=PALETTE[value])
    return image


def _pair_panel(label, inp, out=None):
    font = ImageFont.load_default()
    panel = PILImage.new("RGB", (500, 255), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 8), label, fill="black", font=font)
    draw.text((8, 28), "INPUT", fill="black", font=font)
    left = _grid_image(inp)
    panel.paste(left, (8, 45))
    if out is not None:
        draw.text((258, 28), "OUTPUT", fill="black", font=font)
        right = _grid_image(out)
        panel.paste(right, (258, 45))
    else:
        draw.text((258, 100), "?", fill="black", font=font)
    return panel


def render_task(demos, query):
    panels = [_pair_panel(f"DEMO {index}", row["input"], row["output"]) for index, row in enumerate(demos, 1)]
    panels.append(_pair_panel("QUERY", query))
    rows = (len(panels) + 1) // 2
    image = PILImage.new("RGB", (1010, rows * 265), (224, 224, 224))
    for index, panel in enumerate(panels):
        image.paste(panel, ((index % 2) * 510, (index // 2) * 265))
    if max(image.size) > 1280:
        scale = 1280 / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), PILImage.Resampling.NEAREST)
    return image


def episode(task_id, episode_id, demos, query, output, image_dir):
    output = validate_grid(output)
    path = image_dir / f"{task_id}_{episode_id}.png"
    render_task(demos, query).save(path, optimize=True)
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "images": [str(path)],
        "prompt": [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": task_prompt(demos, query)},
            ],
        }],
        "completion": [{
            "role": "assistant",
            "content": [{"type": "text", "text": json.dumps({"output": output}, separators=(",", ":"))}],
        }],
        "output": output,
    }


def build_split(challenges, solutions, image_dir, leave_one_out):
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id, task in challenges.items():
        train = task["train"]
        if leave_one_out:
            for index, held_out in enumerate(train):
                demos = train[:index] + train[index + 1:]
                rows.append(episode(task_id, f"demo_{index}", demos, held_out["input"], held_out["output"], image_dir))
        for index, test in enumerate(task["test"]):
            rows.append(episode(task_id, f"test_{index}", train, test["input"], solutions[task_id][index], image_dir))
    return rows


def build_datasets(competition_dir, output_dir):
    competition_dir, output_dir = Path(competition_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((competition_dir / name).read_text())
    train_rows = build_split(
        load("arc-agi_training_challenges.json"), load("arc-agi_training_solutions.json"),
        output_dir / "images/train", leave_one_out=True,
    )
    eval_rows = build_split(
        load("arc-agi_evaluation_challenges.json"), load("arc-agi_evaluation_solutions.json"),
        output_dir / "images/eval", leave_one_out=False,
    )
    for name, rows in (("train", train_rows), ("eval", eval_rows)):
        dataset = Dataset.from_list(rows).cast_column("images", Sequence(Image()))
        dataset.save_to_disk(output_dir / name)
    return len(train_rows), len(eval_rows)
