# Data strategy

## Source

Use only the official ARC Prize 2026 ARC-AGI-2 files:

- `arc-agi_training_challenges.json`
- `arc-agi_training_solutions.json`
- `arc-agi_evaluation_challenges.json`
- `arc-agi_evaluation_solutions.json`

The 1,000 training tasks produce 4,308 supervised episodes:

- 3,232 leave-one-demonstration-out episodes.
- 1,076 official training-query episodes.

The 120 evaluation tasks and their 172 query outputs are never included in training.

## Episode format

Each TRL row uses prompt/completion fields and a top-level image list:

```python
{
  "images": [composite_task_image],
  "prompt": [{"role": "user", "content": [
      {"type": "image", "image": composite_task_image},
      {"type": "text", "text": exact_numeric_grid_prompt},
  ]}],
  "completion": [{"role": "assistant", "content": [
      {"type": "text", "text": '{"output":[[...]]}'},
  ]}],
}
```

The composite image shows all demonstrations and the query together. The text duplicates every grid using exact digits, dimensions, and labels. This gives the VLM global spatial context without sacrificing cell-level precision.

## Non-augmentation decision

Do not create nominal samples by color permutation, rotation, reflection, or demonstration reordering. Each episode changes because a different real example is the held-out query. This avoids repeating the prior synthetic-family distribution mismatch.
