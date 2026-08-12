# Data strategy

The notebook generates data locally and deterministically; no external training dataset is required.

Each SFT row has a TRL-compatible `messages` column:

- system: define the role and require JSON-only output;
- user: DSL grammar, 3 input/output demonstrations, and one query input;
- assistant: the canonical JSON DSL program that explains the demonstrations.

The query output is retained only for evaluation. It is never included in the model prompt.

Training episodes vary the actual rule, parameters, object layouts, dimensions, backgrounds, colors, and demonstration order. Supported families include geometric transformations, object selection/recoloring/movement/extraction, foreground cropping/scaling, symmetry completion, marker connection, marker-to-object color transfer, and colored-path traversal. Rotation and color changes are part of this distribution but are not used to inflate one puzzle into many nominal samples.

Default size: 12,000 training episodes and 512 held-out episodes. Set `SMOKE_TEST=True` for a 256/64 pipeline check.

Validation checks in the notebook assert that every target program executes correctly, the hidden query output is absent from the prompt, JSON labels round-trip, and held-out episode scenes use a disjoint random seed range.
