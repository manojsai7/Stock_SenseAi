# Dataset admission

No dataset is currently approved for V2 training. Keep recovered EXP-05 snapshots separate from current production data. Historical reproduction is not proof that a snapshot contains valid exchange sessions.

Before admitting a dataset, record its exact file SHA-256, source/download timestamp, symbol, adjustment settings, exchange-session checks, corporate-action observations, investigation evidence, and completed-session cutoff. Resolve suspicious flat/zero-volume bars against trustworthy source evidence. Do not forward-fill or synthesize missing sessions.

Set the experiment configuration's data gate only after documenting these checks. Colab must verify the exact approved hash before any fitting. Keep the final holdout out of exploratory metrics and candidate selection.
