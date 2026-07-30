# RUNS — lens-kit run ledger

RULE: label/fixture/data-file edits force a NEW run designation. Never extend or
compare metrics across a label or fixture edit under the same run id.
Rows are append-only; DISCARD/KILLED/STOPPED_COST rows stay (negative results are evidence).

This file is YOURS. `compile`, `eval`, `consistency` and `integrity` append a row
here in whatever directory you run them from. It ships empty on purpose — the run
history that matters is the one measured against your data, on your model, on your
dates. A row copied in from someone else's holdout tells you nothing about yours.

| Run id | Date | Command | Checkpoint sha | Labels/data file | Key metrics | Status |
|---|---|---|---|---|---|---|
