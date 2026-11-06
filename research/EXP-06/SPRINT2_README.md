# Sprint 2 data admission workflow

Run from the repository root. Original source files, V1 artifacts and EXP-05 evaluation records must remain intact. Each output directory must be new.

```powershell
python scripts/audit_exp06_data.py --input research/EXP-06/data/exp05_original_processed.csv --output research/EXP-06/data/audited_original_v1
python scripts/audit_exp06_data.py --input data/processed/RELIANCE_NS_processed.csv --output research/EXP-06/data/audited_current_v1
```

These commands produce auditable candidates, not verified training data. Each rejected row preserves its source row identifier and reasons. CSV is the canonical export; Parquet is optional and unnecessary for admission.

For verified admission, supply a complete authoritative session CSV plus a hash-linked evidence JSON using the CLI's `--calendar` and `--evidence` arguments. Evidence must establish source hash, retrieval time, price adjustment policy, provider finalization and full calendar coverage. An exchange holiday list for one year cannot establish every session across a five-year snapshot. A passed feature-equivalence check cannot establish data-provider finalization.

The holdout locking tool freezes configuration, dataset, manifest, calendar, preprocessing and preregistration hashes, and exact partition timestamps. It rejects candidates or insufficient holdouts. It never changes the production registry or trains a model. See `HOLDOUT_DESIGN.md`, `EXPERIMENT_PREREGISTRATION.md`, and `TRAINING_GATE_REPORT.md`.

The existing default training gate remains closed. Recovered historical rows are evidence for reproducing EXP-05 only; their future validity requires the same source checks as any other data.
