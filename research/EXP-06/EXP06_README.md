# EXP-06 independent research engine

Sprint 1 implements code and validation, not a trained challenger. V1 remains unchanged. Training is blocked by default. The Colab notebook contains no executed outputs.

Shared importable Python code lives in `ml/research_v2/`, rather than under the non-importable `EXP-06` directory name. This directory holds experiment configuration, provenance references, protocols, notebook and artifacts. Tests live in the existing root `tests/` suite.

## Implemented

- `pipeline.py`: strict OHLCV/date/MA20 validation, canonical V1 feature reuse, 30-session sequences with explicit input/target timestamps, expanding chronological folds, separate post-EXP-05 holdout, train-only feature and target scalers.
- `models.py`: bounded LSTM, GRU and CNN-LSTM; seeded initialization, MSE/Huber, L2, dropout, clipped Adam, early stopping, learning-rate reduction, best checkpoints.
- `evaluation.py`: equivalent naive/zero-return forecast and Ridge, regression and directional metrics, confusion matrix, volatility ratio.
- `runner.py`: gate and hash checks, fold-specific fitting, predictions, histories, environment/configuration metadata, model/scaler manifests. It does not select a winner or score the holdout.
- `backend/services/research_v2_registry.py`: independent unavailable-by-default registry, integrity and contract checks, own scalers, return-to-price inference. It is not wired to live routes. Callers must supply trusted completed-session and forecast dates; this is not a market-data acquisition service.

Read `SELECTION_PROTOCOL.md` before registering an experiment. Current configuration has an unapproved gate, no approved dataset hash and no holdout boundary. These are intentional blockers, not runnable defaults for market training. Completing the gate requires evidence; changing booleans is not verification.

## Local verification

From the repository root:

```powershell
python -m pytest tests/test_research_v2_engine.py tests/test_research_v2_registry.py -q
python scripts/package_exp06_source.py
```

After dataset approval and configuration completion:

```powershell
python scripts/run_exp06.py --config research/EXP-06/config/experiment_config.json --dataset verified_RELIANCE_NS.csv --output research/EXP-06/artifacts/run_001
```

Run directories must be new. Keep actual training on Colab GPU as requested; local fitting tests use synthetic data only.

## Colab execution

1. Run `python scripts/package_exp06_source.py` locally. This creates `scratch/stocksense_exp06_source.zip` containing shared source, config, provenance and source checksums; no production models or credentials.
2. Open `notebooks/StockSense_EXP06_Colab.ipynb` in Colab and choose **Runtime > Change runtime type > GPU**.
3. Run the upload cell and select the source ZIP. Run dependency and environment cells; a missing GPU stops execution.
4. Once the data gate passes, upload the reviewed CSV named `verified_RELIANCE_NS.csv` and the completed config named `experiment_config.json`. The notebook verifies its SHA-256. The referenced provenance report is included in the source ZIP.
5. Run the shared experiment engine. It validates the gate and reserves the untouched holdout before fitting. It writes development-fold artifacts only.
6. Download `EXP06_colab_artifacts.zip` from the final cell. Preserve it together with the exact input CSV, configuration and source ZIP.

The notebook was structurally checked locally, not executed on Colab. Runtime package installation and GPU availability remain to be verified there. No paid compute is authorized.

## Still outside Sprint 1

Verified training dataset and sufficient uninspected holdout observations; real GPU experiments; uncertainty analysis; controlled final holdout evaluation and selection; shadow registry registration; prospective logger; API/UI integration; deployment. Archived EXP-05 reconstruction does not prove valid source sessions or predictive superiority.
