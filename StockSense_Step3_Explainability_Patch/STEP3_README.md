# StockSense AI — Step 3: live feature sensitivity panel

This patch adds a read-only, on-demand explanation diagnostic for the frozen `stationary-return-v1` production model. It does not retrain or overwrite model weights.

## What it does

- Adds `GET /api/models/{symbol}/explainability` to the FastAPI backend.
- Fetches daily bars on demand, applies the same latest-completed-session rule and freshness check as the prediction pipeline, and uses the existing stationary model plus its fitted feature/target scalers.
- Predicts a baseline next-session return, then repeats inference 7 times. Each pass sets one standardized feature channel to zero across all 30 input sessions. For a StandardScaler, zero is the reference training mean.
- Returns per-feature changes in predicted return (basis points), rounded predicted close, price difference, and whether the model's direction label changes.
- Adds a responsive frontend panel that displays the exact input window, data provenance, model training cutoff, method, and an explicit non-causal disclaimer.

This is a local model-sensitivity diagnostic, not causal attribution, feature importance across the population, proof of predictive edge, or financial advice. It is not the historical EXP-05 Fold 3 feature-ablation report; it is computed from the current completed input window.

## Apply the patch

1. Make sure your research branch and recent commits are safe. Do not reset or discard work.
2. Extract this ZIP into the repository root. It will add the new backend service, frontend component/CSS, test, and this guide.
3. Apply the small edits to existing files:

   ```powershell
   git apply STEP3_Explainability.patch
   ```

4. Run checks from the project root:

   ```powershell
   python -m py_compile backend\services\explainability_service.py backend\api\prediction_routes.py
   pytest tests\test_explainability_service.py tests\test_feature_ablation_explainability.py tests\test_reliance_head_to_head.py tests\test_forward_evaluation_logger.py -q
   ```

5. Build the frontend:

   ```powershell
   cd frontend
   npm run build
   cd ..
   ```

6. Inspect changes before committing:

   ```powershell
   git diff --check
   git status --short
   git diff -- models_storage_stationary\
   ```

7. Run the app locally, select one of the five stationary-model symbols, and inspect the **Local input sensitivity** panel. It should show a live-computed cutoff date and current provider state. A stale-data response or unsupported symbol should appear as a clear unavailable state rather than fabricating results.

## Caution

- No production model artifacts should change. The `git diff -- models_storage_stationary\` command should show no changes.
- This patch has not been run against your private/local model packages in this environment. Run the above tests and frontend build locally before committing.
- The first prospective prediction for the 2026-10-26 target session remains pending until the completed session's actual close is retrieved and `forward_evaluation_logger.py resolve --symbol RELIANCE.NS` is run. Do not resolve it with an intraday or unfinished candle.
