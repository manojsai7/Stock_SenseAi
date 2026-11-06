# StockSense AI — Pre-Defense & Release Verification Checklist

**Release Candidate:** `v2.0-rc2` (Sprint 5 Final Capstone Milestone)  
**Target Environment:** Local Defense / Docker Container / Render Backend / Vercel Frontend  
**Verification Date:** November 2026  

---

## 📋 1. Git Repository & Architecture Integrity
- [x] Working branch created and verified (`codex/sprint5-final-release`).
- [x] Working tree clean; zero untracked artifacts polluting tracked source paths.
- [x] Line endings enforced via `.gitattributes` (`*.csv text eol=lf`).
- [x] Dockerfile updated to package `models_storage_v2/` alongside `models_storage_stationary/`.
- [x] Cloud deployment configuration (`render.yaml`) preserves `PREDICTION_PIPELINE_VERSION: stationary-return-v1`.

---

## 🔒 2. Production V1 Protection & Non-Interference
- [x] Production weights file `models_storage_stationary/RELIANCE_NS/best_model.h5` verified untouched.
- [x] Production scaler files (`scaler_features.joblib`, `scaler_target.joblib`) strictly isolated from research modifications.
- [x] Primary production API `/api/predictions/RELIANCE.NS/latest` returns HTTP 200 with verified V1 inference.
- [x] V2 model is **NOT** promoted to production; strictly preserved in shadow mode.

---

## ⚡ 3. Research Challenger V2 (EXP-06) Verification
- [x] Challenger weights file `models_storage_v2/RELIANCE_NS/best_model.keras` verified.
- [x] Dedicated registry `models_storage_v2/registry_v2.json` tracks model hash, training cutoff, and scaler references.
- [x] Shape contract enforced: input tensor strictly verified as `(1, 30, 7)`.
- [x] Symbol whitelist enforced: non-Reliance requests return clean `HTTP 404 Not Found`.
- [x] Hardware disclosure verified: UI and reports transparently state CPU runtime (~12.99s/fit) for exploratory training.
- [x] Google Colab export package `stocksense_exp06_source.zip` preserved for GPU scaling.

---

## 📊 4. Prospective Evaluation Stream (EXP-06)
- [x] Audit log `research/EXP-06/evaluation/prospective_predictions_log.csv` verified clean of test mock data.
- [x] Locked registry `research/EXP-06/evaluation/locked_evaluation_registry_v2.json` active.
- [x] Dynamic target date calculation verified: November 6, 2026 cutoff targeting November 9, 2026.
- [x] Tri-model trio logged for target date November 9, 2026:
  - GRU V2: ₹1,322.86 (+0.2169%, BULLISH)
  - Production V1: ₹1,319.33 (-0.0509%, BEARISH)
  - Naive Persistence: ₹1,320.00 (0.00%, NEUTRAL)
- [x] Idempotency verified: duplicate requests do not generate orphan records.
- [x] Stale data guard verified: returns HTTP 503 if market data is out-of-sync.

---

## 🖥️ 5. Frontend Quality Assurance (React 18 + Vite)
- [x] Dedicated GRU Hero Card renders with `EXPERIMENTAL — SHADOW EVALUATION` badge.
- [x] CPU runtime disclosure clearly displayed on the Hero card.
- [x] 4-Model Comparative Registry renders Production V1, Challenger V2, EXP-05 Colab, and Naive baselines.
- [x] Interactive Recharts trajectory chart displays historical context and forward predictions.
- [x] Production build (`npm run build`) succeeds cleanly with 0 errors in 34.25s (3,185 modules transformed).

---

## 🧪 6. Automated Regression Test Suite
- [x] Complete test suite executed via `pytest`: **196 / 196 tests passing (100% Green)**.
- [x] `tests/test_v2_serving_and_prospective.py`: 11 passed.
- [x] `tests/test_exp06_readiness.py`: 15 passed.
- [x] `tests/test_research_v2_registry.py`: 16 passed.
- [x] Core stationarity and production tests: 154 passed.
- [x] Zero regressions introduced to existing functionality.

---

## 📚 7. Capstone Defense & Documentation Package
- [x] Comprehensive research report finalized: [`research/FINAL_RESEARCH_REPORT.md`](file:///d:/Downloads/Stock_SenseAi/research/FINAL_RESEARCH_REPORT.md).
- [x] Viva question bank finalized: [`research/VIVA_QUESTION_BANK.md`](file:///d:/Downloads/Stock_SenseAi/research/VIVA_QUESTION_BANK.md) (52 technical Q&As).
- [x] PowerPoint slide deck created: [`research/presentation/StockSense_AI_Defense.pptx`](file:///d:/Downloads/Stock_SenseAi/research/presentation/StockSense_AI_Defense.pptx) (12 widescreen slides).
- [x] Markdown slide transcript created: [`research/presentation/SLIDES.md`](file:///d:/Downloads/Stock_SenseAi/research/presentation/SLIDES.md).
- [x] Live demonstration script finalized: [`research/DEMO_SCRIPT.md`](file:///d:/Downloads/Stock_SenseAi/research/DEMO_SCRIPT.md).
- [x] Master project completion report: [`research/PROJECT_COMPLETION_REPORT.md`](file:///d:/Downloads/Stock_SenseAi/research/PROJECT_COMPLETION_REPORT.md).
- [x] Main repository documentation updated: [`README.md`](file:///d:/Downloads/Stock_SenseAi/README.md).

---

*All checklist items verified and approved for production release and capstone defense.*
