"""
Smoke-test script for Phase 3 API versioned inference:
Tests legacy and stationary-return-v1 pipelines on FastAPI endpoints.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from backend.main import app
from backend.core.config import settings
from backend.core.cache import cache

def run_smoke_tests():
    client = TestClient(app)
    cache.clear()
    print("=" * 60)
    print("STOCKSENSE AI — PHASE 3 API SMOKE TEST")
    print("=" * 60)

    # 1. Test Legacy Pipeline (Explicit query parameter)
    print("\n[1] Testing Legacy Pipeline (Explicit ?pipeline_version=legacy)...")
    res_legacy = client.get("/api/predict/TCS.NS?pipeline_version=legacy")
    assert res_legacy.status_code == 200, f"Legacy request failed: {res_legacy.status_code} {res_legacy.text}"
    leg_data = res_legacy.json()
    print(f"    Status: {res_legacy.status_code}")
    print(f"    Symbol: {leg_data['symbol']}")
    print(f"    Model Version: {leg_data['model_version']}")
    print(f"    Features Count: {len(leg_data['features_used'])}")
    print(f"    Predicted Close: {leg_data['predicted_close']} INR")
    print(f"    Direction: {leg_data['direction']}")
    assert len(leg_data["features_used"]) == 6
    assert leg_data["predicted_close"] > 0
    print("    --> Legacy inference verified successfully.")

    # 2. Test Stationary Pipeline (Explicit query parameter)
    print("\n[2] Testing Stationary Pipeline (Explicit ?pipeline_version=stationary-return-v1)...")
    res_stationary = client.get("/api/predict/TCS.NS?pipeline_version=stationary-return-v1")
    assert res_stationary.status_code == 200, f"Stationary request failed: {res_stationary.status_code} {res_stationary.text}"
    stat_data = res_stationary.json()
    print(f"    Status: {res_stationary.status_code}")
    print(f"    Symbol: {stat_data['symbol']}")
    print(f"    Model Version: {stat_data['model_version']}")
    print(f"    Features Count: {len(stat_data['features_used'])}")
    print(f"    Features: {stat_data['features_used']}")
    print(f"    Last Close: {stat_data['last_close']} INR")
    print(f"    Predicted Close: {stat_data['predicted_close']} INR")
    print(f"    Expected Change: {stat_data['expected_change']} INR ({stat_data['expected_change_pct']}%)")
    print(f"    Direction: {stat_data['direction']}")
    print(f"    MAE Benchmark: {stat_data['mae_benchmark']} (Honest null)")
    assert stat_data["model_version"] == "cnn_lstm_stationary_v1"
    assert len(stat_data["features_used"]) == 7
    assert stat_data["mae_benchmark"] is None
    assert stat_data["predicted_close"] > 0
    print("    --> Stationary-return inference verified successfully.")

    # 3. Test Default Pipeline (no param -> default 'legacy')
    print("\n[3] Testing Default Pipeline Config (No query param -> default 'legacy')...")
    res_default = client.get("/api/predict/TCS.NS")
    assert res_default.status_code == 200
    def_data = res_default.json()
    assert len(def_data["features_used"]) == 6
    print(f"    Default pipeline version: {settings.PREDICTION_PIPELINE_VERSION}")
    print(f"    Resolved model features count: {len(def_data['features_used'])} (legacy default confirmed)")

    # 4. Test Unsupported Stock Ather Energy on stationary pipeline
    print("\n[4] Testing Unsupported Stock ATHERENERG.NS on stationary pipeline...")
    res_ather = client.get("/api/predict/ATHERENERG.NS?pipeline_version=stationary-return-v1")
    print(f"    Status: {res_ather.status_code}")
    print(f"    Detail: {res_ather.json()['detail']}")
    assert res_ather.status_code == 404
    assert "Ather Energy has insufficient historical trading sessions" in res_ather.json()["detail"]
    print("    --> Unsupported symbol correctly rejected with 404.")

    # 5. Test Invalid Pipeline Version
    print("\n[5] Testing Invalid Pipeline Version rejection...")
    res_invalid = client.get("/api/predict/TCS.NS?pipeline_version=invalid-experimental-v9")
    print(f"    Status: {res_invalid.status_code}")
    print(f"    Detail: {res_invalid.json()['detail']}")
    assert res_invalid.status_code == 400
    assert "Invalid prediction pipeline version" in res_invalid.json()["detail"]
    print("    --> Invalid pipeline version rejected with 400.")

    # 6. Test Environment Variable Switch to stationary-return-v1
    print("\n[6] Testing Environment Variable Override (settings.PREDICTION_PIPELINE_VERSION = 'stationary-return-v1')...")
    settings.PREDICTION_PIPELINE_VERSION = "stationary-return-v1"
    res_env_stat = client.get("/api/predict/TCS.NS")
    assert res_env_stat.status_code == 200
    assert res_env_stat.json()["model_version"] == "cnn_lstm_stationary_v1"
    assert len(res_env_stat.json()["features_used"]) == 7
    print("    --> Env switch to 'stationary-return-v1' verified.")

    # Revert to safe default legacy
    settings.PREDICTION_PIPELINE_VERSION = "legacy"
    res_reverted = client.get("/api/predict/TCS.NS")
    assert len(res_reverted.json()["features_used"]) == 6
    print("    --> Safely reverted default to 'legacy'.")

    # 7. Test /api/predictions/{symbol}/latest endpoint
    print("\n[7] Testing /api/predictions/{symbol}/latest with pipeline_version...")
    res_latest = client.get("/api/predictions/RELIANCE.NS/latest?pipeline_version=stationary-return-v1")
    assert res_latest.status_code == 200
    assert res_latest.json()["model_version"] == "cnn_lstm_stationary_v1"
    print("    --> /api/predictions/{symbol}/latest successfully tested.")

    print("\n" + "=" * 60)
    print("ALL API SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_tests()
