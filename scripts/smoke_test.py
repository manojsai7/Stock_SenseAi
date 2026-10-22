#!/usr/bin/env python3
"""
StockSense AI — Production Deployment Smoke Test Suite
Can be executed against localhost or any live production deployment URL:
Usage:
    python scripts/smoke_test.py --url https://stocksense-ai.onrender.com
    python scripts/smoke_test.py --url http://localhost:8000
"""

import sys
import time
import argparse
import httpx

STOCKS = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

def run_smoke_test(base_url: str):
    base_url = base_url.rstrip("/")
    print("\n=======================================================")
    print("  STOCKSENSE AI — PRODUCTION SMOKE TEST")
    print(f"  Target: {base_url}")
    print("=======================================================\n")

    passed = 0
    total = 0

    with httpx.Client(timeout=30.0) as client:
        # 0. Root / and Frontend static serving
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and ("StockSense" in r.text or "<!DOCTYPE html>" in r.text or "html" in r.text):
                print(f"[PASS] GET / ({duration}ms) — Frontend HTML served successfully")
                passed += 1
            else:
                print(f"[FAIL] GET / status={r.status_code}")
        except Exception as e:
            print(f"[FAIL] GET / error: {e}")

        # 0b. OpenAPI /docs
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/docs")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and "swagger" in r.text.lower():
                print(f"[PASS] GET /docs ({duration}ms) — Swagger UI available")
                passed += 1
            else:
                print(f"[FAIL] GET /docs status={r.status_code}")
        except Exception as e:
            print(f"[FAIL] GET /docs error: {e}")

        # 1. Health check
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/api/system/health")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                print(f"[PASS] GET /api/system/health ({duration}ms) — Market Status: {r.json().get('market_status')}")
                passed += 1
            else:
                print(f"[FAIL] GET /api/system/health status={r.status_code}: {r.text}")
        except Exception as e:
            print(f"[FAIL] GET /api/system/health connection error: {e}")

        # 2. Readiness check
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/api/system/ready")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and r.json().get("models_accessible"):
                print(f"[PASS] GET /api/system/ready ({duration}ms) — Status: {r.json().get('status')}")
                passed += 1
            else:
                print(f"[FAIL] GET /api/system/ready status={r.status_code}: {r.text}")
        except Exception as e:
            print(f"[FAIL] GET /api/system/ready error: {e}")

        # 3. Stocks list
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/api/stocks")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and len(r.json()) >= 5:
                print(f"[PASS] GET /api/stocks ({duration}ms) — Listed {len(r.json())} stocks")
                passed += 1
            else:
                print(f"[FAIL] GET /api/stocks status={r.status_code}")
        except Exception as e:
            print(f"[FAIL] GET /api/stocks error: {e}")

        # 3b. Stocks search
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/api/stocks/search?q=TCS")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and len(r.json()) > 0:
                print(f"[PASS] GET /api/stocks/search?q=TCS ({duration}ms) — Found: {len(r.json())} stocks")
                passed += 1
            else:
                print(f"[FAIL] GET /api/stocks/search status={r.status_code}")
        except Exception as e:
            print(f"[FAIL] GET /api/stocks/search error: {e}")

        # 3c. History endpoint test
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/api/stocks/RELIANCE.NS/history?period=1mo")
            duration = round((time.time() - t0) * 1000, 1)
            history_data = r.json().get("data") or r.json().get("history", [])
            if r.status_code == 200 and len(history_data) > 0:
                print(f"[PASS] GET /api/stocks/RELIANCE.NS/history ({duration}ms) — Data points: {len(history_data)}")
                passed += 1
            else:
                print(f"[FAIL] GET /api/stocks/RELIANCE.NS/history status={r.status_code}")
        except Exception as e:
            print(f"[FAIL] GET /api/stocks/RELIANCE.NS/history error: {e}")

        # 4. Benchmark experiments matrix
        total += 1
        t0 = time.time()
        try:
            r = client.get(f"{base_url}/api/benchmark/experiments")
            duration = round((time.time() - t0) * 1000, 1)
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) > 0:
                print(f"[PASS] GET /api/benchmark/experiments ({duration}ms) — Entries: {len(r.json())}")
                passed += 1
            else:
                print(f"[FAIL] GET /api/benchmark/experiments status={r.status_code}")
        except Exception as e:
            print(f"[FAIL] GET /api/benchmark/experiments error: {e}")

        # 5. Test Live Quote and CNN-LSTM Predictions across all 5 stocks
        for sym in STOCKS:
            # Quote
            total += 1
            t0 = time.time()
            try:
                r = client.get(f"{base_url}/api/stocks/{sym}/quote")
                duration = round((time.time() - t0) * 1000, 1)
                if r.status_code == 200:
                    q = r.json()
                    print(f"[PASS] Quote [{sym}] ({duration}ms) — Price: {q.get('price')} {q.get('currency')} (Status: {q.get('market_status')})")
                    passed += 1
                else:
                    print(f"[FAIL] Quote [{sym}] status={r.status_code}")
            except Exception as e:
                print(f"[FAIL] Quote [{sym}] error: {e}")

            # Prediction
            total += 1
            t0 = time.time()
            try:
                r = client.get(
                    f"{base_url}/api/predictions/{sym}/latest"
                    "?pipeline_version=stationary-return-v1"
                )
                duration = round((time.time() - t0) * 1000, 1)
                if r.status_code == 200:
                    pred = r.json()
                    assert pred.get("predicted_close") is not None
                    assert pred.get("pipeline_version") == "stationary-return-v1"
                    assert len(pred.get("features_used", [])) == 7
                    assert pred.get("data_status") in {"ONLINE", "OFFLINE"}
                    assert pred.get("data_as_of_date") == pred.get("latest_completed_session")
                    assert pred.get("data_provider_source")
                    print(
                        f"[PASS] Prediction [{sym}] ({duration}ms) — "
                        f"Data: {pred.get('data_status')} via {pred.get('data_provider_source')} "
                        f"as of {pred.get('data_as_of_date')} | Pred Close: {pred.get('predicted_close')} | "
                        f"Target: {pred.get('target_date')} | Direction: {pred.get('direction')}"
                    )
                    passed += 1
                else:
                    print(f"[FAIL] Prediction [{sym}] status={r.status_code}: {r.text}")
            except Exception as e:
                print(f"[FAIL] Prediction [{sym}] error: {e}")

    print("\n=======================================================")
    print(f"  SMOKE TEST SUMMARY: {passed}/{total} Passed ({round(passed/total*100, 1)}%)")
    print("=======================================================\n")
    return passed == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockSense AI Smoke Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of application")
    args = parser.parse_args()

    success = run_smoke_test(args.url)
    sys.exit(0 if success else 1)
