import urllib.request
import json

symbols = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
print(f"{'SYMBOL':<12} | {'PIPELINE':<22} | {'LAST DATE':<12} | {'COMPL SESS':<12} | {'LAST CLOSE':<10} | {'TARGET DATE':<12} | {'PRED CLOSE':<10} | {'CHANGE %':<10}")
print("-" * 115)

for sym in symbols:
    leg = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/predict/{sym}?pipeline_version=legacy").read().decode())
    stat = json.loads(urllib.request.urlopen(f"http://127.0.0.1:8000/api/predict/{sym}?pipeline_version=stationary-return-v1").read().decode())
    
    match_date = leg["last_observed_date"] == stat["last_observed_date"]
    match_sess = leg["latest_completed_session"] == stat["latest_completed_session"]
    match_close = abs(leg["last_close"] - stat["last_close"]) < 0.01
    match_target = leg["target_date"] == stat["target_date"]
    
    print(f"{sym:<12} | {'legacy':<22} | {leg['last_observed_date']:^12} | {leg['latest_completed_session']:^12} | {leg['last_close']:>10.2f} | {leg['target_date']:^12} | {leg['predicted_close']:>10.2f} | {leg['expected_change_pct']:>8.2f}%")
    print(f"{sym:<12} | {'stationary-return-v1':<22} | {stat['last_observed_date']:^12} | {stat['latest_completed_session']:^12} | {stat['last_close']:>10.2f} | {stat['target_date']:^12} | {stat['predicted_close']:>10.2f} | {stat['expected_change_pct']:>8.2f}%")
    print(f"   Consistency Check: Date={match_date}, Session={match_sess}, Close={match_close}, Target={match_target}")
    print("-" * 115)
