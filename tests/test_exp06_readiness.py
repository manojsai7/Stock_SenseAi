"""All fixtures in these tests are synthetic and exercise audit decisions only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.research_v2.pipeline import fingerprint
from ml.research_v2.readiness import audit_dataset


def synthetic_snapshot(directory):
    dates = pd.bdate_range("2025-01-02", periods=80).strftime("%Y-%m-%d")
    close = 80 + np.sin(np.arange(80) / 3)
    frame = pd.DataFrame({"Date": dates, "Open": close - 0.2, "High": close+0.3,
                          "Low": close-0.4, "Close": close, "Volume": 10000+np.arange(80)})
    source = directory/"raw.csv"
    frame.to_csv(source, index=False)
    calendar = directory/"calendar.csv"
    frame[["Date"]].to_csv(calendar, index=False)
    receipt = directory/"receipt.txt"
    receipt.write_text("Synthetic provider fixture for tests only.")
    attachment = {"path":receipt.name,"sha256":fingerprint(receipt)}
    evidence = dict(symbol="RELIANCE.NS", raw_sha256=fingerprint(source), source="synthetic-provider",
                    calendar_authority="synthetic-calendar", calendar_sha256=fingerprint(calendar),
                    calendar_coverage_start=dates[0], calendar_coverage_end=dates[-1],
                    finalized_through=dates[-1], retrieved_at=dates[-1]+"T16:00:00+05:30",
                    adjustment_policy="unadjusted_ohlcv", adjustment_evidence=attachment,
                    finalization_evidence=attachment)
    return frame, source, calendar, evidence


def run_with_evidence(directory, source, calendar, evidence):
    manifest = directory/"source.json"
    manifest.write_text(json.dumps(evidence))
    return audit_dataset(source, directory/"output", calendar_path=calendar, evidence_path=manifest)


def test_verified_synthetic_source_and_calendar(tmp_path):
    frame, source, calendar, evidence = synthetic_snapshot(tmp_path)
    report = run_with_evidence(tmp_path, source, calendar, evidence)
    assert report["status"] == "verified"
    assert report["blockers"] == []
    assert report["candidate_rows"] == len(frame)
    assert report["feature_parity"]["passed"]
    assert report["calendar_sha256"] == fingerprint(calendar)
    assert report["dataset_sha256"] == fingerprint(tmp_path/"output"/"candidate.csv")
    with pytest.raises(FileExistsError):
        audit_dataset(source,tmp_path/"output")


@pytest.mark.parametrize("broken", ["calendar_bounds","calendar_gap","calendar_hash","provider_hash",
                                    "adjustment_hash","raw_hash","clock_only"])
def test_provenance_fails_closed(tmp_path, broken):
    frame, source, calendar, evidence = synthetic_snapshot(tmp_path)
    if broken == "calendar_bounds":
        evidence["calendar_coverage_start"] = frame.Date.iloc[2]
    elif broken == "calendar_gap":
        source_frame = frame.drop(17)
        source_frame.to_csv(source,index=False)
        evidence["raw_sha256"] = fingerprint(source)
    elif broken == "calendar_hash":
        evidence["calendar_sha256"] = "bad"
    elif broken == "provider_hash":
        evidence["finalization_evidence"] = {"path":"receipt.txt","sha256":"bad"}
    elif broken == "adjustment_hash":
        evidence["adjustment_evidence"] = {"path":"absent","sha256":"bad"}
    elif broken == "raw_hash":
        evidence["raw_sha256"] = "bad"
    else:
        del evidence["finalized_through"]
    report = run_with_evidence(tmp_path,source,calendar,evidence)
    assert report["status"] == "candidate"
    assert report["blockers"]
    if broken == "calendar_gap":
        assert report["missing_sessions"] == [frame.Date.iloc[17]]


def test_quarantine_retains_source_row_ids_and_values(tmp_path):
    frame,source,_,_ = synthetic_snapshot(tmp_path)
    frame.loc[3,"Date"] = frame.loc[2,"Date"]
    frame.loc[4,"Low"] = 999
    frame.loc[5,"Volume"] = 0
    frame.loc[6,"Close"] = np.nan
    frame.loc[7,"Date"] = "bad-date"
    frame.loc[8,"Open"] = np.inf
    frame.to_csv(source,index=False)
    original = source.read_bytes()
    a = audit_dataset(source,tmp_path/"a")
    b = audit_dataset(source,tmp_path/"b")
    assert a["dataset_sha256"] == b["dataset_sha256"]
    assert a["quarantine_sha256"] == b["quarantine_sha256"]
    assert a["source_rows"] == a["candidate_rows"] + a["quarantined_rows"]
    rejected = pd.read_csv(tmp_path/"a"/"quarantine.csv").set_index("source_row_id")
    for identifier,reason in [(3,"duplicate_session_date"),(4,"duplicate_session_date"),
                              (5,"invalid_ohlc"),(6,"zero_volume"),(7,"missing_or_nonfinite_ohlcv"),
                              (8,"invalid_session_date"),(9,"missing_or_nonfinite_ohlcv"),
                              (len(frame),"latest_bar_finalization_unconfirmed")]:
        assert reason in rejected.loc[identifier,"reasons"]
    assert source.read_bytes() == original


def test_empty_input_with_schema(tmp_path):
    data,source,_,_ = synthetic_snapshot(tmp_path)
    data.iloc[:0].to_csv(source,index=False)
    report = audit_dataset(source,tmp_path/"empty")
    assert report["candidate_rows"] == report["quarantined_rows"] == 0
    assert report["status"] == "candidate"
    assert report["missing_sessions"] is None


def test_verified_candidate_dataset_and_manifest():
    manifest_path = Path("research/EXP-06/data/dataset_manifest.json")
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_path = Path("research/EXP-06/data/RELIANCE_NS_verified_v1.csv")
    parquet_path = Path("research/EXP-06/data/RELIANCE_NS_verified_v1.parquet")
    assert csv_path.is_file()
    assert parquet_path.is_file()
    assert manifest["dataset_sha256"] == fingerprint(csv_path)
    assert manifest["parquet_sha256"] == fingerprint(parquet_path)
    assert manifest["number_of_records"] == 1233
    assert manifest["quarantined_records"] == 8
    assert manifest["feature_parity"]["passed"] is True
    assert manifest["price_adjustment_policy"] == "consistently_adjusted_ohlcv"


def test_training_gate_decision_fails_closed():
    gate_path = Path("research/EXP-06/TRAINING_GATE_REPORT.md")
    assert gate_path.is_file()
    content = gate_path.read_text(encoding="utf-8").strip()
    assert content.endswith("NO-GO — TRAINING BLOCKED")
    assert "Condition 8" in content


def test_experiment_lock_fails_on_candidate_status():
    from ml.research_v2.experiment_lock import lock_contents
    with pytest.raises(ValueError, match="Dataset readiness manifest has not verified RELIANCE.NS"):
        lock_contents(
            "research/EXP-06/config/experiment_config.json",
            "research/EXP-06/data/RELIANCE_NS_verified_v1.csv",
            "research/EXP-06/data/dataset_manifest.json",
            "research/EXP-06/data/audited_original_v1/candidate.csv"
        )

