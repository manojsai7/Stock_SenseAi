"""Audit a frozen historical CSV. No provider calls, interpolation, or model training."""
from collections import Counter
from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ml.features.stationary import FEATURE_COLUMNS, REQUIRED_BAR_COLUMNS, compute_stationary_features
from ml.research_v2.pipeline import FEATURE_VERSION, fingerprint

SYMBOL = "RELIANCE.NS"
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def _build_metadata():
    return {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": FEATURE_VERSION,
        "preprocessing_sha256": fingerprint(Path(__file__)),
        "feature_names": FEATURE_COLUMNS,
        "feature_definition_source": "ml/features/stationary.py:compute_stationary_features",
        "feature_definition_sha256": fingerprint(Path(__file__).resolve().parents[1]/"features/stationary.py"),
        "target_definition": "next_session_close / input_end_close - 1",
    }


def _iso_date(value):
    if not isinstance(value, str):
        raise ValueError("Expected an ISO session date")
    parsed = date.fromisoformat(value)
    if value != parsed.isoformat():
        raise ValueError("Expected YYYY-MM-DD")
    return value


def _attachment_matches(evidence_path, reference):
    """Evidence paths are confined to the reviewed evidence directory."""
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return False
    root = evidence_path.resolve().parent
    path = (root / reference["path"]).resolve()
    return (path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0
            and fingerprint(path) == reference.get("sha256"))


def _evidence(raw_hash, calendar_path, evidence_path):
    blockers = []
    evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig")) if evidence_path else {}
    if evidence.get("raw_sha256") != raw_hash or evidence.get("symbol") != SYMBOL:
        blockers.append("unbound_source_snapshot")
    if not str(evidence.get("source") or "").strip():
        blockers.append("missing_provider_identity")
    if evidence.get("adjustment_policy") not in {"unadjusted_ohlcv", "consistently_adjusted_ohlcv"}:
        blockers.append("unknown_adjustment_policy")
    if not _attachment_matches(evidence_path, evidence.get("adjustment_evidence")):
        blockers.append("missing_adjustment_evidence")
    cutoff = None
    try:
        cutoff = _iso_date(evidence.get("finalized_through"))
        retrieved = pd.Timestamp(evidence["retrieved_at"])
        if retrieved.tzinfo is None or retrieved.tz_convert("Asia/Kolkata").date().isoformat() < cutoff:
            raise ValueError("Invalid retrieval timestamp")
        if not _attachment_matches(evidence_path, evidence.get("finalization_evidence")):
            raise ValueError("Missing provider evidence")
        if "unbound_source_snapshot" in blockers:
            raise ValueError("Finalization evidence is for another snapshot")
    except (ValueError, KeyError, TypeError):
        cutoff = None
        blockers.append("unverified_provider_finalization")
    sessions = set()
    coverage = None
    if calendar_path is None:
        blockers.append("missing_session_calendar")
    else:
        calendar = pd.read_csv(calendar_path, dtype=str)
        try:
            dates = [_iso_date(value) for value in calendar["Date"]]
            start = _iso_date(evidence.get("calendar_coverage_start"))
            end = _iso_date(evidence.get("calendar_coverage_end"))
            if (not dates or dates != sorted(set(dates)) or not start <= dates[0] <= dates[-1] <= end
                    or not str(evidence.get("calendar_authority") or "").strip()
                    or fingerprint(calendar_path) != evidence.get("calendar_sha256")):
                raise ValueError("Unbound calendar")
            sessions = set(dates)
            coverage = (start, end)
        except (ValueError, KeyError, TypeError):
            blockers.append("unverified_session_calendar")
    return evidence, cutoff, sessions, coverage, blockers


def feature_parity(bars):
    """Recompute seven causal features and the next-row return using plain NumPy slices."""
    if len(bars) < 51:
        return {"passed": False, "reason": "insufficient_rows_for_30_session_supervised_window"}
    reference, columns = compute_stationary_features(bars[REQUIRED_BAR_COLUMNS])
    expected = []
    close, opn, high, low, volume = [bars[col].to_numpy(dtype=float)
                                     for col in ["Close", "Open", "High", "Low", "Volume"]]
    dates = bars.Date.to_numpy()
    for i in range(20, len(close)-1):
        expected.append([close[i]/close[i-k]-1 for k in (1, 5, 20)] + [
            (close[i]-opn[i])/opn[i], (high[i]-low[i])/close[i-1],
            close[i]/close[i-19:i+1].mean()-1, volume[i]/(volume[i-19:i+1].mean()+1e-8),
            close[i+1]/close[i]-1])
    names = list(FEATURE_COLUMNS) + ["target_return"]
    actual = reference[names].to_numpy()
    if columns != FEATURE_COLUMNS or actual.shape != (len(expected), len(names)):
        return {"passed": False, "reason": "feature_order_or_shape_mismatch"}
    alignment = (np.array_equal(reference.Date.dt.strftime("%Y-%m-%d"), dates[20:-1])
                 and np.array_equal(reference.target_date, dates[21:]))
    differences = np.max(np.abs(actual-np.asarray(expected)), axis=0)
    return {"passed": bool(alignment and np.allclose(actual, expected, rtol=1e-10, atol=1e-12)),
            "compared_rows": len(actual), "dates_match": bool(alignment),
            "max_absolute_error": dict(zip(names, differences.tolist()))}


def audit_dataset(input_path, output_dir, *, calendar_path=None, evidence_path=None):
    """Write immutable candidate, quarantine and manifest files into a new directory."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new audit output directory: {output_dir}")
    calendar_path = Path(calendar_path) if calendar_path else None
    evidence_path = Path(evidence_path) if evidence_path else None
    # Parse and hash the very same bytes.
    from io import BytesIO
    snapshot = input_path.read_bytes()
    raw_hash = sha256(snapshot).hexdigest()
    raw = pd.read_csv(BytesIO(snapshot), dtype=str, keep_default_na=False, skip_blank_lines=False)
    missing_columns = sorted(set(REQUIRED_BAR_COLUMNS)-set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing source columns: {missing_columns}")
    evidence, cutoff, sessions, coverage, blockers = _evidence(raw_hash, calendar_path, evidence_path)
    bars = raw[REQUIRED_BAR_COLUMNS].copy()
    for column in PRICE_COLUMNS + ["Volume"]:
        bars[column] = pd.to_numeric(raw[column], errors="coerce")
    dates = pd.to_datetime(bars.Date, format="%Y-%m-%d", errors="coerce")
    valid_date = dates.notna() & dates.dt.strftime("%Y-%m-%d").eq(raw.Date)
    duplicates = bars.Date.duplicated(keep=False)
    valid_dates = bars.loc[valid_date, "Date"]
    if not valid_dates.is_monotonic_increasing:
        blockers.append("source_rows_out_of_order")
    if valid_dates.empty:
        blockers.append("no_valid_session_dates")
    last_date = valid_dates.max() if len(valid_dates) else None
    first_date = valid_dates.min() if len(valid_dates) else None
    reasons = []
    for i, row in bars.iterrows():
        issues = []
        if not valid_date.iloc[i]:
            issues.append("invalid_session_date")
        if duplicates.iloc[i]:
            issues.append("duplicate_session_date")
        if not np.isfinite(row[PRICE_COLUMNS+["Volume"]].to_numpy(dtype=float)).all():
            issues.append("missing_or_nonfinite_ohlcv")
        else:
            if row.Volume == 0:
                issues.append("zero_volume")
            if row.Volume < 0:
                issues.append("negative_volume")
            if (min(row[PRICE_COLUMNS]) <= 0 or row.High < max(row.Open,row.Close,row.Low)
                    or row.Low > min(row.Open,row.Close,row.High)):
                issues.append("invalid_ohlc")
        if valid_date.iloc[i]:
            if cutoff and row.Date > cutoff:
                issues.append("after_finalized_cutoff")
            elif cutoff is None and row.Date == last_date:
                issues.append("latest_bar_finalization_unconfirmed")
            if coverage and coverage[0] <= row.Date <= coverage[1] and row.Date not in sessions:
                issues.append("off_calendar")
        reasons.append(issues)
    counts = Counter(reason for issues in reasons for reason in issues)
    if any(counts[k] for k in ("invalid_session_date","duplicate_session_date","missing_or_nonfinite_ohlcv",
                                "negative_volume","invalid_ohlc")):
        blockers.append("unresolved_malformed_source_rows")
    keep = np.array([not issues for issues in reasons], dtype=bool)
    clean = bars.loc[keep].copy()
    clean.insert(0,"source_row_id",clean.index+1)
    quarantine = raw.loc[~keep, REQUIRED_BAR_COLUMNS].copy()
    quarantine.insert(0,"source_row_id",quarantine.index+1)
    quarantine["reasons"] = ["|".join(issues) for issues in reasons if issues]
    missing_sessions = None
    if coverage and first_date and cutoff:
        if not coverage[0] <= first_date <= cutoff <= coverage[1]:
            blockers.append("calendar_does_not_cover_full_finalized_range")
        expected = {d for d in sessions if first_date <= d <= cutoff}
        missing_sessions = sorted(expected - set(clean.Date))
        if missing_sessions:
            blockers.append("missing_authorized_sessions")
    if any("off_calendar" in issues and "zero_volume" not in issues and
           "after_finalized_cutoff" not in issues for issues in reasons):
        blockers.append("positive_volume_off_calendar")
    parity = feature_parity(clean)
    if not parity["passed"]:
        blockers.append("feature_parity_failed")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir.parent) as stage:
        directory = Path(stage) / "audit"
        directory.mkdir()
        for name, frame in [("candidate", clean), ("quarantine", quarantine)]:
            frame.to_csv(directory/f"{name}.csv", index=False, lineterminator="\n")
        manifest = {
            **_build_metadata(), "schema_version": 1, "symbol": SYMBOL, "status": "candidate" if blockers else "verified",
            "blockers": blockers, "raw_sha256": raw_hash, "dataset_sha256": fingerprint(directory/"candidate.csv"),
            "quarantine_sha256": fingerprint(directory/"quarantine.csv"),
            "calendar_sha256": fingerprint(calendar_path) if calendar_path else None,
            "evidence_sha256": fingerprint(evidence_path) if evidence_path else None,
            "source": evidence.get("source"), "adjustment_policy": evidence.get("adjustment_policy"),
            "retrieved_at": evidence.get("retrieved_at"), "provider_finalized_through": cutoff,
            "source_rows": len(raw), "candidate_rows": len(clean), "quarantined_rows": len(quarantine),
            "first_source_date": first_date, "last_source_date": last_date,
            "first_candidate_date": clean.Date.min() if len(clean) else None,
            "last_candidate_date": clean.Date.max() if len(clean) else None,
            "quarantine_reason_counts": dict(sorted(counts.items())), "missing_sessions": missing_sessions,
            "feature_parity": parity, "source_row_id_definition": "One-based CSV data record after header",
        }
        (directory/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n",encoding="utf-8")
        directory.rename(output_dir)
    return manifest
