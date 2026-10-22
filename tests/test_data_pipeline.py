import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml.data.acquisition import ResearchDataAcquisition, RESEARCH_SYMBOLS
from backend.core.config import settings


@pytest.fixture
def sample_clean_df():
    """Generates 40 rows of clean, synthetic daily OHLCV data."""
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(40)]
    closes = [100.0 + i * 0.5 for i in range(40)]
    opens = [c - 0.2 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [o - 0.5 for o in opens]
    volumes = [100000 + i * 500 for i in range(40)]

    return pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes
    })


def test_atherenerg_symbol_support():
    """Verifies that ATHERENERG.NS is recognized as a supported research stock."""
    assert "ATHERENERG.NS" in RESEARCH_SYMBOLS
    assert "ATHERENERG.NS" in settings.SUPPORTED_STOCKS
    assert any(s["symbol"] == "ATHERENERG.NS" for s in settings.DEFAULT_STOCKS)


def test_ohlc_validation_valid_data(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    val = acq.validate_dataset(sample_clean_df, "RELIANCE.NS")

    assert val["is_valid"] is True
    assert val["missing_values"] == 0
    assert val["duplicate_rows"] == 0
    assert val["invalid_ohlc_rows"] == 0
    assert val["is_chronologically_sorted"] is True
    assert len(val["issues"]) == 0


def test_ohlc_validation_detects_invalid_high_low(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    corrupt_df = sample_clean_df.copy()
    # Violate High >= Low on row 5
    corrupt_df.loc[5, "High"] = corrupt_df.loc[5, "Low"] - 10.0

    val = acq.validate_dataset(corrupt_df, "TCS.NS")
    assert val["is_valid"] is False
    assert val["invalid_ohlc_rows"] >= 1
    assert any("High < Low" in issue for issue in val["issues"])


def test_ohlc_validation_detects_low_above_open(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    corrupt_df = sample_clean_df.copy()
    # Violate Low <= Open
    corrupt_df.loc[3, "Low"] = corrupt_df.loc[3, "Open"] + 5.0

    val = acq.validate_dataset(corrupt_df, "INFY.NS")
    assert val["is_valid"] is False
    assert any("Low price above Open" in issue for issue in val["issues"])


def test_duplicate_date_detection(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    dup_df = sample_clean_df.copy()
    dup_df.loc[10, "Date"] = dup_df.loc[9, "Date"]

    val = acq.validate_dataset(dup_df, "HDFCBANK.NS")
    assert val["is_valid"] is False
    assert val["duplicate_rows"] == 1
    assert any("duplicate timestamps" in issue for issue in val["issues"])


def test_missing_values_detection(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    nan_df = sample_clean_df.copy()
    nan_df.loc[2, "Close"] = np.nan

    val = acq.validate_dataset(nan_df, "ICICIBANK.NS")
    assert val["is_valid"] is False
    assert val["missing_values"] == 1


def test_chronological_ordering_detection(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    shuffled_df = sample_clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val = acq.validate_dataset(shuffled_df, "TCS.NS")
    assert val["is_chronologically_sorted"] is False
    assert val["is_valid"] is False


def test_ma20_calculation_correctness(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    processed = acq.compute_processed_features(sample_clean_df, "TCS.NS")

    assert "MA20" in processed.columns
    # First 19 entries must be NaN for 20-period moving average
    assert processed["MA20"].iloc[:19].isnull().all()
    # 20th entry (index 19) must equal the average of the first 20 closes
    expected_ma20_idx19 = sample_clean_df["Close"].iloc[:20].mean()
    assert np.isclose(processed["MA20"].iloc[19], expected_ma20_idx19)

    # 21st entry (index 20) must equal the average of rows 1..20
    expected_ma20_idx20 = sample_clean_df["Close"].iloc[1:21].mean()
    assert np.isclose(processed["MA20"].iloc[20], expected_ma20_idx20)


def test_manifest_generation_schema(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    val = acq.validate_dataset(sample_clean_df, "ATHERENERG.NS")
    manifest = acq.create_manifest(sample_clean_df, "ATHERENERG.NS", val)

    assert manifest["symbol"] == "ATHERENERG.NS"
    assert manifest["source"] == "Yahoo Finance"
    assert manifest["provider_package"] == "yfinance"
    assert manifest["interval"] == "1d"
    assert "retrieval_timestamp" in manifest
    assert "start_date" in manifest
    assert "end_date" in manifest
    assert manifest["row_count"] == len(sample_clean_df)
    assert "MA20" not in manifest["columns"]  # Raw schema
    assert "validation" in manifest
    assert "history_limitation_note" in manifest
    assert "Ather Energy" in manifest["history_limitation_note"]


def test_raw_snapshot_immutability(sample_clean_df, tmp_path):
    acq = ResearchDataAcquisition(raw_dir=tmp_path / "raw", manifests_dir=tmp_path / "manifests", processed_dir=tmp_path / "processed")
    # First save
    path1, is_new1 = acq.save_raw_snapshot(sample_clean_df, "RELIANCE.NS", force=False)
    assert is_new1 is True
    assert path1.exists()

    # Second save without force -> should not overwrite
    modified_df = sample_clean_df.copy()
    modified_df.loc[0, "Close"] = 99999.0
    path2, is_new2 = acq.save_raw_snapshot(modified_df, "RELIANCE.NS", force=False)
    assert is_new2 is False

    # Check file on disk still contains original value
    disk_df = pd.read_csv(path1)
    assert disk_df.loc[0, "Close"] == sample_clean_df.loc[0, "Close"]
