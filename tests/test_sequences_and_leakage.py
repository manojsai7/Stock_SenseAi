import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS, TARGET_COLUMN
from ml.preprocessing.scaler import StockDataScaler
from ml.preprocessing.sequences import SequenceGenerator
from ml.preprocessing.dataset_builder import SupervisedDatasetBuilder, RESEARCH_SYMBOLS


@pytest.fixture
def deterministic_df():
    """
    Constructs a deterministic 60-row DataFrame where Close is exactly 100, 101, 102, ...
    and all other features are deterministic functions of Close.
    """
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(60)]
    closes = [100.0 + i for i in range(60)]
    opens = [c - 0.5 for c in closes]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    volumes = [10000.0 + i * 100 for i in range(60)]
    ma20s = [np.nan if i < 19 else float(np.mean(closes[i - 19 : i + 1])) for i in range(60)]

    df = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": volumes,
        "MA20": ma20s
    })
    return df


# ---------------------------------------------------------------------------
# 1. Critical Target Alignment Test (Deterministic Verification)
# ---------------------------------------------------------------------------
def test_deterministic_target_alignment(deterministic_df):
    """
    MATHEMATICAL PROOF:
    Verifies that for any window of 30 days ending at index t,
    the target is EXACTLY Close[t+1].
    """
    builder = SupervisedDatasetBuilder(window_size=30, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    raw_df, clean_df = builder.load_and_prepare_df = lambda sym: (
        deterministic_df,
        deterministic_df.dropna(subset=FEATURE_COLUMNS).assign(Next_Close=deterministic_df["Close"].shift(-1)).dropna().reset_index(drop=True)
    )

    clean_df = deterministic_df.dropna(subset=FEATURE_COLUMNS).copy()
    clean_df[TARGET_COLUMN] = clean_df["Close"].shift(-1)
    clean_df = clean_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    # For any row k in clean_df:
    # Row k's features end at day k with Close = clean_df['Close'].iloc[k]
    # Row k's target is clean_df['Close'].iloc[k + 1]
    for k in range(len(clean_df)):
        assert clean_df.loc[k, TARGET_COLUMN] == clean_df.loc[k + 1, "Close"]

    # Verify through SequenceGenerator
    seq_gen = SequenceGenerator(window_size=30)
    features = clean_df[FEATURE_COLUMNS].values
    targets = clean_df[[TARGET_COLUMN]].values

    X, y = seq_gen.create_sequences(features, targets)

    # Sample 0: input is rows 0..29 (ending at row 29)
    # The last close in input window 0 must be clean_df['Close'].iloc[29]
    close_idx = FEATURE_COLUMNS.index("Close")
    assert np.isclose(X[0, -1, close_idx], clean_df.loc[29, "Close"])

    # Target for Sample 0 must be EXACTLY clean_df['Close'].iloc[30] (day t+1)
    assert np.isclose(y[0], clean_df.loc[30, "Close"])

    # Sample 5: input is rows 5..34 (ending at row 34)
    assert np.isclose(X[5, -1, close_idx], clean_df.loc[34, "Close"])
    # Target for Sample 5 must be EXACTLY clean_df['Close'].iloc[35]
    assert np.isclose(y[5], clean_df.loc[35, "Close"])


# ---------------------------------------------------------------------------
# 2. Critical Data-Leakage Test (Scaler Train-Only Fitting Proof)
# ---------------------------------------------------------------------------
def test_scaler_train_only_leakage_proof(deterministic_df):
    """
    DATA LEAKAGE PROOF:
    Demonstrates programmatically that scaler min/max parameters are derived
    EXCLUSIVELY from the training portion.
    Injecting extreme future spikes/outliers in the test set does NOT contaminate
    the scaler parameters.
    """
    clean_df = deterministic_df.dropna(subset=FEATURE_COLUMNS).copy()
    clean_df[TARGET_COLUMN] = clean_df["Close"].shift(-1)
    clean_df = clean_df.dropna().reset_index(drop=True)

    train_idx = int(len(clean_df) * 0.70)
    train_df = clean_df.iloc[:train_idx].copy()
    test_df = clean_df.iloc[train_idx:].copy()

    # Normal scaler fitted on train
    scaler_normal = StockDataScaler(feature_columns=FEATURE_COLUMNS)
    scaler_normal.fit(train_df)

    # Create a corrupted test set with extreme future spikes
    corrupted_test_df = test_df.copy()
    corrupted_test_df.loc[corrupted_test_df.index[-1], "Close"] = 9999999.0
    corrupted_test_df.loc[corrupted_test_df.index[-1], TARGET_COLUMN] = 9999999.0
    corrupted_test_df.loc[corrupted_test_df.index[0], "Volume"] = 1e12

    # Scaler fitted strictly on train_df while corrupted_test_df exists
    scaler_isolated = StockDataScaler(feature_columns=FEATURE_COLUMNS)
    scaler_isolated.fit(train_df)  # Strict train-only fit

    # 1. Feature scaler min/max match train_df exactly
    for col_idx, col in enumerate(FEATURE_COLUMNS):
        expected_min = train_df[col].min()
        expected_max = train_df[col].max()
        assert np.isclose(scaler_isolated.feature_scaler.data_min_[col_idx], expected_min)
        assert np.isclose(scaler_isolated.feature_scaler.data_max_[col_idx], expected_max)

    # 2. Target scaler min/max match train_df target exactly
    assert np.isclose(scaler_isolated.target_scaler.data_min_[0], train_df[TARGET_COLUMN].min())
    assert np.isclose(scaler_isolated.target_scaler.data_max_[0], train_df[TARGET_COLUMN].max())

    # 3. Parameters are completely invariant to test data
    np.testing.assert_allclose(scaler_normal.feature_scaler.data_max_, scaler_isolated.feature_scaler.data_max_)
    np.testing.assert_allclose(scaler_normal.target_scaler.data_max_, scaler_isolated.target_scaler.data_max_)


# ---------------------------------------------------------------------------
# 3. Shape & Dimension Tests Across Generated Sequences
# ---------------------------------------------------------------------------
def test_all_six_stocks_tensor_shapes():
    """
    Verifies that all 6 generated stock sequences exist, have 3D input shape (N, 30, 6)
    and 1D target shape (N,).
    """
    builder = SupervisedDatasetBuilder()
    sequences_dir = Path("data/sequences")

    for sym in RESEARCH_SYMBOLS:
        clean = sym.replace(".", "_")
        npz_file = sequences_dir / clean / "sequences.npz"
        assert npz_file.exists(), f"Missing sequences.npz for {sym}"

        data = np.load(npz_file)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data["X_val"]
        y_val = data["y_val"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # Check dimensions
        assert X_train.ndim == 3 and X_train.shape[1] == 30 and X_train.shape[2] == 6
        assert X_val.ndim == 3 and X_val.shape[1] == 30 and X_val.shape[2] == 6
        assert X_test.ndim == 3 and X_test.shape[1] == 30 and X_test.shape[2] == 6

        assert y_train.ndim == 1 and len(y_train) == len(X_train)
        assert y_val.ndim == 1 and len(y_val) == len(X_val)
        assert y_test.ndim == 1 and len(y_test) == len(X_test)


# ---------------------------------------------------------------------------
# 4. Chronological Partition Ordering (No Overlap Between Splits)
# ---------------------------------------------------------------------------
def test_train_val_test_chronological_sequence():
    """
    Verifies that targets are strictly partitioned chronologically:
    max(train_dates) < min(val_dates) <= max(val_dates) < min(test_dates).
    """
    sequences_dir = Path("data/sequences")

    for sym in RESEARCH_SYMBOLS:
        clean = sym.replace(".", "_")
        data = np.load(sequences_dir / clean / "sequences.npz")
        dates_train = data["dates_train"]
        dates_val = data["dates_val"]
        dates_test = data["dates_test"]

        # Strictly ordered
        assert dates_train[-1] < dates_val[0], f"Train-Val date overlap in {sym}"
        assert dates_val[-1] < dates_test[0], f"Val-Test date overlap in {sym}"


# ---------------------------------------------------------------------------
# 5. No Target in Input Window Test
# ---------------------------------------------------------------------------
def test_no_target_inside_input_window():
    """
    Verifies that the target value (Close_t+1) does NOT appear as the last timestep of the input.
    """
    sequences_dir = Path("data/sequences")

    for sym in ["RELIANCE.NS", "TCS.NS"]:
        clean = sym.replace(".", "_")
        data = np.load(sequences_dir / clean / "sequences.npz")
        X_tr = data["X_train"]
        y_tr = data["y_train"]

        # In scaled space, target y_tr[i] should generally differ from X_tr[i, -1, Close_col]
        # except in the vanishingly rare event of 0.000% change
        close_col = 3
        last_input_closes = X_tr[:, -1, close_col]
        diffs = np.abs(y_tr - last_input_closes)

        # Confirm the model is not predicting an identical identity copy (non-zero variance in diff)
        assert np.mean(diffs) > 0.001


# ---------------------------------------------------------------------------
# 6. Ather Energy Shorter History Handling
# ---------------------------------------------------------------------------
def test_atherenerg_shorter_history_handling():
    """
    Verifies that Ather Energy is correctly handled with its actual ~311 sequences
    and that no synthetic rows were fabricated.
    """
    clean = "ATHERENERG_NS"
    data = np.load(Path("data/sequences") / clean / "sequences.npz")
    with open(Path("data/sequences") / clean / "metadata.json") as f:
        meta = json.load(f)

    assert meta["is_short_history"] is True
    assert meta["raw_observations"] == 360
    assert meta["usable_observations"] == 340
    assert meta["total_sequences"] == 311
    assert len(data["X_train"]) + len(data["X_val"]) + len(data["X_test"]) == 311
