import numpy as np
import pandas as pd
import pytest
from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS, TARGET_COLUMN
from ml.preprocessing.sequences import SequenceGenerator

def test_ma20_calculation():
    extractor = TechnicalFeatureExtractor(ma_window=20)
    # Generate 50 daily sample candles
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    closes = np.arange(100.0, 150.0)
    df = pd.DataFrame({
        "Date": dates,
        "Open": closes - 1.0,
        "High": closes + 2.0,
        "Low": closes - 2.0,
        "Close": closes,
        "Volume": np.full(50, 1000000),
    })

    featured = extractor.add_features(df)

    # First 19 rows of MA20 must be NaN
    assert featured["MA20"].iloc[:19].isna().all()
    # 20th row (index 19) must be the exact average of first 20 closes
    expected_ma20_at_20 = np.mean(closes[:20])
    assert np.isclose(featured["MA20"].iloc[19], expected_ma20_at_20)

    # Prepare inference features drops NaN rows
    inference_df = extractor.prepare_inference_features(df)
    assert len(inference_df) == 50 - 19  # 31 rows
    assert not inference_df["MA20"].isna().any()

def test_sequence_generator_shape():
    generator = SequenceGenerator(window_size=30)
    # 35 rows of scaled features (6 features)
    features = np.random.rand(35, 6).astype(np.float32)
    targets = np.random.rand(35, 1).astype(np.float32)

    X, y = generator.create_sequences(features, targets)
    # (35 - 30 + 1) = 6 sequences
    assert X.shape == (6, 30, 6)
    assert y.shape == (6,)

def test_inference_sequence_shape():
    generator = SequenceGenerator(window_size=30)
    features = np.random.rand(30, 6).astype(np.float32)
    seq = generator.create_inference_sequence(features)
    assert seq.shape == (1, 30, 6)

def test_insufficient_sequence_data_raises():
    generator = SequenceGenerator(window_size=30)
    features = np.random.rand(25, 6).astype(np.float32)
    with pytest.raises(ValueError, match="Need at least 30"):
        generator.create_inference_sequence(features)
