import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.models.lstm import build_lstm_model
from ml.models.baseline import NaivePersistenceModel
from ml.analysis.model_inspector import inspect_model_architecture
from ml.analysis.cnn_instrumentation import CNNInstrumentation
from ml.evaluation.metrics import calculate_metrics, compute_direction_signs, compute_direction_labels


def test_model_input_and_output_shapes():
    """Verifies that CNN-LSTM and LSTM have exact (None, 30, 6) input and (None, 1) output."""
    cnn_lstm = build_cnn_lstm_model(input_shape=(30, 6))
    lstm = build_lstm_model(input_shape=(30, 6))

    assert cnn_lstm.input_shape == (None, 30, 6)
    assert cnn_lstm.output_shape == (None, 1)

    assert lstm.input_shape == (None, 30, 6)
    assert lstm.output_shape == (None, 1)


def test_model_parameter_existence_and_count():
    """Verifies that trainable parameters exist and parameter counts are positive."""
    model = build_cnn_lstm_model(input_shape=(30, 6))
    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_weights)

    assert total_params > 20000
    assert trainable_params == total_params  # All layers trainable by default


def test_forward_pass_finite_predictions():
    """Verifies that forward pass produces finite, non-NaN predictions on random batch."""
    model = build_cnn_lstm_model(input_shape=(30, 6))
    X_dummy = np.random.uniform(0.0, 1.0, size=(8, 30, 6)).astype(np.float32)

    y_pred = model(X_dummy, training=False).numpy()

    assert y_pred.shape == (8, 1)
    assert not np.isnan(y_pred).any()
    assert not np.isinf(y_pred).any()


def test_loss_computation_no_nan():
    """Verifies that loss computation does not return NaN."""
    model = build_cnn_lstm_model(input_shape=(30, 6))
    X_dummy = np.random.uniform(0.0, 1.0, size=(16, 30, 6)).astype(np.float32)
    y_dummy = np.random.uniform(0.0, 1.0, size=(16, 1)).astype(np.float32)

    loss_fn = tf.keras.losses.MeanSquaredError()
    preds = model(X_dummy, training=True)
    loss = loss_fn(y_dummy, preds).numpy()

    assert not np.isnan(loss)
    assert loss >= 0.0


def test_model_architecture_inspection(tmp_path):
    """Verifies that model_inspector generates model summary text."""
    model = build_cnn_lstm_model(input_shape=(30, 6))
    report = inspect_model_architecture(model, symbol="TEST_EQUITY", save_dir=tmp_path)

    assert report["total_params"] > 0
    assert "conv1d" in report["detailed_layer_params"]
    assert "lstm" in report["detailed_layer_params"]
    assert Path(report["report_path"]).exists()


def test_cnn_kernel_and_bias_extraction(tmp_path):
    """Verifies that Conv1D kernels (3, 6, 64) and biases (64,) are extracted correctly."""
    model = build_cnn_lstm_model(input_shape=(30, 6), conv_filters=64, kernel_size=3)
    instrumentor = CNNInstrumentation(output_dir=tmp_path)

    kernels, biases = instrumentor.extract_conv1d_weights(model, symbol="TEST_STOCK")

    assert kernels.shape == (3, 6, 64)
    assert biases.shape == (64,)
    assert (tmp_path / "TEST_STOCK" / "cnn" / "kernels.npy").exists()
    assert (tmp_path / "TEST_STOCK" / "cnn" / "biases.npy").exists()


def test_cnn_activation_extraction(tmp_path):
    """Verifies that Conv1D activations are extracted with shape (30, 64)."""
    model = build_cnn_lstm_model(input_shape=(30, 6), conv_filters=64)
    instrumentor = CNNInstrumentation(output_dir=tmp_path)

    sample_seq = np.random.uniform(0.0, 1.0, size=(30, 6)).astype(np.float32)
    result = instrumentor.extract_activations(model, sample_seq, symbol="TEST_STOCK", sample_index=0)

    assert result["activation_shape"] == [30, 64]
    assert Path(result["csv_path"]).exists()
    assert Path(result["plot_path"]).exists()
    assert len(result["top_filters"]) > 0


def test_directional_accuracy_calculation():
    """Verifies directional accuracy calculation matches 3-class exact hit rate formula."""
    actual = np.array([105.0, 102.0, 110.0, 95.0, 100.0])
    previous = np.array([100.0, 104.0, 105.0, 100.0, 100.0])
    # Directions:
    # 1: 105 > 100  -> UP (+1)
    # 2: 102 < 104  -> DOWN (-1)
    # 3: 110 > 105  -> UP (+1)
    # 4: 95 < 100   -> DOWN (-1)
    # 5: 100 == 100 -> FLAT (0)

    # Perfect prediction
    pred_perfect = np.array([108.0, 101.0, 112.0, 90.0, 100.0])
    m_perfect = calculate_metrics(actual, pred_perfect, previous)
    assert m_perfect["directional_accuracy"] == 100.0

    # 3 of 5 correct (60%)
    pred_partial = np.array([108.0, 106.0, 112.0, 102.0, 100.0])  # UP(hit), UP(miss), UP(hit), UP(miss), FLAT(hit) -> 3/5 = 60%
    m_partial = calculate_metrics(actual, pred_partial, previous)
    assert m_partial["directional_accuracy"] == 60.0


def test_directional_classes_rising_falling_unchanged_near_equal():
    """
    Verifies that directional labeling satisfies:
    1. Rising prices -> UP (+1)
    2. Falling prices -> DOWN (-1)
    3. Exactly unchanged prices -> FLAT (0), NEVER UP
    4. Near-equal prices within tolerance -> FLAT (0), NEVER UP
    """
    previous = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    # Case 1: Rising (105.0, 100.5)
    # Case 2: Falling (95.0, 99.0)
    # Case 3: Exactly unchanged (100.0)
    # Case 4: Near-equal within atol=1e-5 (100.000005, 99.999995)
    prices = np.array([105.0, 95.0, 100.0, 100.000005, 99.999995, 100.5])

    signs = compute_direction_signs(prices, previous, atol=1e-5)
    labels = compute_direction_labels(prices, previous, atol=1e-5)

    assert signs[0] == 1 and labels[0] == "UP"         # Rising
    assert signs[1] == -1 and labels[1] == "DOWN"      # Falling
    assert signs[2] == 0 and labels[2] == "FLAT"       # Exactly unchanged -> FLAT, never UP
    assert signs[3] == 0 and labels[3] == "FLAT"       # Near-equal (+5e-6) -> FLAT, never UP
    assert signs[4] == 0 and labels[4] == "FLAT"       # Near-equal (-5e-6) -> FLAT, never UP
    assert signs[5] == 1 and labels[5] == "UP"         # Rising (+0.5) -> UP

    # Explicit assertion: Equal price must never be classified as UP
    equal_price = np.array([100.0])
    assert compute_direction_signs(equal_price, np.array([100.0]))[0] != 1
    assert compute_direction_labels(equal_price, np.array([100.0]))[0] != "UP"
    assert compute_direction_labels(equal_price, np.array([100.0]))[0] == "FLAT"


def test_naive_persistence_directional_predictions():
    """
    Verifies NaivePersistenceModel forecasts FLAT direction for all timesteps
    since PredictedClose[t+1] == Close[t] identically.
    """
    model = NaivePersistenceModel(close_col_index=3)
    X = np.random.uniform(100.0, 200.0, size=(10, 30, 6)).astype(np.float32)

    dirs = model.predict_direction(X)
    assert len(dirs) == 10
    assert np.all(dirs == "FLAT")
    assert "UP" not in dirs


def test_checkpoint_callback_creation(tmp_path):
    """Verifies that ModelCheckpoint saves a valid .keras artifact."""
    model = build_cnn_lstm_model(input_shape=(30, 6))
    ckpt_path = tmp_path / "test_model.keras"

    X = np.random.uniform(0.0, 1.0, size=(32, 30, 6)).astype(np.float32)
    y = np.random.uniform(0.0, 1.0, size=(32, 1)).astype(np.float32)

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        save_best_only=False
    )
    model.fit(X, y, epochs=1, callbacks=[callback], verbose=0)

    assert ckpt_path.exists()
    loaded = tf.keras.models.load_model(ckpt_path)
    assert loaded is not None
