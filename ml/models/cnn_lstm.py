import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, LeakyReLU, Input

def build_cnn_lstm_model(
    input_shape: tuple = (30, 6),
    conv_filters: int = 64,
    kernel_size: int = 3,
    lstm_units: int = 50,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    loss: str = "mse"
) -> Sequential:
    """
    TensorFlow / Keras Hybrid CNN-LSTM Neural Network:
    1. Input layer: (window_size, num_features)
    2. Conv1D: Extracts local temporal patterns (moving averages, short-term volatility swings).
    3. MaxPooling1D: Filters noise and condenses temporal dimensions.
    4. LSTM: Learns sequential long-range dependencies across trading intervals.
    5. Dropout: Regularization to prevent overfitting on historical noise.
    6. Dense + LeakyReLU: Non-saturating projection allowing regime shifts.
    7. Dense(1): Pure linear output predicting next trading day's closing price.
    """
    model = Sequential([
        Input(shape=input_shape, name="input_window"),
        Conv1D(
            filters=conv_filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            name="conv1d"
        ),
        MaxPooling1D(pool_size=2, name="maxpool1d"),
        LSTM(lstm_units, return_sequences=False, name="lstm"),
        Dropout(dropout, name="dropout"),
        Dense(dense_units, name="dense_proj"),
        LeakyReLU(negative_slope=0.1, name="leaky_relu"),
        Dense(1, name="dense_output")
    ], name="TensorFlow_CNN_LSTM_Stock_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["mae", "mse"]
    )
    return model
