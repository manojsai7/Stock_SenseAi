import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, Input

def build_lstm_model(
    input_shape: tuple = (30, 6),
    units: int = 50,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001
) -> Sequential:
    """
    TensorFlow / Keras Standard LSTM Neural Network Benchmark:
    Input -> LSTM -> Dropout -> Dense (LeakyReLU) -> Dense(1)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units),
        LeakyReLU(negative_slope=0.1),
        Dense(1)
    ], name="TensorFlow_LSTM_Stock_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="huber",
        metrics=["mae", "mse"]
    )
    return model
