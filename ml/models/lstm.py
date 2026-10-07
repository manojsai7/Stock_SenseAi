import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU, Input

def build_lstm_model(
    input_shape: tuple = (30, 6),
    units: int = 50,
    dense_units: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    loss: str = "mse"
) -> Sequential:
    """
    TensorFlow / Keras Standard LSTM Neural Network Benchmark:
    Input -> LSTM -> Dropout -> Dense (LeakyReLU) -> Dense(1)
    """
    model = Sequential([
        Input(shape=input_shape, name="input_window"),
        LSTM(units, return_sequences=False, name="lstm"),
        Dropout(dropout, name="dropout"),
        Dense(dense_units, name="dense_proj"),
        LeakyReLU(negative_slope=0.1, name="leaky_relu"),
        Dense(1, name="dense_output")
    ], name="TensorFlow_LSTM_Stock_Model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["mae", "mse"]
    )
    return model
