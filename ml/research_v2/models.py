"""Bounded architectures with explicit deterministic initialization."""
def build_model(architecture, seed=42, learning_rate=0.001, loss='mse', units=32, dropout=0.2, l2=0.0001):
    import tensorflow as tf
    if architecture not in ('lstm', 'gru', 'cnn_lstm') or loss not in ('mse', 'huber'):
        raise ValueError('Unsupported architecture or loss')
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    layers = tf.keras.layers
    regularizer = tf.keras.regularizers.l2(l2)
    model = tf.keras.Sequential([layers.Input((30, 7))])
    if architecture == 'cnn_lstm':
        model.add(layers.Conv1D(32, 3, padding='causal', activation='relu', kernel_regularizer=regularizer))
        model.add(layers.MaxPooling1D(2))
    recurrent = layers.GRU if architecture == 'gru' else layers.LSTM
    model.add(recurrent(units, kernel_regularizer=regularizer))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
                  loss=tf.keras.losses.Huber() if loss == 'huber' else 'mse')
    return model

def callbacks(checkpoint, patience=8):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN
    return [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', patience=max(2, patience//2), factor=0.5),
            ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True), TerminateOnNaN()]
