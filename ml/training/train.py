import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from tensorflow.keras.callbacks import EarlyStopping

from ml.data.ingestion import DataIngestionService, DEFAULT_TICKERS
from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS
from ml.preprocessing.scaler import StockDataScaler
from ml.preprocessing.sequences import SequenceGenerator
from ml.models.baseline import NaivePersistenceModel
from ml.models.lstm import build_lstm_model
from ml.models.cnn_lstm import build_cnn_lstm_model
from ml.evaluation.metrics import calculate_metrics

# Suppress excessive tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models_storage")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class ModelTrainingPipeline:
    def __init__(self, window_size: int = 30, epochs: int = 40, batch_size: int = 32):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_extractor = TechnicalFeatureExtractor(ma_window=20)
        self.seq_generator = SequenceGenerator(window_size=window_size)
        self.ingestion = DataIngestionService()

    def train_stock(self, symbol: str, period: str = "5y") -> Dict[str, Any]:
        logger.info(f"========== Training Pipeline Started for {symbol} ==========")
        clean_symbol = symbol.replace(".", "_")
        stock_model_dir = MODELS_DIR / clean_symbol
        stock_model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Fetch & clean raw data
        raw_df = self.ingestion.get_stock_data(symbol, period=period)
        if len(raw_df) < 200:
            raise ValueError(f"Insufficient historical data for {symbol}: {len(raw_df)} rows")

        # 2. Feature Extraction
        featured_df = self.feature_extractor.prepare_training_features(raw_df)
        total_rows = len(featured_df)
        logger.info(f"Feature dataset prepared: {total_rows} rows with features {FEATURE_COLUMNS}")

        # 3. Chronological split (70% train, 15% val, 15% test) - STRICT NO LEAKAGE
        train_idx = int(total_rows * 0.70)
        val_idx = int(total_rows * 0.85)

        train_df = featured_df.iloc[:train_idx].copy()
        val_df = featured_df.iloc[train_idx:val_idx].copy()
        test_df = featured_df.iloc[val_idx:].copy()

        logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # 4. Scaling (Fitted exclusively on train_df)
        scaler = StockDataScaler()
        scaler.fit(train_df)
        scaler_path = stock_model_dir / "scaler.pkl"
        scaler.save(scaler_path)

        train_feat_sc, train_tgt_sc = scaler.transform(train_df)
        val_feat_sc, val_tgt_sc = scaler.transform(val_df)
        test_feat_sc, test_tgt_sc = scaler.transform(test_df)

        # 5. Sequence generation
        X_train, y_train = self.seq_generator.create_sequences(train_feat_sc, train_tgt_sc)
        X_val, y_val = self.seq_generator.create_sequences(val_feat_sc, val_tgt_sc)
        X_test, y_test = self.seq_generator.create_sequences(test_feat_sc, test_tgt_sc)

        logger.info(f"Sequence tensors: X_train shape={X_train.shape}, X_test shape={X_test.shape}")

        # Ground truth test values in actual currency (INR)
        actual_test_prices = scaler.inverse_transform_target(y_test)
        # Previous day close for directional accuracy
        # In test_df, the test sequence i corresponds to test_df row window_size - 1 + i
        close_idx_in_features = FEATURE_COLUMNS.index("Close")
        prev_close_sc = X_test[:, -1, close_idx_in_features]
        # Invert feature scaler close column or transform target
        prev_close_prices = scaler.target_scaler.inverse_transform(prev_close_sc.reshape(-1, 1)).flatten()

        test_dates = test_df["Date"].iloc[self.window_size - 1 : self.window_size - 1 + len(y_test)].dt.strftime("%Y-%m-%d").tolist()

        # 6. Benchmark Model 1: Naive Persistence Model
        logger.info("Evaluating Benchmark Model 1: Naive Persistence...")
        naive_model = NaivePersistenceModel(close_col_index=close_idx_in_features)
        naive_pred_sc = naive_model.predict(X_test)
        naive_pred_prices = scaler.target_scaler.inverse_transform(naive_pred_sc.reshape(-1, 1)).flatten()
        naive_metrics = calculate_metrics(actual_test_prices, naive_pred_prices, prev_close_prices)
        logger.info(f"Naive Metrics: {naive_metrics}")

        # 7. Benchmark Model 2: Standard LSTM
        logger.info("Training Benchmark Model 2: Standard LSTM...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        lstm_model = build_lstm_model(input_shape=input_shape)
        
        callbacks_lstm = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        ]
        lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_lstm,
            verbose=0
        )
        lstm_path = stock_model_dir / "model_lstm.keras"
        lstm_model.save(lstm_path)

        lstm_pred_sc = lstm_model.predict(X_test, verbose=0).flatten()
        lstm_pred_prices = scaler.inverse_transform_target(lstm_pred_sc)
        lstm_metrics = calculate_metrics(actual_test_prices, lstm_pred_prices, prev_close_prices)
        logger.info(f"LSTM Metrics: {lstm_metrics}")

        # 8. Target Model: Hybrid CNN-LSTM
        logger.info("Training Target Model: Hybrid CNN-LSTM...")
        cnn_lstm_model = build_cnn_lstm_model(input_shape=input_shape)
        callbacks_cnn_lstm = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        ]
        history = cnn_lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_cnn_lstm,
            verbose=0
        )
        cnn_lstm_path = stock_model_dir / "model_cnn_lstm.keras"
        cnn_lstm_model.save(cnn_lstm_path)

        cnn_lstm_pred_sc = cnn_lstm_model.predict(X_test, verbose=0).flatten()
        cnn_lstm_pred_prices = scaler.inverse_transform_target(cnn_lstm_pred_sc)
        cnn_lstm_metrics = calculate_metrics(actual_test_prices, cnn_lstm_pred_prices, prev_close_prices)
        logger.info(f"CNN-LSTM Metrics: {cnn_lstm_metrics}")

        # 9. Prepare test sample trajectory for UI visualization (last 60 test points)
        sample_slice = slice(-60, None) if len(actual_test_prices) > 60 else slice(0, None)
        trajectory = []
        for d, act, n_p, l_p, c_p in zip(
            test_dates[sample_slice],
            actual_test_prices[sample_slice],
            naive_pred_prices[sample_slice],
            lstm_pred_prices[sample_slice],
            cnn_lstm_pred_prices[sample_slice]
        ):
            trajectory.append({
                "date": d,
                "actual": round(float(act), 2),
                "naive": round(float(n_p), 2),
                "lstm": round(float(l_p), 2),
                "cnn_lstm": round(float(c_p), 2)
            })

        # 10. Save metadata
        metadata = {
            "symbol": symbol,
            "trained_at": datetime.utcnow().isoformat(),
            "model_version": "v1.0",
            "window_size": self.window_size,
            "prediction_horizon": "1 trading day ahead",
            "features": FEATURE_COLUMNS,
            "architecture": {
                "name": "CNN-LSTM Hybrid",
                "conv1d_filters": 64,
                "kernel_size": 3,
                "max_pooling": 2,
                "lstm_units": 50,
                "dropout": 0.2,
                "dense_units": 32,
                "output_units": 1
            },
            "dataset_summary": {
                "total_samples": total_rows,
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
            },
            "metrics_comparison": {
                "naive": naive_metrics,
                "lstm": lstm_metrics,
                "cnn_lstm": cnn_lstm_metrics
            },
            "test_trajectory": trajectory
        }

        meta_path = stock_model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model and metadata successfully saved to {stock_model_dir}")
        return metadata

    def train_all(self, symbols: List[str] = DEFAULT_TICKERS) -> Dict[str, Any]:
        results = {}
        for sym in symbols:
            try:
                results[sym] = self.train_stock(sym)
            except Exception as e:
                logger.error(f"Failed to train for {sym}: {e}", exc_info=True)
                results[sym] = {"error": str(e)}
        return results

if __name__ == "__main__":
    pipeline = ModelTrainingPipeline(window_size=30, epochs=30, batch_size=32)
    # Train primary demonstration stock TCS.NS first
    pipeline.train_stock("TCS.NS")
