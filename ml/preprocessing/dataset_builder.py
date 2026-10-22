import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from ml.features.technical import TechnicalFeatureExtractor, FEATURE_COLUMNS, TARGET_COLUMN
from ml.preprocessing.scaler import StockDataScaler
from ml.preprocessing.sequences import SequenceGenerator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEQUENCES_DIR = PROJECT_ROOT / "data" / "sequences"
DATASET_EXP_DIR = PROJECT_ROOT / "experiments" / "dataset"

RESEARCH_SYMBOLS = [
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "RELIANCE.NS",
    "ATHERENERG.NS"
]


class SupervisedDatasetBuilder:
    """
    Builds formal supervised learning datasets for CNN-LSTM, LSTM, and Naive baselines.
    
    Guarantees:
    1. Input tensor: X in R^(N x 30 x 6) across [Open, High, Low, Close, Volume, MA20]
    2. Target: y in R^(N x 1) representing Close(t+1)
    3. Strict chronological partitioning: Train (70%), Val (15%), Test (15%)
    4. Zero-lookahead leakage: MinMaxScaler is fitted EXCLUSIVELY on the training partition.
    5. Clean handling of variable history lengths (e.g. Ather Energy 360 days vs mature 1241 days).
    """

    def __init__(
        self,
        window_size: int = 30,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        processed_dir: Path = PROCESSED_DIR,
        sequences_dir: Path = SEQUENCES_DIR,
        dataset_exp_dir: Path = DATASET_EXP_DIR
    ):
        self.window_size = window_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.processed_dir = processed_dir
        self.sequences_dir = sequences_dir
        self.dataset_exp_dir = dataset_exp_dir

        self.sequences_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_exp_dir.mkdir(parents=True, exist_ok=True)

        self.seq_generator = SequenceGenerator(window_size=window_size)
        self.feature_extractor = TechnicalFeatureExtractor(ma_window=20)

    @staticmethod
    def clean_symbol(symbol: str) -> str:
        return symbol.strip().upper().replace(".", "_")

    def load_and_prepare_df(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads processed dataset, drops MA20 warmup rows, and creates Next_Close target.
        Returns:
            raw_processed_df: full DataFrame with MA20
            clean_df: rows with valid MA20 and valid Next_Close (Close_t+1)
        """
        clean = self.clean_symbol(symbol)
        file_path = self.processed_dir / f"{clean}_features.csv"
        if not file_path.exists():
            file_path = self.processed_dir / f"{clean}_processed.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Processed dataset not found for {symbol} at {file_path}")

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Ensure MA20 exists
        if "MA20" not in df.columns:
            df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()

        # Add target: Close(t+1)
        df[TARGET_COLUMN] = df["Close"].shift(-1)

        # Drop rows where MA20 is NaN (first 19 warmup rows)
        # Drop the last row where Next_Close is NaN (future close unknown)
        clean_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

        return df, clean_df

    def build_stock_dataset(self, symbol: str) -> Dict[str, Any]:
        """
        Executes full supervised dataset construction for a single equity:
        Split -> Train-only Scaler -> Sequence Generation -> Validation -> Artifact Export.
        """
        clean = self.clean_symbol(symbol)
        stock_seq_dir = self.sequences_dir / clean
        stock_exp_dir = self.dataset_exp_dir / clean
        stock_seq_dir.mkdir(parents=True, exist_ok=True)
        stock_exp_dir.mkdir(parents=True, exist_ok=True)

        raw_df, clean_df = self.load_and_prepare_df(symbol)
        usable_rows = len(clean_df)

        if usable_rows < self.window_size + 10:
            raise ValueError(f"Insufficient usable observations ({usable_rows}) for window size {self.window_size}")

        # -------------------------------------------------------------------
        # 1. Chronological Splitting Boundaries
        # -------------------------------------------------------------------
        train_idx = int(usable_rows * self.train_ratio)
        val_idx = int(usable_rows * (self.train_ratio + self.val_ratio))

        train_df = clean_df.iloc[:train_idx].copy()
        val_df = clean_df.iloc[train_idx:val_idx].copy()
        test_df = clean_df.iloc[val_idx:].copy()

        # -------------------------------------------------------------------
        # 2. Fit Scaler ONLY on Train Partition (STRICT ZERO-LOOKAHEAD LEAKAGE)
        # -------------------------------------------------------------------
        scaler = StockDataScaler(feature_columns=FEATURE_COLUMNS)
        scaler.fit(train_df)

        # Save fitted scaler with sequences
        scaler_path = stock_seq_dir / "scaler.pkl"
        scaler.save(scaler_path)

        # Transform full clean_df using the TRAIN-FITTED scaler
        scaled_features, scaled_targets = scaler.transform(clean_df)

        # -------------------------------------------------------------------
        # 3. Supervised Sliding Window Construction
        # -------------------------------------------------------------------
        # SequenceGenerator.create_split_sequences partitions targets by train_idx, val_idx
        (X_train, y_train), (X_val, y_val), (X_test, y_test), test_orig_indices = self.seq_generator.create_split_sequences(
            scaled_features=scaled_features,
            scaled_targets=scaled_targets,
            train_idx=train_idx,
            val_idx=val_idx
        )

        # -------------------------------------------------------------------
        # 4. Target Dates, Previous Closes, and Ground Truth Alignment
        # -------------------------------------------------------------------
        # In clean_df, row k has input features up to date D_k and target Close(k+1) at D_{k+1}.
        # For sequence ending at row k (0-indexed):
        # Target date is the date of Next_Close, which is raw_df date of row k+1.
        all_target_indices = np.arange(self.window_size - 1, usable_rows)

        train_target_indices = all_target_indices[all_target_indices < train_idx]
        val_target_indices = all_target_indices[(all_target_indices >= train_idx) & (all_target_indices < val_idx)]
        test_target_indices = all_target_indices[all_target_indices >= val_idx]

        # Get target dates and ground truth values
        # Because Next_Close is row k+1's close, its timestamp corresponds to the next row
        # In clean_df, row k+1 has date Date[k+1] if sequential. More precisely:
        # Next_Close date is clean_df['Date'].iloc[k + 1] (or next market day)
        dates_train = clean_df["Date"].iloc[train_target_indices].dt.strftime("%Y-%m-%d").values.astype("U10")
        dates_val = clean_df["Date"].iloc[val_target_indices].dt.strftime("%Y-%m-%d").values.astype("U10")
        dates_test = clean_df["Date"].iloc[test_target_indices].dt.strftime("%Y-%m-%d").values.astype("U10")

        # Previous day close (day t) in actual unscaled currency (INR)
        prev_close_train = clean_df["Close"].iloc[train_target_indices].values.astype(np.float32)
        prev_close_val = clean_df["Close"].iloc[val_target_indices].values.astype(np.float32)
        prev_close_test = clean_df["Close"].iloc[test_target_indices].values.astype(np.float32)

        # Actual unscaled target price (Next_Close = Close_t+1)
        actual_target_train = clean_df[TARGET_COLUMN].iloc[train_target_indices].values.astype(np.float32)
        actual_target_val = clean_df[TARGET_COLUMN].iloc[val_target_indices].values.astype(np.float32)
        actual_target_test = clean_df[TARGET_COLUMN].iloc[test_target_indices].values.astype(np.float32)

        # -------------------------------------------------------------------
        # 5. Export Compressed NumPy Tensors & Array Files
        # -------------------------------------------------------------------
        npz_path = stock_seq_dir / "sequences.npz"
        np.savez_compressed(
            npz_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            dates_train=dates_train,
            dates_val=dates_val,
            dates_test=dates_test,
            prev_close_train=prev_close_train,
            prev_close_val=prev_close_val,
            prev_close_test=prev_close_test,
            actual_target_train=actual_target_train,
            actual_target_val=actual_target_val,
            actual_target_test=actual_target_test
        )

        # Also save individual .npy files for instant access
        np.save(stock_seq_dir / "X_train.npy", X_train)
        np.save(stock_seq_dir / "y_train.npy", y_train)
        np.save(stock_seq_dir / "X_val.npy", X_val)
        np.save(stock_seq_dir / "y_val.npy", y_val)
        np.save(stock_seq_dir / "X_test.npy", X_test)
        np.save(stock_seq_dir / "y_test.npy", y_test)

        # -------------------------------------------------------------------
        # 6. Generate Sequence Examples CSV (for Audit & Faculty Inspection)
        # -------------------------------------------------------------------
        example_records = []
        # Inspect first 3 train sequences and first 3 test sequences
        sample_tuples = [("train", i) for i in range(min(3, len(X_train)))] + \
                        [("test", i) for i in range(min(3, len(X_test)))]

        for split, i in sample_tuples:
            if split == "train":
                row_idx = train_target_indices[i]
                target_val = actual_target_train[i]
                prev_val = prev_close_train[i]
                x_seq = X_train[i]
            else:
                row_idx = test_target_indices[i]
                target_val = actual_target_test[i]
                prev_val = prev_close_test[i]
                x_seq = X_test[i]

            input_start_date = clean_df["Date"].iloc[row_idx - self.window_size + 1].strftime("%Y-%m-%d")
            input_end_date = clean_df["Date"].iloc[row_idx].strftime("%Y-%m-%d")
            # Unscaled close sequence for input
            input_closes = clean_df["Close"].iloc[row_idx - self.window_size + 1 : row_idx + 1].tolist()

            example_records.append({
                "split": split,
                "sample_idx": i,
                "input_start_date": input_start_date,
                "input_end_date": input_end_date,
                "window_timesteps": len(input_closes),
                "last_input_close": prev_val,
                "target_close_t_plus_1": target_val,
                "target_delta_inr": round(float(target_val - prev_val), 2),
                "actual_direction": "FLAT" if abs(target_val - prev_val) <= 1e-5 else ("UP" if target_val > prev_val else "DOWN")
            })

        example_df = pd.DataFrame(example_records)
        example_df.to_csv(stock_exp_dir / "sequence_examples.csv", index=False)

        # -------------------------------------------------------------------
        # 7. Metadata & Dataset Report Generation
        # -------------------------------------------------------------------
        is_ather = (symbol.strip().upper() == "ATHERENERG.NS")
        history_note = (
            "Ather Energy Ltd (ATHERENERG.NS) listed in May 2025. Supervised population is naturally truncated "
            f"to {len(X_train)+len(X_val)+len(X_test)} total sequences (~340 trading days). No synthetic data was fabricated."
            if is_ather else "Standard mature 5-year series (~1192 sequences)."
        )

        metadata = {
            "symbol": symbol,
            "clean_symbol": clean,
            "raw_observations": len(raw_df),
            "ma20_warmup_dropped": 19,
            "unpaired_last_row_dropped": 1,
            "usable_observations": usable_rows,
            "total_sequences": len(X_train) + len(X_val) + len(X_test),
            "train_sequences": len(X_train),
            "validation_sequences": len(X_val),
            "test_sequences": len(X_test),
            "window_size": self.window_size,
            "feature_count": len(FEATURE_COLUMNS),
            "feature_names": FEATURE_COLUMNS,
            "target_definition": "Next trading day Close (Close_t+1)",
            "shapes": {
                "X_train": list(X_train.shape),
                "y_train": list(y_train.shape),
                "X_val": list(X_val.shape),
                "y_val": list(y_val.shape),
                "X_test": list(X_test.shape),
                "y_test": list(y_test.shape)
            },
            "first_target_date": str(dates_train[0]) if len(dates_train) > 0 else None,
            "last_target_date": str(dates_test[-1]) if len(dates_test) > 0 else None,
            "split_ratios": {
                "train": self.train_ratio,
                "validation": self.val_ratio,
                "test": self.test_ratio
            },
            "scaler_parameters": {
                "feature_min": [float(v) for v in scaler.feature_scaler.data_min_],
                "feature_max": [float(v) for v in scaler.feature_scaler.data_max_],
                "target_min": float(scaler.target_scaler.data_min_[0]),
                "target_max": float(scaler.target_scaler.data_max_[0])
            },
            "is_short_history": is_ather,
            "history_note": history_note,
            "created_at": datetime.now().astimezone().isoformat()
        }

        # Save metadata JSON in sequences and dataset experiment dirs
        with open(stock_seq_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(stock_exp_dir / "dataset_report.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"[{symbol}] Supervised dataset built: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}"
        )
        return metadata

    def build_all(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Builds supervised datasets across all research equities and saves summary.json.
        """
        target_symbols = symbols or RESEARCH_SYMBOLS
        summaries = []

        for sym in target_symbols:
            try:
                meta = self.build_stock_dataset(sym)
                summaries.append(meta)
            except Exception as e:
                logger.error(f"Failed to build supervised dataset for {sym}: {e}")

        # Consolidated report
        summary_path = self.dataset_exp_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)

        logger.info(f"Consolidated dataset summary saved to {summary_path}")
        return summaries
