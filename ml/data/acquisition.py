import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RESEARCH_SYMBOLS = [
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "RELIANCE.NS",
    "ATHERENERG.NS"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFESTS_DIR = DATA_DIR / "manifests"
PROCESSED_DIR = DATA_DIR / "processed"


class ResearchDataAcquisition:
    """
    Acquires, validates, and archives daily OHLCV market datasets for research.
    Enforces immutable raw storage, strict data validation, provenance manifests,
    and chronological feature generation with 20-day Simple Moving Average (MA20).
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        manifests_dir: Path = MANIFESTS_DIR,
        processed_dir: Path = PROCESSED_DIR
    ):
        self.raw_dir = raw_dir
        self.manifests_dir = manifests_dir
        self.processed_dir = processed_dir

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def clean_symbol(symbol: str) -> str:
        return symbol.strip().upper().replace(".", "_")

    def fetch_raw_ohlcv(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """
        Downloads historical daily OHLCV from Yahoo Finance via yfinance.
        For newly listed equities (e.g., ATHERENERG.NS), automatically falls back
        to max available period.
        """
        clean = symbol.strip().upper()
        # Ather Energy listed in May 2025, requesting 5y works or max
        download_period = "max" if clean == "ATHERENERG.NS" else period

        logger.info(f"Downloading daily data for {clean} (period={download_period})...")
        ticker = yf.Ticker(clean)
        df = ticker.history(period=download_period, interval="1d", auto_adjust=False)

        if df is None or df.empty:
            df = yf.download(clean, period=download_period, interval="1d", auto_adjust=False, progress=False)

        if df is None or df.empty:
            raise ValueError(f"No market data returned for '{clean}' from Yahoo Finance.")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Resolve date column
        date_col = None
        for candidate in ["Date", "Datetime", "index"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = df.columns[0]

        df = df.rename(columns={date_col: "Date"})
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        present_cols = [c for c in required_cols if c in df.columns]
        df = df[present_cols]

        # Standardize timezone and datetime
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        # Cast numerics
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort chronologically
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    def validate_dataset(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Validates data integrity:
        - Missing values (Nulls/NaNs)
        - Duplicate dates
        - Invalid OHLC relationships:
            High < Low
            High < Open or High < Close
            Low > Open or Low > Close
            Volume < 0
        - Chronological monotonicity
        """
        issues = []
        missing_count = int(df.isnull().sum().sum())
        duplicate_count = int(df.duplicated(subset=["Date"]).sum())

        # Check OHLC logical integrity
        invalid_high_low = int((df["High"] < df["Low"]).sum())
        invalid_high_open = int((df["High"] < df["Open"]).sum())
        invalid_high_close = int((df["High"] < df["Close"]).sum())
        invalid_low_open = int((df["Low"] > df["Open"]).sum())
        invalid_low_close = int((df["Low"] > df["Close"]).sum())
        invalid_volume = int((df["Volume"] < 0).sum())

        invalid_ohlc_total = (
            invalid_high_low +
            invalid_high_open +
            invalid_high_close +
            invalid_low_open +
            invalid_low_close +
            invalid_volume
        )

        if missing_count > 0:
            issues.append(f"{missing_count} missing values detected.")
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate timestamps detected.")
        if invalid_high_low > 0:
            issues.append(f"{invalid_high_low} rows where High < Low.")
        if invalid_high_open > 0 or invalid_high_close > 0:
            issues.append(f"High price below Open or Close in {invalid_high_open + invalid_high_close} rows.")
        if invalid_low_open > 0 or invalid_low_close > 0:
            issues.append(f"Low price above Open or Close in {invalid_low_open + invalid_low_close} rows.")
        if invalid_volume > 0:
            issues.append(f"Negative volume detected in {invalid_volume} rows.")

        is_monotonic = bool(df["Date"].is_monotonic_increasing)
        if not is_monotonic:
            issues.append("Timestamps are not strictly monotonically increasing.")

        # Special limitation recording for Ather Energy
        is_ather = (symbol.strip().upper() == "ATHERENERG.NS")
        limitation_note = (
            "Ather Energy Ltd (ATHERENERG.NS) was listed on the NSE in May 2025. "
            "Available market history is naturally limited to ~360 trading sessions. "
            "No synthetic historical data was fabricated to maintain strict empirical integrity."
            if is_ather else "Standard mature equity series (~5 years of trading sessions)."
        )

        return {
            "symbol": symbol,
            "rows": len(df),
            "start_date": df["Date"].min().strftime("%Y-%m-%d") if not df.empty else None,
            "end_date": df["Date"].max().strftime("%Y-%m-%d") if not df.empty else None,
            "missing_values": missing_count,
            "duplicate_rows": duplicate_count,
            "invalid_ohlc_rows": invalid_ohlc_total,
            "is_chronologically_sorted": is_monotonic,
            "is_valid": (missing_count == 0 and duplicate_count == 0 and invalid_ohlc_total == 0 and is_monotonic),
            "issues": issues,
            "history_limitation_note": limitation_note
        }

    def save_raw_snapshot(self, df: pd.DataFrame, symbol: str, force: bool = False) -> Tuple[Path, bool]:
        """
        Saves raw immutable snapshot under data/raw/.
        Never silently overwrites unless force=True.
        """
        clean = self.clean_symbol(symbol)
        raw_symbol_file = self.raw_dir / f"{symbol}.csv"
        raw_clean_file = self.raw_dir / f"{clean}.csv"

        already_exists = raw_symbol_file.exists() or raw_clean_file.exists()
        if already_exists and not force:
            logger.info(f"Raw snapshot for {symbol} already exists. Retaining immutable snapshot.")
            target_path = raw_symbol_file if raw_symbol_file.exists() else raw_clean_file
            return target_path, False

        # Save primary raw file with exact symbol and clean alias
        df.to_csv(raw_symbol_file, index=False)
        if raw_symbol_file.name != raw_clean_file.name:
            df.to_csv(raw_clean_file, index=False)

        logger.info(f"Saved raw immutable snapshot for {symbol} to {raw_symbol_file}")
        return raw_symbol_file, True

    def create_manifest(self, df: pd.DataFrame, symbol: str, validation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates and saves a machine-readable dataset provenance manifest.
        """
        clean = self.clean_symbol(symbol)
        now_iso = datetime.now().astimezone().isoformat()

        manifest = {
            "symbol": symbol,
            "clean_symbol": clean,
            "source": "Yahoo Finance",
            "provider_package": "yfinance",
            "interval": "1d",
            "retrieval_timestamp": now_iso,
            "start_date": validation["start_date"],
            "end_date": validation["end_date"],
            "row_count": len(df),
            "columns": list(df.columns),
            "validation": {
                "missing_values": validation["missing_values"],
                "duplicate_rows": validation["duplicate_rows"],
                "invalid_ohlc_rows": validation["invalid_ohlc_rows"],
                "is_chronologically_sorted": validation["is_chronologically_sorted"],
                "is_valid": validation["is_valid"],
                "issues": validation["issues"]
            },
            "history_limitation_note": validation["history_limitation_note"]
        }

        # Save individual manifest
        manifest_file = self.manifests_dir / f"{clean}_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def compute_processed_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Computes 20-day Simple Moving Average (MA20) on Close price.
        Preserves strictly chronological ordering.
        Outputs: Date, Open, High, Low, Close, Volume, MA20.
        """
        data = df.copy()
        data = data.sort_values("Date").reset_index(drop=True)

        # 20-day Simple Moving Average
        data["MA20"] = data["Close"].rolling(window=20, min_periods=20).mean()

        clean = self.clean_symbol(symbol)
        features_path = self.processed_dir / f"{clean}_features.csv"
        processed_path = self.processed_dir / f"{clean}_processed.csv"

        data.to_csv(features_path, index=False)
        data.to_csv(processed_path, index=False)
        logger.info(f"Saved processed features for {symbol} to {features_path}")
        return data

    def acquire_stock(self, symbol: str, force: bool = False) -> Dict[str, Any]:
        """
        Full ingestion pipeline for a single stock:
        Download -> Validate -> Save Raw -> Create Manifest -> Compute Processed Features.
        """
        df = self.fetch_raw_ohlcv(symbol)
        validation = self.validate_dataset(df, symbol)
        raw_path, is_new = self.save_raw_snapshot(df, symbol, force=force)
        manifest = self.create_manifest(df, symbol, validation)
        processed_df = self.compute_processed_features(df, symbol)

        return {
            "symbol": symbol,
            "raw_path": str(raw_path),
            "manifest": manifest,
            "rows": len(df),
            "processed_rows": len(processed_df),
            "start_date": validation["start_date"],
            "end_date": validation["end_date"],
            "is_valid": validation["is_valid"]
        }

    def acquire_all(self, symbols: Optional[List[str]] = None, force: bool = False) -> Dict[str, Any]:
        """
        Runs the acquisition pipeline across all supported research stocks
        and saves a consolidated master manifest.
        """
        target_symbols = symbols or RESEARCH_SYMBOLS
        results = {}
        for sym in target_symbols:
            try:
                res = self.acquire_stock(sym, force=force)
                results[sym] = res
            except Exception as e:
                logger.error(f"Error acquiring dataset for {sym}: {e}")
                results[sym] = {"error": str(e), "is_valid": False}

        # Save consolidated manifest
        master_manifest_path = self.manifests_dir / "manifest.json"
        with open(master_manifest_path, "w") as f:
            json.dump({
                "generated_at": datetime.now().astimezone().isoformat(),
                "total_equities": len(results),
                "equities": results
            }, f, indent=2)

        return results
