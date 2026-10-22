import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Set non-interactive backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EDA_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class StockEDAPipeline:
    """
    Automated Exploratory Data Analysis (EDA) Pipeline for StockSense AI.
    Generates academic-grade visualizations, time series characterization,
    statistical summaries, and empirical data reports for every supported equity.
    """

    def __init__(
        self,
        output_dir: Path = EDA_OUTPUT_DIR,
        processed_dir: Path = PROCESSED_DIR,
        raw_dir: Path = RAW_DIR
    ):
        self.output_dir = output_dir
        self.processed_dir = processed_dir
        self.raw_dir = raw_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Matplotlib style configuration
        sns.set_theme(style="darkgrid")
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 10

    @staticmethod
    def clean_symbol(symbol: str) -> str:
        return symbol.strip().upper().replace(".", "_")

    def load_stock_df(self, symbol: str) -> pd.DataFrame:
        clean = self.clean_symbol(symbol)
        file_path = self.processed_dir / f"{clean}_features.csv"
        if not file_path.exists():
            file_path = self.processed_dir / f"{clean}_processed.csv"
        if not file_path.exists():
            file_path = self.raw_dir / f"{symbol}.csv"
        if not file_path.exists():
            file_path = self.raw_dir / f"{clean}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"No processed or raw dataset found for {symbol} at {file_path}")

        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # Ensure MA20 exists
        if "MA20" not in df.columns:
            df["MA20"] = df["Close"].rolling(window=20, min_periods=20).mean()

        # Compute daily returns
        df["Return"] = df["Close"].pct_change()
        # Compute 20-day rolling volatility (std of daily returns)
        df["Rolling_Volatility"] = df["Return"].rolling(window=20, min_periods=20).std()
        return df

    def generate_stock_eda(self, symbol: str) -> Dict[str, Any]:
        """
        Runs comprehensive EDA for a single stock and saves all visualizations and metrics.
        """
        clean = self.clean_symbol(symbol)
        stock_eda_dir = self.output_dir / clean
        stock_eda_dir.mkdir(parents=True, exist_ok=True)

        df = self.load_stock_df(symbol)
        dates = df["Date"]

        is_ather = (symbol.strip().upper() == "ATHERENERG.NS")
        data_limitations = (
            "Listed on NSE in May 2025. Data population restricted to post-listing sessions (~360 trading days). "
            "Shorter historical horizon than mature peers; zero synthetic rows fabricated."
            if is_ather else "Standard mature historical equity series (5 years)."
        )

        # -------------------------------------------------------------------
        # 1. Close-Price Time Series (price_history.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.plot(dates, df["Close"], color="#2563eb", linewidth=1.8, label=f"{symbol} Close Price")
        ax.set_title(f"{symbol} — Historical Daily Close Price", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price (INR)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "price_history.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 2. Volume Time Series (volume_history.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.bar(dates, df["Volume"], color="#64748b", alpha=0.6, width=1.0, label="Daily Volume")
        vol_ma20 = df["Volume"].rolling(20, min_periods=20).mean()
        ax.plot(dates, vol_ma20, color="#f59e0b", linewidth=1.5, label="20-Day Volume MA")
        ax.set_title(f"{symbol} — Daily Trading Volume & 20-Day MA", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Volume (Shares)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "volume_history.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 3. Daily Returns (daily_returns.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(dates, df["Return"] * 100, color="#0ea5e9", linewidth=1.0, alpha=0.85, label="Daily Return (%)")
        ax.axhline(0, color="#ef4444", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(f"{symbol} — Daily Percentage Returns", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Daily Return (%)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "daily_returns.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 4. Return Distribution (returns_distribution.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        clean_returns = (df["Return"].dropna() * 100).values
        sns.histplot(clean_returns, kde=True, ax=ax, color="#6366f1", bins=40, stat="density", alpha=0.6)
        # Normal distribution reference
        mu, std = np.mean(clean_returns), np.std(clean_returns)
        x_norm = np.linspace(clean_returns.min(), clean_returns.max(), 200)
        p_norm = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std) ** 2)
        ax.plot(x_norm, p_norm, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Normal Fit (μ={mu:.2f}%, σ={std:.2f}%)")
        ax.set_title(f"{symbol} — Empirical Daily Return Distribution", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Daily Return (%)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "returns_distribution.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 5. Rolling 20-Day Volatility (rolling_volatility.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        ax.plot(dates, df["Rolling_Volatility"] * 100, color="#ec4899", linewidth=1.5, label="20-Day Rolling Volatility (%)")
        mean_vol = (df["Rolling_Volatility"] * 100).mean()
        ax.axhline(mean_vol, color="#475569", linestyle=":", linewidth=1.2, label=f"Mean Volatility ({mean_vol:.2f}%)")
        ax.set_title(f"{symbol} — 20-Day Rolling Daily Return Volatility", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Volatility (%)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "rolling_volatility.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 6. Close vs MA20 (close_vs_ma20.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.plot(dates, df["Close"], color="#1e293b", linewidth=1.5, label="Close Price")
        ax.plot(dates, df["MA20"], color="#10b981", linewidth=1.8, label="20-Day SMA (MA20)")
        ax.set_title(f"{symbol} — Close Price vs. 20-Day Moving Average", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price (INR)", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "close_vs_ma20.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 7. OHLC Correlation Matrix (correlation_heatmap.png)
        # -------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
        corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "MA20", "Return", "Rolling_Volatility"] if c in df.columns]
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax, linewidths=0.5)
        ax.set_title(f"{symbol} — Feature Correlation Matrix", fontsize=12, fontweight="bold", pad=10)
        fig.tight_layout()
        fig.savefig(stock_eda_dir / "correlation_heatmap.png")
        plt.close(fig)

        # -------------------------------------------------------------------
        # 8. Descriptive Statistics (descriptive_stats.json)
        # -------------------------------------------------------------------
        stats_dict = {}
        numeric_cols = ["Open", "High", "Low", "Close", "Volume", "MA20", "Return", "Rolling_Volatility"]
        for col in numeric_cols:
            if col in df.columns:
                series = df[col].dropna()
                stats_dict[col] = {
                    "count": int(series.count()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "p25": float(series.quantile(0.25)),
                    "median": float(series.median()),
                    "p75": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "skewness": float(series.skew()) if len(series) > 2 else 0.0,
                    "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0
                }

        with open(stock_eda_dir / "descriptive_stats.json", "w") as f:
            json.dump(stats_dict, f, indent=2)

        # -------------------------------------------------------------------
        # 9. Data Quality & Limitations Report (data_quality_report.json)
        # -------------------------------------------------------------------
        quality_report = {
            "symbol": symbol,
            "total_rows": len(df),
            "start_date": df["Date"].min().strftime("%Y-%m-%d"),
            "end_date": df["Date"].max().strftime("%Y-%m-%d"),
            "null_values_per_column": {col: int(df[col].isnull().sum()) for col in df.columns},
            "duplicate_dates": int(df.duplicated(subset=["Date"]).sum()),
            "ohlc_logical_errors": int(
                ((df["High"] < df["Low"]) |
                 (df["High"] < df["Open"]) |
                 (df["High"] < df["Close"]) |
                 (df["Low"] > df["Open"]) |
                 (df["Low"] > df["Close"])).sum()
            ),
            "data_limitations": data_limitations
        }

        with open(stock_eda_dir / "data_quality_report.json", "w") as f:
            json.dump(quality_report, f, indent=2)

        # Return single-stock summary for master report
        return {
            "symbol": symbol,
            "clean_symbol": clean,
            "first_date": quality_report["start_date"],
            "last_date": quality_report["end_date"],
            "row_count": len(df),
            "missing_values": int(df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum().sum()),
            "duplicate_rows": quality_report["duplicate_dates"],
            "mean_return": float(df["Return"].mean()),
            "std_return": float(df["Return"].std()),
            "min_return": float(df["Return"].min()),
            "max_return": float(df["Return"].max()),
            "mean_volume": float(df["Volume"].mean()),
            "price_min": float(df["Close"].min()),
            "price_max": float(df["Close"].max()),
            "data_limitations": data_limitations,
            "eda_dir": str(stock_eda_dir)
        }

    def run_all(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Executes EDA for all research symbols and generates the master summary.json.
        """
        target_symbols = symbols or [
            "TCS.NS",
            "INFY.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "RELIANCE.NS",
            "ATHERENERG.NS"
        ]

        summary_records = []
        for sym in target_symbols:
            try:
                rec = self.generate_stock_eda(sym)
                summary_records.append(rec)
                logger.info(f"Generated EDA for {sym}")
            except Exception as e:
                logger.error(f"Failed to generate EDA for {sym}: {e}")

        # Write master summary.json
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_records, f, indent=2)

        logger.info(f"Consolidated EDA summary saved to {summary_path}")
        return summary_records
