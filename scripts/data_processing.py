from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PANEL_PATH = PROJECT_DIR / "data" / "raw" / "stocks_panel_daily_qfq_baostock.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
CLEAN_DATA_PATH = PROCESSED_DIR / "clean_daily_data.csv"

STANDARD_COLS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_change",
    "turnover",
]


def load_panel(path: str | Path = RAW_PANEL_PATH) -> pd.DataFrame:
    """Load a raw daily stock panel and normalize basic dtypes."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return standardize_schema(df)


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, dates, tickers, numeric fields, and ordering."""
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {
        "code": "ticker",
        "symbol": "ticker",
        "trade_date": "date",
        "pctchg": "pct_change",
        "pct_chg": "pct_change",
        "turn": "turnover",
    }
    df = df.rename(columns=rename_map)

    missing = [col for col in STANDARD_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[STANDARD_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.extract(r"(\d{6})", expand=False)
    df["ticker"] = df["ticker"].str.zfill(6)

    numeric_cols = [col for col in STANDARD_COLS if col not in {"date", "ticker"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.drop_duplicates(["ticker", "date"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def clean_daily_panel(
    df: pd.DataFrame,
    min_obs_ratio: float = 0.8,
    start_date: str | None = None,
    end_date: str | None = None,
    min_price: float = 1e-6,
) -> pd.DataFrame:
    """Clean daily A-share panel data for factor research."""
    df = standardize_schema(df)

    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df[df["close"] > min_price].copy()
    for col in ["open", "high", "low", "volume", "amount", "turnover"]:
        if col in df.columns:
            df = df[df[col].notna()]

    obs_count = df.groupby("ticker")["date"].nunique()
    if obs_count.empty:
        raise ValueError("No observations remain after basic cleaning.")

    min_obs = int(np.ceil(obs_count.max() * min_obs_ratio))
    valid_tickers = obs_count[obs_count >= min_obs].index
    df = df[df["ticker"].isin(valid_tickers)].copy()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def save_clean_data(
    input_path: str | Path = RAW_PANEL_PATH,
    output_path: str | Path = CLEAN_DATA_PATH,
    min_obs_ratio: float = 0.8,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load raw panel data, clean it, save it, and return the cleaned panel."""
    df = load_panel(input_path)
    clean = clean_daily_panel(
        df,
        min_obs_ratio=min_obs_ratio,
        start_date=start_date,
        end_date=end_date,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    return clean


def data_quality_report(df: pd.DataFrame) -> dict[str, object]:
    """Return compact quality diagnostics for a daily panel."""
    df = standardize_schema(df)
    summary = df.groupby("ticker")["date"].agg(["min", "max", "count"])
    return {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "missing_ratio": df.isna().mean().sort_values(ascending=False),
        "ticker_observation_summary": summary["count"].describe(),
    }


if __name__ == "__main__":
    clean_df = save_clean_data()
    report = data_quality_report(clean_df)
    print(f"Saved clean data to {CLEAN_DATA_PATH}")
    print(f"Rows: {report['rows']:,}, tickers: {report['tickers']}")
    print(f"Date range: {report['start_date']} -> {report['end_date']}")
