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

OPTIONAL_COLS = [
    "tradestatus",
    "is_st",
    "isst",
    "industry",
    "market_cap",
    "float_market_cap",
    "index_start_date",
    "index_end_date",
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

    keep_cols = STANDARD_COLS + [col for col in OPTIONAL_COLS if col in df.columns]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.extract(r"(\d{6})", expand=False)
    df["ticker"] = df["ticker"].str.zfill(6)

    numeric_cols = [col for col in keep_cols if col not in {"date", "ticker", "industry"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "isst" in df.columns and "is_st" not in df.columns:
        df = df.rename(columns={"isst": "is_st"})

    for col in ["index_start_date", "index_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

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


def add_trading_constraints(
    df: pd.DataFrame,
    st_limit: float = 0.05,
    main_board_limit: float = 0.10,
    growth_board_limit: float = 0.20,
    limit_tolerance: float = 0.002,
) -> pd.DataFrame:
    """Infer tradability, ST, suspension, and limit-up/down flags.

    The function uses explicit ``tradestatus``/``is_st`` columns when present.
    If older data does not contain them, it falls back to volume and return
    based inference so the rest of the research workflow still runs.
    """
    out = standardize_schema(df).sort_values(["ticker", "date"]).copy()

    out["prev_close"] = out.groupby("ticker")["close"].shift(1)
    if "ret_1d" not in out.columns:
        out["ret_1d"] = out.groupby("ticker")["close"].pct_change()

    if "is_st" in out.columns:
        out["is_st_flag"] = out["is_st"].fillna(0).astype(float).astype(bool)
    else:
        out["is_st_flag"] = False

    if "tradestatus" in out.columns:
        out["is_suspended"] = out["tradestatus"].fillna(1).astype(float) != 1
    else:
        out["is_suspended"] = (out["volume"].fillna(0) <= 0) | (out["amount"].fillna(0) <= 0)

    tickers = out["ticker"].astype(str)
    is_growth_board = tickers.str.startswith(("300", "301", "688"))
    out["limit_threshold"] = np.select(
        [out["is_st_flag"], is_growth_board],
        [st_limit, growth_board_limit],
        default=main_board_limit,
    )

    pct_ret = out["pct_change"] / 100.0
    out["is_limit_up"] = pct_ret >= (out["limit_threshold"] - limit_tolerance)
    out["is_limit_down"] = pct_ret <= (-out["limit_threshold"] + limit_tolerance)
    out["is_tradable"] = ~(out["is_suspended"] | out["is_st_flag"])
    out["can_buy"] = out["is_tradable"] & ~out["is_limit_up"]
    out["can_sell"] = out["is_tradable"] & ~out["is_limit_down"]
    return out.reset_index(drop=True)


def apply_point_in_time_universe(
    df: pd.DataFrame,
    membership: pd.DataFrame | None = None,
    start_col: str = "index_start_date",
    end_col: str = "index_end_date",
) -> pd.DataFrame:
    """Filter rows to stocks that were in the universe on each date.

    ``membership`` can contain ticker plus start/end dates. If omitted, the
    function looks for the same columns in ``df``. Missing end dates mean the
    constituent is still active.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    if membership is None:
        if start_col not in out.columns:
            return out
        membership_cols = ["ticker", start_col]
        if end_col in out.columns:
            membership_cols.append(end_col)
        membership = out[membership_cols].drop_duplicates()
    else:
        membership = membership.copy()

    membership["ticker"] = membership["ticker"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    membership[start_col] = pd.to_datetime(membership[start_col], errors="coerce")
    if end_col in membership.columns:
        membership[end_col] = pd.to_datetime(membership[end_col], errors="coerce")
    else:
        membership[end_col] = pd.NaT

    merged = out.merge(membership[["ticker", start_col, end_col]], on="ticker", how="left", suffixes=("", "_member"))
    in_universe = merged[start_col].isna() | (
        (merged["date"] >= merged[start_col])
        & (merged[end_col].isna() | (merged["date"] <= merged[end_col]))
    )
    return merged.loc[in_universe, out.columns].reset_index(drop=True)


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
