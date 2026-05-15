from __future__ import annotations

import pandas as pd


def panel_overview(df: pd.DataFrame) -> dict[str, object]:
    """Create a compact overview of a stock panel."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return {
        "rows": len(df),
        "tickers": df["ticker"].nunique(),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "trading_days": df["date"].nunique(),
        "avg_names_per_day": df.groupby("date")["ticker"].nunique().mean(),
    }


def daily_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily cross-sectional return and liquidity statistics."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    return (
        df.groupby("date")
        .agg(
            n_stocks=("ticker", "nunique"),
            equal_weight_ret=("ret_1d", "mean"),
            median_ret=("ret_1d", "median"),
            total_amount=("amount", "sum"),
            avg_turnover=("turnover", "mean"),
        )
        .reset_index()
    )


def ticker_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize observations, date range, and simple return stats by ticker."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "ret_1d" not in df.columns:
        df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    return (
        df.groupby("ticker")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            observations=("date", "count"),
            ann_return=("ret_1d", lambda s: (1 + s.dropna()).prod() ** (252 / max(len(s.dropna()), 1)) - 1),
            ann_vol=("ret_1d", lambda s: s.std() * (252**0.5)),
            avg_amount=("amount", "mean"),
        )
        .reset_index()
    )
