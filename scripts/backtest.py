from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return float(drawdown.min())


def run_long_short_backtest(
    df: pd.DataFrame,
    score_col: str = "score",
    return_col: str = "ret_1d",
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    fee_bps: float = 10.0,
    min_obs: int = 30,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Daily close-to-close long-short backtest.

    Signals are formed at date t and applied to date t+1 returns to avoid look-ahead.
    """
    panel = df.sort_values(["date", "ticker"]).copy()
    panel["next_ret"] = panel.groupby("ticker")[return_col].shift(-1)

    daily_rows = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    fee_rate = fee_bps / 10000

    for date, group in panel.groupby("date"):
        temp = group[["ticker", score_col, "next_ret"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(temp) < min_obs:
            continue

        low_cut = temp[score_col].quantile(bottom_quantile)
        high_cut = temp[score_col].quantile(1 - top_quantile)
        long_names = set(temp.loc[temp[score_col] >= high_cut, "ticker"])
        short_names = set(temp.loc[temp[score_col] <= low_cut, "ticker"])

        long_ret = temp.loc[temp["ticker"].isin(long_names), "next_ret"].mean()
        short_ret = temp.loc[temp["ticker"].isin(short_names), "next_ret"].mean()
        gross_ret = long_ret - short_ret

        if prev_long or prev_short:
            long_turnover = 1 - len(long_names & prev_long) / max(len(long_names), 1)
            short_turnover = 1 - len(short_names & prev_short) / max(len(short_names), 1)
            turnover = 0.5 * (long_turnover + short_turnover)
        else:
            turnover = 1.0
        cost = turnover * fee_rate

        daily_rows.append(
            {
                "date": date,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "gross_ret": gross_ret,
                "turnover": turnover,
                "cost": cost,
                "ret": gross_ret - cost,
                "n_long": len(long_names),
                "n_short": len(short_names),
            }
        )
        prev_long = long_names
        prev_short = short_names

    result = pd.DataFrame(daily_rows)
    if result.empty:
        return result, {}

    result["nav"] = (1 + result["ret"]).cumprod()
    result["long_nav"] = (1 + result["long_ret"]).cumprod()
    result["short_nav"] = (1 + result["short_ret"]).cumprod()

    n_days = len(result)
    ann_return = result["nav"].iloc[-1] ** (252 / n_days) - 1
    ann_vol = result["ret"].std(ddof=1) * np.sqrt(252)
    metrics = {
        "annual_return": float(ann_return),
        "annual_vol": float(ann_vol),
        "sharpe": float(ann_return / ann_vol) if ann_vol else np.nan,
        "max_drawdown": _max_drawdown(result["nav"]),
        "win_rate": float((result["ret"] > 0).mean()),
        "avg_turnover": float(result["turnover"].mean()),
        "total_return": float(result["nav"].iloc[-1] - 1),
        "n_days": float(n_days),
    }
    return result, metrics
