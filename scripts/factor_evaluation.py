from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def calc_daily_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    forward_return_col: str = "fwd_ret_20d",
    method: str = "spearman",
    min_obs: int = 20,
) -> pd.DataFrame:
    """Calculate daily cross-sectional IC for each factor."""
    records: list[dict[str, object]] = []
    for date, group in df.groupby("date"):
        for factor in factor_cols:
            temp = group[[factor, forward_return_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(temp) < min_obs:
                continue
            if temp[factor].nunique() < 2 or temp[forward_return_col].nunique() < 2:
                continue
            ic = temp[factor].corr(temp[forward_return_col], method=method)
            records.append({"date": date, "factor": factor, "ic": ic})
    return pd.DataFrame(records)


def summarize_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IC mean, volatility, t-stat, positive ratio, and IR."""
    rows = []
    for factor, group in ic_df.dropna(subset=["ic"]).groupby("factor"):
        ic = group["ic"]
        n = len(ic)
        mean = ic.mean()
        std = ic.std(ddof=1)
        t_stat = mean / std * np.sqrt(n) if std and n > 1 else np.nan
        p_value = stats.ttest_1samp(ic, 0, nan_policy="omit").pvalue if n > 1 else np.nan
        rows.append(
            {
                "factor": factor,
                "ic_mean": mean,
                "ic_std": std,
                "ic_ir": mean / std if std else np.nan,
                "t_stat": t_stat,
                "p_value": p_value,
                "positive_ratio": (ic > 0).mean(),
                "n_days": n,
            }
        )
    return pd.DataFrame(rows).sort_values("ic_mean", key=lambda s: s.abs(), ascending=False)


def quantile_return_analysis(
    df: pd.DataFrame,
    factor_col: str,
    forward_return_col: str = "fwd_ret_20d",
    n_quantiles: int = 5,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Analyze next-period returns by daily factor quantile."""
    pieces = []
    for date, group in df.groupby("date"):
        temp = group[["ticker", factor_col, forward_return_col]].dropna()
        if len(temp) < max(min_obs, n_quantiles):
            continue
        try:
            temp["quantile"] = pd.qcut(
                temp[factor_col],
                q=n_quantiles,
                labels=False,
                duplicates="drop",
            ) + 1
        except ValueError:
            continue
        pieces.append(
            temp.groupby("quantile", observed=True)[forward_return_col]
            .mean()
            .rename(date)
        )

    if not pieces:
        return pd.DataFrame(columns=["quantile", "mean_return", "std_return", "count"])

    quantile_returns = pd.concat(pieces, axis=1).T
    return (
        quantile_returns.agg(["mean", "std", "count"])
        .T.rename(columns={"mean": "mean_return", "std": "std_return", "count": "count"})
        .reset_index()
    )


def combine_factors_by_ic(
    df: pd.DataFrame,
    factor_cols: list[str],
    ic_summary: pd.DataFrame,
    output_col: str = "score",
) -> pd.DataFrame:
    """Combine factors using the sign and magnitude of IC means."""
    weights = ic_summary.set_index("factor").reindex(factor_cols)["ic_mean"].fillna(0.0)
    if weights.abs().sum() == 0:
        weights = pd.Series(1.0, index=factor_cols)
    weights = weights / weights.abs().sum()

    out = df.copy()
    out[output_col] = out[factor_cols].mul(weights, axis=1).sum(axis=1, min_count=1)
    return out
