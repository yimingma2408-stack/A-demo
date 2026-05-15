from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scripts.backtest import TransactionCostConfig, run_long_short_backtest
    from scripts.factor_evaluation import calc_daily_ic
except ModuleNotFoundError:
    from backtest import TransactionCostConfig, run_long_short_backtest
    from factor_evaluation import calc_daily_ic


def temporal_train_test_split(
    df: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split panel data by date for explicit sample-in/sample-out checks."""
    split = pd.Timestamp(split_date)
    panel = df.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    return panel[panel["date"] < split].copy(), panel[panel["date"] >= split].copy()


def walk_forward_ic_scores(
    df: pd.DataFrame,
    factor_cols: list[str],
    forward_return_col: str = "fwd_ret_20d",
    train_window: int = 504,
    test_window: int = 21,
    min_train_days: int = 252,
    score_col: str = "score",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create out-of-sample factor scores using rolling IC weights."""
    panel = df.sort_values(["date", "ticker"]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    dates = pd.Index(sorted(panel["date"].dropna().unique()))
    if len(dates) <= min_train_days:
        raise ValueError(f"Need more than {min_train_days} dates for walk-forward validation.")

    daily_ic = calc_daily_ic(panel, factor_cols, forward_return_col=forward_return_col)
    ic_wide = daily_ic.pivot(index="date", columns="factor", values="ic").reindex(dates)
    rolling_ic = ic_wide.rolling(train_window, min_periods=min_train_days).mean().shift(1)

    scored_parts: list[pd.DataFrame] = []
    weight_rows: list[dict[str, object]] = []
    start = min_train_days

    while start < len(dates):
        pred_dates = dates[start : min(start + test_window, len(dates))]
        weights = rolling_ic.loc[pd.Timestamp(pred_dates[0])].reindex(factor_cols).fillna(0.0)
        if weights.abs().sum() == 0:
            weights = pd.Series(1.0, index=factor_cols)
        weights = weights / weights.abs().sum()

        for factor, weight in weights.items():
            weight_rows.append(
                {
                    "rebalance_date": pd.Timestamp(pred_dates[0]),
                    "factor": factor,
                    "weight": float(weight),
                }
            )

        test = panel[panel["date"].isin(pred_dates)].copy()
        test[score_col] = test[factor_cols].replace([np.inf, -np.inf], np.nan).mul(weights, axis=1).sum(axis=1, min_count=1)
        scored_parts.append(test)
        start += test_window

    if not scored_parts:
        raise ValueError("No walk-forward scores generated.")

    return pd.concat(scored_parts, ignore_index=True), pd.DataFrame(weight_rows)


def run_walk_forward_ic_backtest(
    df: pd.DataFrame,
    factor_cols: list[str],
    forward_return_col: str = "fwd_ret_20d",
    train_window: int = 504,
    test_window: int = 21,
    min_train_days: int = 252,
    cost_config: TransactionCostConfig | None = None,
) -> dict[str, object]:
    """Run rolling out-of-sample IC weighting and long-short backtest."""
    scored, weights = walk_forward_ic_scores(
        df,
        factor_cols,
        forward_return_col=forward_return_col,
        train_window=train_window,
        test_window=test_window,
        min_train_days=min_train_days,
    )
    backtest, metrics = run_long_short_backtest(scored, cost_config=cost_config)
    return {
        "scored": scored,
        "weights": weights,
        "backtest": backtest,
        "metrics": metrics,
    }
