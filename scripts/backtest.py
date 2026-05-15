from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionCostConfig:
    """One-way trading cost assumptions in basis points."""

    commission_bps: float = 3.0
    stamp_tax_bps: float = 5.0
    slippage_bps: float = 5.0
    market_impact_bps_per_turnover: float = 10.0


def _estimate_trading_cost(
    one_way_turnover: float,
    config: TransactionCostConfig,
    legacy_fee_bps: float | None = None,
) -> float:
    if legacy_fee_bps is not None:
        return one_way_turnover * legacy_fee_bps / 10000

    commission = config.commission_bps / 10000
    stamp_tax = config.stamp_tax_bps / 10000
    slippage = config.slippage_bps / 10000
    impact = config.market_impact_bps_per_turnover / 10000 * one_way_turnover
    return one_way_turnover * (2 * commission + stamp_tax + 2 * slippage + impact)


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
    fee_bps: float | None = None,
    cost_config: TransactionCostConfig | None = None,
    min_obs: int = 30,
    apply_trade_constraints: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Daily close-to-close long-short backtest.

    Signals are formed at date t and applied to date t+1 returns to avoid look-ahead.
    """
    panel = df.sort_values(["date", "ticker"]).copy()
    panel["next_ret"] = panel.groupby("ticker")[return_col].shift(-1)
    cost_config = cost_config or TransactionCostConfig()

    if apply_trade_constraints:
        defaults = {
            "can_buy": True,
            "can_sell": True,
            "is_tradable": True,
        }
        for col, default in defaults.items():
            if col not in panel.columns:
                panel[col] = default
        panel["next_is_tradable"] = panel.groupby("ticker")["is_tradable"].shift(-1)
        panel["next_is_tradable"] = panel["next_is_tradable"].where(panel["next_is_tradable"].notna(), False).astype(bool)
    else:
        panel["can_buy"] = True
        panel["can_sell"] = True
        panel["next_is_tradable"] = True

    daily_rows = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for date, group in panel.groupby("date"):
        temp = group[
            ["ticker", score_col, "next_ret", "can_buy", "can_sell", "next_is_tradable"]
        ].replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col, "next_ret"])
        temp = temp[temp["next_is_tradable"]]
        if len(temp) < min_obs:
            continue

        low_cut = temp[score_col].quantile(bottom_quantile)
        high_cut = temp[score_col].quantile(1 - top_quantile)
        long_names = set(temp.loc[(temp[score_col] >= high_cut) & temp["can_buy"], "ticker"])
        short_names = set(temp.loc[(temp[score_col] <= low_cut) & temp["can_sell"], "ticker"])
        if not long_names or not short_names:
            continue

        long_ret = temp.loc[temp["ticker"].isin(long_names), "next_ret"].mean()
        short_ret = temp.loc[temp["ticker"].isin(short_names), "next_ret"].mean()
        gross_ret = long_ret - short_ret

        if prev_long or prev_short:
            long_turnover = 1 - len(long_names & prev_long) / max(len(long_names), 1)
            short_turnover = 1 - len(short_names & prev_short) / max(len(short_names), 1)
            turnover = 0.5 * (long_turnover + short_turnover)
        else:
            turnover = 1.0
        cost = _estimate_trading_cost(turnover, cost_config, legacy_fee_bps=fee_bps)

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


def run_ml_long_short_backtest(
    df: pd.DataFrame,
    factor_cols: list[str],
    target_col: str = "fwd_ret_5d",
    model=None,
    train_window: int = 504,
    retrain_freq: int = 21,
    score_col: str = "ml_score",
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    fee_bps: float | None = None,
    cost_config: TransactionCostConfig | None = None,
    min_obs: int = 30,
    apply_trade_constraints: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    from sklearn.base import clone as clone_model

    panel = df.sort_values(["ticker", "date"]).copy()
    dates = sorted(panel["date"].unique())

    if len(dates) <= train_window:
        raise ValueError(f"Need >{train_window} dates, got {len(dates)}")

    if model is None:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    all_scores: list[pd.DataFrame] = []
    current_model = None

    for i, pred_date in enumerate(dates[train_window:]):
        if i % retrain_freq == 0:
            train = panel[panel["date"] < pred_date]
            X_train = train[factor_cols].values.astype(float)
            y_train = train[target_col].values.astype(float)
            mask = ~(
                np.isnan(X_train).any(axis=1)
                | np.isnan(y_train)
                | np.isinf(X_train).any(axis=1)
                | np.isinf(y_train)
            )
            if mask.sum() < 10:
                continue
            current_model = clone_model(model)
            current_model.fit(X_train[mask], y_train[mask])

        if current_model is None:
            continue

        pred = panel[panel["date"] == pred_date]
        X_pred = pred[factor_cols].values.astype(float)
        mask = ~(np.isnan(X_pred).any(axis=1) | np.isinf(X_pred).any(axis=1))
        scores = np.full(len(pred), np.nan)
        if mask.any():
            scores[mask] = current_model.predict(X_pred[mask])
        all_scores.append(
            pd.DataFrame({
                "date": pred_date,
                "ticker": pred["ticker"].values,
                score_col: scores,
            })
        )

    if not all_scores:
        raise ValueError("No predictions generated — check data and training window.")

    score_df = pd.concat(all_scores, ignore_index=True)
    panel = panel.merge(score_df, on=["date", "ticker"], how="left")

    return run_long_short_backtest(
        panel,
        score_col=score_col,
        return_col="ret_1d",
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        fee_bps=fee_bps,
        cost_config=cost_config,
        min_obs=min_obs,
        apply_trade_constraints=apply_trade_constraints,
    )
