from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts.backtest import TransactionCostConfig, run_long_short_backtest, run_ml_long_short_backtest
    from scripts.data_processing import (
        CLEAN_DATA_PATH,
        RAW_PANEL_PATH,
        add_trading_constraints,
        apply_point_in_time_universe,
        clean_daily_panel,
        load_panel,
    )
    from scripts.factor_evaluation import calc_daily_ic, combine_factors_by_ic, summarize_ic
    from scripts.factors import build_factor_panel
    from scripts.validation import run_walk_forward_ic_backtest
except ModuleNotFoundError:
    from backtest import TransactionCostConfig, run_long_short_backtest, run_ml_long_short_backtest
    from data_processing import (
        CLEAN_DATA_PATH,
        RAW_PANEL_PATH,
        add_trading_constraints,
        apply_point_in_time_universe,
        clean_daily_panel,
        load_panel,
    )
    from factor_evaluation import calc_daily_ic, combine_factors_by_ic, summarize_ic
    from factors import build_factor_panel
    from validation import run_walk_forward_ic_backtest


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def run_full_pipeline(
    raw_path: str | Path = RAW_PANEL_PATH,
    clean_path: str | Path = CLEAN_DATA_PATH,
    min_obs_ratio: float = 0.8,
    forward_return_col: str = "fwd_ret_20d",
    max_tickers: int | None = None,
    membership_path: str | Path | None = None,
    apply_constraints: bool = True,
    neutralize: bool = False,
    run_oos: bool = True,
    cost_config: TransactionCostConfig | None = None,
) -> dict[str, object]:
    """Run cleaning, factor construction, IC evaluation, and backtest."""
    raw = load_panel(raw_path)
    clean = clean_daily_panel(raw, min_obs_ratio=min_obs_ratio)
    if membership_path is not None:
        membership = pd.read_csv(membership_path, encoding="utf-8-sig")
        clean = apply_point_in_time_universe(clean, membership)
    else:
        clean = apply_point_in_time_universe(clean)
    if apply_constraints:
        clean = add_trading_constraints(clean)
    if max_tickers is not None:
        keep_tickers = clean["ticker"].drop_duplicates().head(max_tickers).tolist()
        clean = clean[clean["ticker"].isin(keep_tickers)].copy()

    clean_path = Path(clean_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

    factor_panel, factor_cols = build_factor_panel(clean, neutralize=neutralize)
    factor_path = OUTPUT_DIR / "factor_panel.csv"
    factor_panel.to_csv(factor_path, index=False, encoding="utf-8-sig")

    ic = calc_daily_ic(factor_panel, factor_cols, forward_return_col=forward_return_col)
    ic_summary = summarize_ic(ic)
    scored = combine_factors_by_ic(factor_panel, factor_cols, ic_summary)
    backtest, metrics = run_long_short_backtest(scored, cost_config=cost_config)

    ic_path = OUTPUT_DIR / "factor_ic_daily.csv"
    ic_summary_path = OUTPUT_DIR / "factor_ic_summary.csv"
    score_path = OUTPUT_DIR / "factor_panel_scored.csv"
    backtest_path = OUTPUT_DIR / "backtest_nav.csv"

    ic.to_csv(ic_path, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(ic_summary_path, index=False, encoding="utf-8-sig")
    scored.to_csv(score_path, index=False, encoding="utf-8-sig")
    backtest.to_csv(backtest_path, index=False, encoding="utf-8-sig")

    oos_result = None
    if run_oos:
        oos_result = run_walk_forward_ic_backtest(
            factor_panel,
            factor_cols,
            forward_return_col=forward_return_col,
            cost_config=cost_config,
        )
        oos_score_path = OUTPUT_DIR / "factor_panel_oos_scored.csv"
        oos_weights_path = OUTPUT_DIR / "factor_oos_weights.csv"
        oos_backtest_path = OUTPUT_DIR / "backtest_oos_nav.csv"
        oos_result["scored"].to_csv(oos_score_path, index=False, encoding="utf-8-sig")
        oos_result["weights"].to_csv(oos_weights_path, index=False, encoding="utf-8-sig")
        oos_result["backtest"].to_csv(oos_backtest_path, index=False, encoding="utf-8-sig")

    return {
        "clean": clean,
        "factor_panel": factor_panel,
        "factor_cols": factor_cols,
        "ic": ic,
        "ic_summary": ic_summary,
        "scored": scored,
        "backtest": backtest,
        "metrics": metrics,
        "oos": oos_result,
        "paths": {
            "clean": clean_path,
            "factor_panel": factor_path,
            "ic": ic_path,
            "ic_summary": ic_summary_path,
            "scored": score_path,
            "backtest": backtest_path,
            **(
                {
                    "oos_scored": oos_score_path,
                    "oos_weights": oos_weights_path,
                    "oos_backtest": oos_backtest_path,
                }
                if oos_result is not None
                else {}
            ),
        },
    }


def run_ml_pipeline(
    raw_path: str | Path = RAW_PANEL_PATH,
    clean_path: str | Path = CLEAN_DATA_PATH,
    min_obs_ratio: float = 0.8,
    forward_return_col: str = "fwd_ret_20d",
    max_tickers: int | None = None,
    model=None,
    target_col: str = "fwd_ret_5d",
    train_window: int = 504,
    retrain_freq: int = 21,
    apply_constraints: bool = True,
    neutralize: bool = False,
    cost_config: TransactionCostConfig | None = None,
) -> dict[str, object]:
    from sklearn.svm import SVR

    raw = load_panel(raw_path)
    clean = clean_daily_panel(raw, min_obs_ratio=min_obs_ratio)
    if apply_constraints:
        clean = add_trading_constraints(clean)
    if max_tickers is not None:
        keep_tickers = clean["ticker"].drop_duplicates().head(max_tickers).tolist()
        clean = clean[clean["ticker"].isin(keep_tickers)].copy()

    clean_path = Path(clean_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

    factor_panel, factor_cols = build_factor_panel(clean, neutralize=neutralize)
    factor_path = OUTPUT_DIR / "factor_panel.csv"
    factor_panel.to_csv(factor_path, index=False, encoding="utf-8-sig")

    model = model or SVR(kernel="rbf")
    backtest, metrics = run_ml_long_short_backtest(
        factor_panel, factor_cols, model=model,
        target_col=target_col, train_window=train_window,
        retrain_freq=retrain_freq, cost_config=cost_config,
    )
    backtest_path = OUTPUT_DIR / "backtest_nav.csv"
    backtest.to_csv(backtest_path, index=False, encoding="utf-8-sig")

    return {
        "clean": clean,
        "factor_panel": factor_panel,
        "factor_cols": factor_cols,
        "backtest": backtest,
        "metrics": metrics,
        "paths": {"backtest": backtest_path},
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", action="store_true", help="Use ML model instead of IC weighting")
    parser.add_argument("--kernel", default="rbf", help="Kernel for SVR: linear, poly, rbf, sigmoid")
    parser.add_argument("--no-oos", action="store_true", help="Skip walk-forward out-of-sample validation")
    parser.add_argument("--neutralize", action="store_true", help="Neutralize factors by industry/size when metadata exists")
    parser.add_argument("--membership-path", default=None, help="Optional point-in-time universe membership CSV")
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-tax-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--impact-bps", type=float, default=10.0)
    args = parser.parse_args()
    cost_config = TransactionCostConfig(
        commission_bps=args.commission_bps,
        stamp_tax_bps=args.stamp_tax_bps,
        slippage_bps=args.slippage_bps,
        market_impact_bps_per_turnover=args.impact_bps,
    )

    if args.ml:
        from sklearn.svm import SVR
        model = SVR(kernel=args.kernel)
        result = run_ml_pipeline(model=model, neutralize=args.neutralize, cost_config=cost_config)
        print("\nML Backtest metrics:")
    else:
        result = run_full_pipeline(
            membership_path=args.membership_path,
            neutralize=args.neutralize,
            run_oos=not args.no_oos,
            cost_config=cost_config,
        )
        print("IC summary:")
        print(result["ic_summary"].head(10))
        print("\nBacktest metrics:")

    print(pd.Series(result["metrics"]))
    if result.get("oos") is not None:
        print("\nWalk-forward OOS metrics:")
        print(pd.Series(result["oos"]["metrics"]))
    print("\nSaved outputs:")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
