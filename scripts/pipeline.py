from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts.backtest import run_long_short_backtest
    from scripts.data_processing import CLEAN_DATA_PATH, RAW_PANEL_PATH, clean_daily_panel, load_panel
    from scripts.factor_evaluation import calc_daily_ic, combine_factors_by_ic, summarize_ic
    from scripts.factors import build_factor_panel
except ModuleNotFoundError:
    from backtest import run_long_short_backtest
    from data_processing import CLEAN_DATA_PATH, RAW_PANEL_PATH, clean_daily_panel, load_panel
    from factor_evaluation import calc_daily_ic, combine_factors_by_ic, summarize_ic
    from factors import build_factor_panel


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def run_full_pipeline(
    raw_path: str | Path = RAW_PANEL_PATH,
    clean_path: str | Path = CLEAN_DATA_PATH,
    min_obs_ratio: float = 0.8,
    forward_return_col: str = "fwd_ret_20d",
    max_tickers: int | None = None,
) -> dict[str, object]:
    """Run cleaning, factor construction, IC evaluation, and backtest."""
    raw = load_panel(raw_path)
    clean = clean_daily_panel(raw, min_obs_ratio=min_obs_ratio)
    if max_tickers is not None:
        keep_tickers = clean["ticker"].drop_duplicates().head(max_tickers).tolist()
        clean = clean[clean["ticker"].isin(keep_tickers)].copy()

    clean_path = Path(clean_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

    factor_panel, factor_cols = build_factor_panel(clean)
    factor_path = OUTPUT_DIR / "factor_panel.csv"
    factor_panel.to_csv(factor_path, index=False, encoding="utf-8-sig")

    ic = calc_daily_ic(factor_panel, factor_cols, forward_return_col=forward_return_col)
    ic_summary = summarize_ic(ic)
    scored = combine_factors_by_ic(factor_panel, factor_cols, ic_summary)
    backtest, metrics = run_long_short_backtest(scored)

    ic_path = OUTPUT_DIR / "factor_ic_daily.csv"
    ic_summary_path = OUTPUT_DIR / "factor_ic_summary.csv"
    score_path = OUTPUT_DIR / "factor_panel_scored.csv"
    backtest_path = OUTPUT_DIR / "backtest_nav.csv"

    ic.to_csv(ic_path, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(ic_summary_path, index=False, encoding="utf-8-sig")
    scored.to_csv(score_path, index=False, encoding="utf-8-sig")
    backtest.to_csv(backtest_path, index=False, encoding="utf-8-sig")

    return {
        "clean": clean,
        "factor_panel": factor_panel,
        "factor_cols": factor_cols,
        "ic": ic,
        "ic_summary": ic_summary,
        "scored": scored,
        "backtest": backtest,
        "metrics": metrics,
        "paths": {
            "clean": clean_path,
            "factor_panel": factor_path,
            "ic": ic_path,
            "ic_summary": ic_summary_path,
            "scored": score_path,
            "backtest": backtest_path,
        },
    }


def main() -> None:
    result = run_full_pipeline()
    print("IC summary:")
    print(result["ic_summary"].head(10))
    print("\nBacktest metrics:")
    print(pd.Series(result["metrics"]))
    print("\nSaved outputs:")
    for name, path in result["paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
