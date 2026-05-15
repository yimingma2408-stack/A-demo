import unittest

import numpy as np
import pandas as pd

from scripts.backtest import TransactionCostConfig, run_long_short_backtest
from scripts.data_processing import add_trading_constraints, apply_point_in_time_universe
from scripts.factors import neutralize_factors_by_date
from scripts.validation import walk_forward_ic_scores


class ResearchWorkflowTest(unittest.TestCase):
    def test_trading_constraints_infer_suspension_and_limits(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "ticker": ["000001", "000001", "000001"],
                "open": [10, 11, 11],
                "high": [10, 11, 11],
                "low": [10, 11, 11],
                "close": [10, 11, 11],
                "volume": [100, 100, 0],
                "amount": [1000, 1100, 0],
                "pct_change": [0, 10, 0],
                "turnover": [1, 1, 0],
            }
        )

        out = add_trading_constraints(df)

        self.assertFalse(bool(out.loc[1, "can_buy"]))
        self.assertTrue(bool(out.loc[1, "is_limit_up"]))
        self.assertTrue(bool(out.loc[2, "is_suspended"]))
        self.assertFalse(bool(out.loc[2, "is_tradable"]))

    def test_point_in_time_universe_filters_future_members(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "ticker": ["000001", "000001"],
                "open": [1, 1],
                "high": [1, 1],
                "low": [1, 1],
                "close": [1, 1],
                "volume": [1, 1],
                "amount": [1, 1],
                "pct_change": [0, 0],
                "turnover": [1, 1],
            }
        )
        membership = pd.DataFrame(
            {"ticker": ["000001"], "index_start_date": ["2024-01-03"], "index_end_date": [None]}
        )

        out = apply_point_in_time_universe(df, membership)

        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.loc[0, "date"].date()), "2024-01-03")

    def test_neutralization_removes_size_exposure(self):
        rows = []
        for size in range(1, 31):
            rows.append(
                {
                    "date": "2024-01-02",
                    "ticker": f"{size:06d}",
                    "factor": np.log(size),
                    "industry": "bank" if size % 2 else "tech",
                    "float_market_cap": size,
                }
            )
        df = pd.DataFrame(rows)

        out = neutralize_factors_by_date(df, ["factor"], min_obs=10)
        self.assertLess(out["factor"].abs().max(), 1e-10)

    def test_backtest_applies_transaction_costs_and_constraints(self):
        dates = pd.date_range("2024-01-01", periods=4)
        rows = []
        for date in dates:
            for i in range(40):
                rows.append(
                    {
                        "date": date,
                        "ticker": f"{i:06d}",
                        "score": i,
                        "ret_1d": 0.01 if i >= 32 else -0.01,
                        "can_buy": i != 39,
                        "can_sell": True,
                        "is_tradable": True,
                    }
                )
        df = pd.DataFrame(rows)

        bt, metrics = run_long_short_backtest(
            df,
            cost_config=TransactionCostConfig(
                commission_bps=1,
                stamp_tax_bps=1,
                slippage_bps=1,
                market_impact_bps_per_turnover=0,
            ),
        )

        self.assertFalse(bt.empty)
        self.assertLess(bt["ret"].iloc[0], bt["gross_ret"].iloc[0])
        self.assertGreater(metrics["n_days"], 0)

    def test_walk_forward_scores_are_generated_after_training_window(self):
        dates = pd.date_range("2023-01-01", periods=40)
        rows = []
        for date in dates:
            for i in range(25):
                rows.append(
                    {
                        "date": date,
                        "ticker": f"{i:06d}",
                        "factor_z": i,
                        "fwd_ret_20d": i / 1000,
                        "ret_1d": i / 10000,
                        "is_tradable": True,
                        "can_buy": True,
                        "can_sell": True,
                    }
                )
        df = pd.DataFrame(rows)

        scored, weights = walk_forward_ic_scores(
            df,
            ["factor_z"],
            train_window=20,
            test_window=5,
            min_train_days=20,
        )

        self.assertIn("score", scored.columns)
        self.assertFalse(weights.empty)
        self.assertGreaterEqual(scored["date"].min(), dates[20])


if __name__ == "__main__":
    unittest.main()
