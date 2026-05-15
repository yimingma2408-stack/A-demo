import pandas as pd
import numpy as np


def add_daily_return(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    为每只股票计算日收益率（日频收益率因子）

    日收益率计算公式：ret_1d = (price_t - price_{t-1}) / price_{t-1}

    Args:
        df: 股票面板数据，必须包含列：
            - ticker: 股票代码
            - date: 交易日期
            - price_col: 价格列（默认"close"，即收盘价）
        price_col: 用于计算收益率的价格列名称，默认为"close"

    Returns:
        新增"ret_1d"列的DataFrame，包含每日收益率

    Notes:
        - 函数会先按ticker和date排序，确保时序正确
        - 每只股票的第一个交易日ret_1d为NaN（无前置数据）
        - 返回的是数据副本，不修改原DataFrame
    """
    # 按股票代码和日期排序，确保时间顺序正确（收益率计算的前提）
    df = df.sort_values(["ticker", "date"]).copy()

    # 按股票分组，计算每组价格的日变化百分比（即日收益率）
    df["ret_1d"] = df.groupby("ticker")[price_col].pct_change()

    return df



def add_forward_return(
    df: pd.DataFrame,
    price_col: str = "close",
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Add forward return over a given horizon.

    This is the future return to be predicted by factors.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    future_price = df.groupby("ticker")[price_col].shift(-horizon)
    df[f"fwd_ret_{horizon}d"] = future_price / df[price_col] - 1

    return df


def add_reversal_factor(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
) -> pd.DataFrame:
    """
    Short-term reversal factor.

    Higher value means stronger past underperformance.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    past_ret = df.groupby("ticker")[price_col].pct_change(window)
    df[f"rev_{window}d"] = -past_ret

    return df


def add_momentum_factor(
    df: pd.DataFrame,
    price_col: str = "close",
    lookback: int = 120,
    skip: int = 20,
) -> pd.DataFrame:
    """
    Medium-term momentum factor.

    Use return from t-lookback to t-skip.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    price_lag_skip = df.groupby("ticker")[price_col].shift(skip)
    price_lag_lookback = df.groupby("ticker")[price_col].shift(lookback)

    df[f"mom_{lookback}_{skip}d"] = price_lag_skip / price_lag_lookback - 1

    return df


def add_volatility_factor(
    df: pd.DataFrame,
    ret_col: str = "ret_1d",
    window: int = 60,
) -> pd.DataFrame:
    """
    Low-volatility factor.

    Higher value means lower past volatility.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    vol = (
        df.groupby("ticker")[ret_col]
        .rolling(window)
        .std()
        .reset_index(level=0, drop=True)
    )

    df[f"vol_{window}d"] = vol
    df[f"lowvol_{window}d"] = -vol

    return df


def add_liquidity_factor(
    df: pd.DataFrame,
    amount_col: str = "amount",
    window: int = 20,
) -> pd.DataFrame:
    """
    Liquidity factor based on rolling average amount.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    avg_amount = (
        df.groupby("ticker")[amount_col]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"liq_{window}d"] = avg_amount

    return df


def add_turnover_factor(
    df: pd.DataFrame,
    turnover_col: str = "turnover",
    window: int = 20,
) -> pd.DataFrame:
    """
    Rolling average turnover factor.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    avg_turnover = (
        df.groupby("ticker")[turnover_col]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"turnover_{window}d"] = avg_turnover

    return df


def add_trend_factor(
    df: pd.DataFrame,
    price_col: str = "close",
    short_window: int = 20,
    long_window: int = 120,
) -> pd.DataFrame:
    """
    Moving-average trend factor.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    ma_short = (
        df.groupby("ticker")[price_col]
        .rolling(short_window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ma_long = (
        df.groupby("ticker")[price_col]
        .rolling(long_window)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"trend_{short_window}_{long_window}d"] = ma_short / ma_long - 1

    return df


def add_price_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Build a basic but useful A-share daily factor set."""
    df = add_daily_return(df)
    df = add_forward_return(df, horizon=5)
    df = add_forward_return(df, horizon=20)
    df = add_reversal_factor(df, window=20)
    df = add_momentum_factor(df, lookback=120, skip=20)
    df = add_volatility_factor(df, window=20)
    df = add_volatility_factor(df, window=60)
    df = add_liquidity_factor(df, window=20)
    df = add_turnover_factor(df, window=20)
    df = add_trend_factor(df, short_window=20, long_window=120)

    df = df.sort_values(["ticker", "date"]).copy()
    close = df.groupby("ticker")["close"]
    high = df.groupby("ticker")["high"]
    low = df.groupby("ticker")["low"]
    volume = df.groupby("ticker")["volume"]

    df["amplitude_20d"] = (
        high.rolling(20).max().reset_index(level=0, drop=True)
        / low.rolling(20).min().reset_index(level=0, drop=True)
        - 1
    )
    df["volume_ratio_20d"] = df["volume"] / volume.rolling(20).mean().reset_index(level=0, drop=True)
    df["price_to_120d_high"] = close.transform(lambda s: s / s.rolling(120).max())
    df["log_amount"] = np.log1p(df["amount"])
    return df


def get_default_factor_cols() -> list[str]:
    """Return the default factor columns created by add_price_volume_factors."""
    return [
        "rev_20d",
        "mom_120_20d",
        "lowvol_20d",
        "lowvol_60d",
        "liq_20d",
        "turnover_20d",
        "trend_20_120d",
        "amplitude_20d",
        "volume_ratio_20d",
        "price_to_120d_high",
        "log_amount",
    ]


def winsorize_by_date(
    df: pd.DataFrame,
    cols: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Clip factor outliers cross-sectionally by date."""
    df = df.copy()
    for col in cols:
        bounds = df.groupby("date")[col].quantile([lower, upper]).unstack()
        bounds.columns = ["lower", "upper"]
        df = df.join(bounds, on="date")
        df[col] = df[col].clip(df["lower"], df["upper"])
        df = df.drop(columns=["lower", "upper"])
    return df


def zscore_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectionally z-score factor columns by date."""
    df = df.copy()
    grouped = df.groupby("date")
    for col in cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0, np.nan)
        df[f"{col}_z"] = (df[col] - mean) / std
    return df


def neutralize_factors_by_date(
    df: pd.DataFrame,
    cols: list[str],
    industry_col: str = "industry",
    size_col: str = "float_market_cap",
    min_obs: int = 20,
) -> pd.DataFrame:
    """Neutralize factors against industry dummies and log market cap by date.

    If industry or market-cap columns are unavailable, the function uses the
    controls that exist. With no usable controls it returns the input unchanged.
    """
    control_cols = []
    if industry_col in df.columns:
        control_cols.append(industry_col)
    elif "industry" in df.columns:
        industry_col = "industry"
        control_cols.append(industry_col)

    if size_col not in df.columns and "market_cap" in df.columns:
        size_col = "market_cap"
    if size_col in df.columns:
        control_cols.append(size_col)

    if not control_cols:
        return df

    out = df.copy()
    for date, idx in out.groupby("date").groups.items():
        group = out.loc[idx]
        controls = pd.DataFrame(index=group.index)

        if industry_col in group.columns:
            dummies = pd.get_dummies(group[industry_col].fillna("unknown"), prefix="ind", dtype=float)
            if len(dummies.columns) > 1:
                controls = controls.join(dummies.iloc[:, 1:])

        if size_col in group.columns:
            size = pd.to_numeric(group[size_col], errors="coerce")
            controls["log_size"] = np.log(size.where(size > 0))

        controls = controls.replace([np.inf, -np.inf], np.nan)
        if controls.empty:
            continue

        for col in cols:
            y = pd.to_numeric(group[col], errors="coerce")
            data = controls.copy()
            data["_y"] = y
            data = data.dropna()
            if len(data) < max(min_obs, len(controls.columns) + 2):
                continue

            x = np.column_stack([np.ones(len(data)), data[controls.columns].to_numpy(dtype=float)])
            beta = np.linalg.lstsq(x, data["_y"].to_numpy(dtype=float), rcond=None)[0]
            fitted = x @ beta
            out.loc[data.index, col] = data["_y"] - fitted

    return out


def build_factor_panel(
    df: pd.DataFrame,
    factor_cols: list[str] | None = None,
    winsorize: bool = True,
    neutralize: bool = False,
    industry_col: str = "industry",
    size_col: str = "float_market_cap",
    zscore: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create factors, optional winsorization, and z-scored model inputs."""
    panel = add_price_volume_factors(df)
    factor_cols = factor_cols or get_default_factor_cols()

    if winsorize:
        panel = winsorize_by_date(panel, factor_cols)
    if neutralize:
        panel = neutralize_factors_by_date(
            panel,
            factor_cols,
            industry_col=industry_col,
            size_col=size_col,
        )
    if zscore:
        panel = zscore_by_date(panel, factor_cols)
        factor_cols = [f"{col}_z" for col in factor_cols]

    return panel, factor_cols
