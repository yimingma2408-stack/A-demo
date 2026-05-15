from pathlib import Path
import time
import random

import baostock as bs
import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Global settings
# =========================================================

START_DATE = "20200101"
END_DATE = "20241231"
ADJUST = "qfq"

DATA_SOURCE = "baostock"

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
    "tradestatus",
    "is_st",
]


# =========================================================
# BaoStock login status
# =========================================================

_BAOSTOCK_LOGGED_IN = False


def baostock_login() -> None:
    """
    Login to BaoStock once.
    """
    global _BAOSTOCK_LOGGED_IN

    if _BAOSTOCK_LOGGED_IN:
        return

    lg = bs.login()
    print("BaoStock login:", lg.error_code, lg.error_msg)

    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

    _BAOSTOCK_LOGGED_IN = True


def baostock_logout() -> None:
    """
    Logout from BaoStock.
    """
    global _BAOSTOCK_LOGGED_IN

    if _BAOSTOCK_LOGGED_IN:
        bs.logout()
        _BAOSTOCK_LOGGED_IN = False


# =========================================================
# Helper functions
# =========================================================

def normalize_date(date_str: str) -> str:
    """
    Convert YYYYMMDD or YYYY-MM-DD to YYYY-MM-DD.

    BaoStock requires dates in YYYY-MM-DD format.
    """
    date_str = str(date_str)

    if "-" in date_str:
        return date_str

    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"

    raise ValueError(f"Invalid date format: {date_str}")


def stock_code_to_baostock_code(stock_code: str) -> str:
    """
    Convert normal A-share stock code to BaoStock code.

    Examples
    --------
    600519 -> sh.600519
    000001 -> sz.000001
    300750 -> sz.300750
    """
    stock_code = str(stock_code).zfill(6)

    if stock_code.startswith("6"):
        return f"sh.{stock_code}"

    if stock_code.startswith(("0", "2", "3")):
        return f"sz.{stock_code}"

    raise ValueError(f"Unsupported A-share stock code: {stock_code}")


def baostock_code_to_ticker(bs_code: str) -> str:
    """
    Convert BaoStock code to normal ticker.

    Examples
    --------
    sh.600519 -> 600519
    sz.000001 -> 000001
    """
    return str(bs_code).split(".")[-1].zfill(6)


def adjust_to_baostock_flag(adjust: str) -> str:
    """
    Convert adjustment type to BaoStock adjustflag.

    BaoStock adjustflag:
        1: 后复权
        2: 前复权
        3: 不复权
    """
    adjust = adjust.lower()

    if adjust in ["qfq", "forward"]:
        return "2"

    if adjust in ["hfq", "backward"]:
        return "1"

    if adjust in ["none", "raw", "bfq", ""]:
        return "3"

    raise ValueError(f"Unknown adjust type: {adjust}")



# =========================================================
# Single-stock downloader
# =========================================================

def download_one_stock(
    stock_code: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    adjust: str = ADJUST,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    """
    Download daily A-share data for one stock from BaoStock.

    Parameters
    ----------
    stock_code:
        Stock code, e.g. "000001", "600519".
    start_date:
        Start date in YYYYMMDD or YYYY-MM-DD format.
    end_date:
        End date in YYYYMMDD or YYYY-MM-DD format.
    adjust:
        Adjustment type.
        "qfq" means forward-adjusted price.
        "hfq" means backward-adjusted price.
        "none" means unadjusted price.
    max_retries:
        Maximum number of retry attempts.
    sleep_seconds:
        Waiting time between retries.

    Returns
    -------
    pd.DataFrame
        Standardized stock data with columns:
        date, ticker, open, high, low, close, volume, amount, pct_change, turnover.
    """
    baostock_login()

    stock_code = str(stock_code).zfill(6)
    bs_code = stock_code_to_baostock_code(stock_code)

    start_date = normalize_date(start_date)
    end_date = normalize_date(end_date)
    adjustflag = adjust_to_baostock_flag(adjust)

    fields = (
        "date,code,open,high,low,close,"
        "volume,amount,turn,pctChg,tradestatus,isST"
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{stock_code}] Download attempt {attempt}/{max_retries}...")

            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjustflag,
            )

            if rs.error_code != "0":
                raise RuntimeError(
                    f"BaoStock error_code={rs.error_code}, error_msg={rs.error_msg}"
                )

            data = []
            while rs.next():
                data.append(rs.get_row_data())

            df = pd.DataFrame(data, columns=rs.fields)

            print(f"[{stock_code}] Raw shape: {df.shape}")
            print(f"[{stock_code}] Raw columns: {list(df.columns)}")

            if df.empty:
                print(f"[{stock_code}] Empty data returned.")
                time.sleep(sleep_seconds + random.random())
                continue

            rename_dict = {
                "date": "date",
                "code": "ticker",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount",
                "pctChg": "pct_change",
                "turn": "turnover",
                "isST": "is_st",
            }

            df = df.rename(columns=rename_dict)

            df["ticker"] = df["ticker"].map(baostock_code_to_ticker)

            missing_cols = [col for col in STANDARD_COLS if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"[{stock_code}] Missing columns after renaming: {missing_cols}. "
                    f"Current columns: {list(df.columns)}"
                )

            df = df[STANDARD_COLS].copy()

            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = df["ticker"].astype(str).str.zfill(6)

            numeric_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "pct_change",
                "turnover",
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

            return df

        except Exception as exc:
            print(f"[{stock_code}] Error: {exc}")
            time.sleep(sleep_seconds + random.random())

    print(f"[{stock_code}] Failed after {max_retries} attempts.")
    return pd.DataFrame(columns=STANDARD_COLS)


# =========================================================
# Local cache loader
# =========================================================

def load_or_download_one_stock(
    stock_code: str,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    adjust: str = ADJUST,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load stock data from local CSV if available.
    Otherwise download from BaoStock.

    Parameters
    ----------
    stock_code:
        Stock code.
    start_date:
        Start date in YYYYMMDD or YYYY-MM-DD format.
    end_date:
        End date in YYYYMMDD or YYYY-MM-DD format.
    adjust:
        Adjustment type.
    force_download:
        If True, ignore local file and re-download.

    Returns
    -------
    pd.DataFrame
        Standardized stock data.
    """
    stock_code = str(stock_code).zfill(6)
    output_file = DATA_DIR / f"{stock_code}_daily_{adjust}_{DATA_SOURCE}.csv"

    if output_file.exists() and not force_download:
        print(f"[{stock_code}] Local file exists. Loading from local CSV.")

        try:
            df = pd.read_csv(output_file)
            df["date"] = pd.to_datetime(df["date"])
            df["ticker"] = df["ticker"].astype(str).str.zfill(6)

            missing_cols = [col for col in STANDARD_COLS if col not in df.columns]
            if missing_cols:
                print(f"[{stock_code}] Local file missing columns: {missing_cols}. Re-downloading.")
            elif df.empty:
                print(f"[{stock_code}] Local file is empty. Re-downloading.")
            else:
                return df[STANDARD_COLS].copy()

        except Exception as exc:
            print(f"[{stock_code}] Failed to read local file: {exc}. Re-downloading.")

    df = download_one_stock(
        stock_code=stock_code,
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
    )

    if not df.empty:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"[{stock_code}] Saved to {output_file}")

    return df


# =========================================================
# Panel downloader
# =========================================================

def download_stock_panel(
    symbols: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    adjust: str = ADJUST,
    force_download: bool = False,
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Download or load a panel of A-share stocks.

    Parameters
    ----------
    symbols:
        List of stock codes.
    start_date:
        Start date in YYYYMMDD or YYYY-MM-DD format.
    end_date:
        End date in YYYYMMDD or YYYY-MM-DD format.
    adjust:
        Adjustment type.
    force_download:
        If True, re-download all stocks.
    sleep_seconds:
        Waiting time between stocks.

    Returns
    -------
    pd.DataFrame
        Panel data of all successfully downloaded stocks.
    """
    data_list = []

    baostock_login()

    try:
        for symbol in symbols:
            symbol = str(symbol).zfill(6)

            print("=" * 80)
            print(f"Processing {symbol}")

            df = load_or_download_one_stock(
                stock_code=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
                force_download=force_download,
            )

            if df.empty:
                print(f"[{symbol}] Skipped because data is empty.")
                continue

            data_list.append(df)

            time.sleep(sleep_seconds + random.random())

    finally:
        baostock_logout()

    if not data_list:
        raise RuntimeError(
            "No valid stock data was downloaded or loaded. "
            "Please test one stock first, check network, BaoStock, stock codes, and date format."
        )

    panel = pd.concat(data_list, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel_file = DATA_DIR / f"stocks_panel_daily_{adjust}_{DATA_SOURCE}.csv"
    panel.to_csv(panel_file, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Panel data saved to:", panel_file)

    print("\nPanel head:")
    print(panel.head())

    print("\nPanel summary:")
    summary = (
        panel.groupby("ticker")["date"]
        .agg(
            count="count",
            min_date="min",
            max_date="max",
        )
    )
    print(summary)

    return panel


# # =========================================================
# # Main
# # =========================================================

# def main() -> None:
#     symbols = [
#         # 金融
#         "000001",  # 平安银行
#         "600000",  # 浦发银行
#         "600036",  # 招商银行
#         "601318",  # 中国平安
#         "600030",  # 中信证券

#         # 消费
#         "600519",  # 贵州茅台
#         "000858",  # 五粮液
#         "000333",  # 美的集团
#         "000651",  # 格力电器
#         "600887",  # 伊利股份

#         # 医药
#         "600276",  # 恒瑞医药
#         "300760",  # 迈瑞医疗
#         "000538",  # 云南白药

#         # 科技与电子
#         "002415",  # 海康威视
#         "000725",  # 京东方A
#         "002475",  # 立讯精密
#         "603501",  # 韦尔股份
#         "000063",  # 中兴通讯

#         # 新能源与汽车
#         "300750",  # 宁德时代
#         "002594",  # 比亚迪
#         "601012",  # 隆基绿能
#         "600438",  # 通威股份

#         # 工业与材料
#         "601668",  # 中国建筑
#         "600309",  # 万华化学
#         "600031",  # 三一重工
#         "601899",  # 紫金矿业

#         # 能源与公用事业
#         "601088",  # 中国神华
#         "600028",  # 中国石化
#         "601857",  # 中国石油
#         "600900",  # 长江电力
#     ]

    

#     panel = download_stock_panel(
#         symbols=symbols,
#         start_date="20150101",
#         end_date="20251231",
#         adjust="qfq",
#         force_download=False,
#         sleep_seconds=1.0,
#     )

#     print("\nDownload completed.")
#     print("Panel shape:", panel.shape)


# =========================================================
# HS300 constituent loader
# =========================================================

# =========================================================
# HS300 constituent loader
# =========================================================

def get_hs300_symbols(index_date: str | None = None) -> list[str]:
    """
    Get CSI 300 / HS300 constituent stock codes from BaoStock.

    Parameters
    ----------
    index_date:
        Date in YYYYMMDD or YYYY-MM-DD format.
        If None, BaoStock returns the latest available HS300 constituents.
        Some BaoStock versions may not support the date argument, so this
        function falls back to query_hs300_stocks() automatically.

    Returns
    -------
    list[str]
        Normal 6-digit stock codes, e.g. ["600519", "000001"].
    """
    baostock_login()

    if index_date is not None:
        index_date = normalize_date(index_date)

    try:
        if index_date is None:
            rs = bs.query_hs300_stocks()
        else:
            try:
                rs = bs.query_hs300_stocks(date=index_date)
            except TypeError:
                print(
                    "[HS300] Current BaoStock version does not support "
                    "date argument. Falling back to latest constituents."
                )
                rs = bs.query_hs300_stocks()

        if rs.error_code != "0":
            raise RuntimeError(
                f"BaoStock error_code={rs.error_code}, error_msg={rs.error_msg}"
            )

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        df = pd.DataFrame(data, columns=rs.fields)

        if df.empty:
            raise RuntimeError("Empty HS300 constituent list returned by BaoStock.")

        print("[HS300] Raw columns:", list(df.columns))
        print("[HS300] Raw shape:", df.shape)

        # Usually BaoStock returns columns like:
        # date, code, code_name
        if "code" not in df.columns:
            raise ValueError(f"'code' column not found. Current columns: {list(df.columns)}")

        df["ticker"] = df["code"].astype(str).str.split(".").str[-1].str.zfill(6)

        symbols = sorted(df["ticker"].unique().tolist())

        # Save constituent table for reproducibility.
        if index_date is None:
            output_file = DATA_DIR / "hs300_constituents_latest_baostock.csv"
        else:
            output_file = DATA_DIR / f"hs300_constituents_{index_date.replace('-', '')}_baostock.csv"

        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"[HS300] Constituents saved to: {output_file}")
        print(f"[HS300] Number of stocks: {len(symbols)}")

        return symbols

    except Exception as exc:
        raise RuntimeError(f"Failed to get HS300 symbols: {exc}") from exc


def main() -> None:
    # 如果你想获取最新沪深 300 成分股：
    symbols = get_hs300_symbols(index_date=20251231)

    # 如果你想固定为某个日期的沪深 300 成分股，例如 2024 年末：
    # symbols = get_hs300_symbols(index_date="20241231")

    print("\nHS300 symbols:")
    print(symbols)

    panel = download_stock_panel(
        symbols=symbols,
        start_date="20160101",
        end_date="20251231",
        adjust="qfq",
        force_download=False,
        sleep_seconds=1.0,
    )

    print("\nDownload completed.")
    print("Panel shape:", panel.shape)





if __name__ == "__main__":
    main()
