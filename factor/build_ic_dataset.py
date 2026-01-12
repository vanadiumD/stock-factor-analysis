# build_ic_dataset.py
import os
import gc
import time
import traceback
import pandas as pd
from datetime import datetime, timedelta

from fetch_a_tushare import main as fetch_tushare_main
import logging

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "build_ic_dataset.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def log(msg):
    print(msg)
    logging.info(msg)


RAW_DIR = "A"
FACTOR_DIR = "factor"
TIME_FACTOR_DIR = "factor/time_factors"


# =========================
# 日期工具
# =========================
def today_trade_date():
    now = datetime.now()
    # 15:00 前默认没收盘
    if now.hour < 15:
        now -= timedelta(days=1)
    return now.strftime("%Y%m%d")


# =========================
# 股票池（只加载一次）
# =========================
def load_universe():
    path = f"{FACTOR_DIR}/purify.h5"
    if not os.path.exists(path):
        raise FileNotFoundError("purify.h5 not found")

    df = pd.read_hdf(path, key="factors", columns=["ts_code"])
    universe = set(df["ts_code"].dropna().unique())
    log(f">>> universe loaded: {len(universe)} stocks")
    return universe


# =========================
# as-of 财务因子
# =========================
def latest_financial_asof(df, trade_date):
    df = df[df["end_date"] <= trade_date]
    if df.empty:
        return None
    return df.sort_values("end_date").iloc[-1]


# =========================
# 简化版扁平化（核心）
# =========================
def flatten_one_day(df_raw, trade_date, universe):
    # ★ 关键：只按 purify 得到的 universe 做过滤
    df_raw = df_raw[df_raw["ts_code"].isin(universe)]

    records = []

    for ts_code, g in df_raw.groupby("ts_code"):
        fin = latest_financial_asof(g, trade_date)
        if fin is None:
            # 允许该股票这一天缺失
            continue

        out = {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "end_date": fin["end_date"],
            # 财务因子（as-of）
            "roe": fin.get("roe"),
            "roe_dt": fin.get("roe_dt"),
            "netprofit_margin": fin.get("netprofit_margin"),
            "debt_to_assets": fin.get("debt_to_assets"),
            "ocf_to_or": fin.get("ocf_to_or"),
        }

        # 当日行情（允许缺失）
        today = g[g["trade_date"] == trade_date]
        if not today.empty:
            row = today.iloc[0]
            for col in [
                "close",
                "pe_ttm",
                "pb",
                "total_mv",
                "turnover_rate",
                "name",
                "industry",
                "market",
            ]:
                if col in row:
                    out[col] = row[col]

        records.append(out)

    return pd.DataFrame(records)



# =========================
# 单日构建
# =========================
def build_one_day(date, universe):
    out_path = f"{TIME_FACTOR_DIR}/latest_factors_{date}.h5"
    if os.path.exists(out_path):
        log(f">>> {date} already done, skip")
        return

    raw_path = f"{RAW_DIR}/stock_factors_{date}.h5"

    if not os.path.exists(raw_path):
        log(f">>> raw missing for {date}, fetching from tushare")
        fetch_tushare_main(trade_date=date)

    if not os.path.exists(raw_path):
        log(f"[WARN] fetch failed for {date}, skip")
        return

    df_raw = pd.read_hdf(raw_path, key="factors")
    df = flatten_one_day(df_raw, date, universe)

    if df.empty:
        log(f"[WARN] empty snapshot for {date}")
        return

    os.makedirs(TIME_FACTOR_DIR, exist_ok=True)
    df.to_hdf(
        out_path,
        key="factors",
        mode="w",
        format="table",
        data_columns=["ts_code"]
    )

    log(f">>> saved {out_path}, shape={df.shape}")
    del df, df_raw
    gc.collect()


# =========================
# 主流程
# =========================
def main(max_back_days=500):
    universe = load_universe()
    start_date = today_trade_date()

    cur = datetime.strptime(start_date, "%Y%m%d")

    for _ in range(max_back_days):
        date = cur.strftime("%Y%m%d")
        try:
            log(f"\n=== processing {date} ===")
            build_one_day(date, universe)
        except KeyboardInterrupt:
            log("\n>>> interrupted, safe exit")
            break
        except Exception:
            log(f"[FATAL] error on {date}")
            traceback.log_exc()
            time.sleep(3)

        cur -= timedelta(days=1)


if __name__ == "__main__":
    main()
