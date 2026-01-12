import os
import datetime
import pandas as pd
import tushare as ts
from dotenv import load_dotenv
import time
import random
import numpy as np
import json


# ========== 基本配置 ==========
load_dotenv()
TUSHARE_TOKEN = os.getenv("TUSHARE_KEY")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

TODAY = datetime.date.today().strftime("%Y%m%d")
OUTPUT_PATH ="A"
# ========= HDF5 Schema Cache =========
_HDF_SCHEMA_CACHE = {}


def safe_append(store, key, df, data_columns=None):
    if df is None or df.empty:
        return

    if key not in _HDF_SCHEMA_CACHE:
        raise RuntimeError(
            f"[FATAL] schema for key={key} not initialized. "
            f"Init _HDF_SCHEMA_CACHE['{key}'] before calling safe_append."
        )

    schema = _HDF_SCHEMA_CACHE[key]
    cols = schema["cols"]
    dtypes = schema["dtypes"]

    # 1) 补缺列
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan

    # 2) 丢弃多余列 + 固定列顺序
    df = df[cols]

    # 3) dtype 强制（写库前兜底）
    for col, dtype in dtypes.items():
        if dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "object":
            df[col] = df[col].astype("object")

    store.append(
        key,
        df,
        format="table",
        data_columns=data_columns
    )

def load_selected_fin_cols(json_path):
    """
    从 preanalysis 生成的 json 中读取有效因子
    并据此唯一确定 financials 的 schema
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cols = json.load(f)

    base_cols = ["ts_code", "end_date"]
    cols = [c for c in cols if c not in base_cols]
    fin_cols = base_cols + cols

    dtypes = {}
    for col in fin_cols:
        if col in ("ts_code", "end_date"):
            dtypes[col] = "object"
        else:
            dtypes[col] = "float64"

    _HDF_SCHEMA_CACHE["financials"] = {"cols": fin_cols, "dtypes": dtypes}

    print(f"[SCHEMA] financials schema set by load_selected_fin_cols (n_cols={len(fin_cols)})")
    return fin_cols


def infer_all_fin_cols(sample_ts="600000.SH"):
    """
    拉一条样本，推断 fina_indicator 的全部字段
    并据此唯一确定 financials 的 schema
    """
    df = pro.fina_indicator(ts_code=sample_ts, limit=1)
    if df is None or df.empty:
        raise RuntimeError(f"[FATAL] infer_all_fin_cols failed on sample_ts={sample_ts}")

    fin_cols = df.columns.tolist()

    dtypes = {}
    for col in fin_cols:
        if col in ("ts_code", "trade_date", "end_date"):
            dtypes[col] = "object"
        else:
            dtypes[col] = "float64"

    _HDF_SCHEMA_CACHE["financials"] = {"cols": fin_cols, "dtypes": dtypes}

    print(f"[SCHEMA] financials schema set by infer_all_fin_cols (n_cols={len(fin_cols)})")
    return fin_cols



def safe_call(func, *args, max_retry=5, sleep_seconds=60, **kwargs):
    """
    带限速处理的 tushare 调用包装
    """
    for attempt in range(1, max_retry + 1):
        try:
            df = func(*args, **kwargs)
            if (
                df is not None
                and not df.empty
                and "ts_code" in df.columns
            ):
                return df
            else:
                print(f"[WARN] empty response, attempt={attempt}")
        except Exception as e:
            print(f"[WARN] tushare error: {e}, attempt={attempt}")

        if attempt < max_retry:
            wait = sleep_seconds + random.uniform(0, 5)
            print(f"[INFO] sleeping {wait:.1f}s due to rate limit")
            time.sleep(wait)

    return pd.DataFrame()

def load_completed_ts_codes(h5_path, key="completed"):
    if not os.path.exists(h5_path):
        return set()

    try:
        df = pd.read_hdf(h5_path, key=key)
        return set(df["ts_code"].dropna().unique())
    except (KeyError, FileNotFoundError):
        return set()
    
# ========== 1. 获取股票池 ==========
def fetch_stock_basic():
    """
    获取当前上市的 A 股股票池
    """
    df = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date"
    )
    return df


# ========== 2. 获取 Tushare 已计算的可靠因子 ==========
def fetch_daily_basic(ts_codes, trade_date):
    """
    daily_basic 正确姿势：
    - 一次性拉取全市场某交易日
    - 本地按 ts_codes 过滤
    """
    print("[INFO] fetching daily_basic for whole market")

    df = safe_call(
        pro.daily_basic,
        trade_date=trade_date,
        fields=(
            "ts_code,trade_date,"
            "close,turnover_rate,volume_ratio,"
            "pe,pe_ttm,pb,ps,ps_ttm,"
            "dv_ratio,dv_ttm,"
            "total_mv,float_mv"
        )
    )

    if df.empty:
        return df

    # ★ 只保留本次需要的股票
    df = df[df["ts_code"].isin(ts_codes)].reset_index(drop=True)
    return df



# ========== 3. 获取财务因子（ROE / ROA / 杠杆） ==========
def fetch_financial_factors_stream(
    ts_codes,
    store,
    fin_cols,
    batch_size=30
):
    """
    财务因子流式爬取（支持全量 / 精选）
    """
    for i in range(0, len(ts_codes), batch_size):
        batch = ts_codes[i:i + batch_size]
        print(f"[INFO] fina_indicator batch {i} - {i + len(batch)}")

        for code in batch:
            df = safe_call(
                pro.fina_indicator,
                ts_code=code,
                fields=",".join(fin_cols)
            )

            if df is None or df.empty:
                continue

            # —— 强制 schema 对齐（关键） ——
            for col in fin_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            df = df[fin_cols]
            # 只保留最新一期（按 end_date 倒序取第一行）
            if "end_date" in df.columns:
                df = df.sort_values("end_date", ascending=False).head(1).reset_index(drop=True)
            else:
                # 没有 end_date 的 schema 就只能保留第一行
                df = df.head(1).reset_index(drop=True)

            safe_append(
                store,
                key="financials",
                df=df,
                data_columns=["ts_code"]
            )


            safe_append(
                store,
                key="completed",
                df=pd.DataFrame({"ts_code": [code]}),
                data_columns=["ts_code"]
            )

            del df

        time.sleep(1.5)




# ========== 4. 因子扩展 ==========
def build_derived_factors(df):
    """
    构造常用派生因子
    """
    df = df.copy()

    # Size
    df["log_total_mv"] = df["total_mv"].apply(
        lambda x: np.nan if x <= 0 else np.log(x)
    )

    # Value
    df["inv_pb"] = df["pb"].apply(
        lambda x: np.nan if x <= 0 else 1.0 / x
    )
    df["inv_pe_ttm"] = df["pe_ttm"].apply(
        lambda x: np.nan if x <= 0 else 1.0 / x
    )

    # Yield
    df["dividend_yield"] = df["dv_ttm"]

    # Quality
    df["roe_roa_gap"] = df["roe"] - df["roa"]

    return df

def get_effective_trade_date(cutoff_hour=15, lookback_days=30):
    now = datetime.datetime.now()

    if now.hour < cutoff_hour:
        end_date = (now.date() - datetime.timedelta(days=1)).strftime("%Y%m%d")
    else:
        end_date = now.date().strftime("%Y%m%d")

    start_date = (
        datetime.datetime.strptime(end_date, "%Y%m%d")
        - datetime.timedelta(days=lookback_days)
    ).strftime("%Y%m%d")

    cal = pro.trade_cal(
        exchange="SSE",
        is_open=1,
        start_date=start_date,
        end_date=end_date
    )

    if cal.empty:
        raise RuntimeError("[FATAL] trade_cal returned empty")

    cal = cal.sort_values("cal_date")
    return cal.iloc[-1]["cal_date"]

def print_schema_brief(schema_dict, max_show=10, head=5, tail=5):
    """
    精简打印 HDF schema：
    - 每个 key 单独一段
    - 列数 <= max_show: 全部打印
    - 否则: 前 head + 后 tail
    """
    print(">>> HDF5 schema summary:")

    for key, schema in schema_dict.items():
        cols = schema.get("cols", [])
        n = len(cols)

        print(f"\n[{key}] n_cols={n}")

        if n <= max_show:
            print("  cols:", cols)
        else:
            shown = cols[:head] + ["..."] + cols[-tail:]
            print("  cols:", shown)


# ========== 5. 主流程 ==========
def main(
    trade_date=None,
    fin_mode="all",   # "selected" | "all"
    fin_json_path="columns.json"
):
    print(">>> 获取股票池")
    stock_basic = fetch_stock_basic()
    if trade_date is None:
        trade_date = get_effective_trade_date()
        print(f">>> 使用有效交易日: {trade_date}")
    # 确保输出目录存在
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    out_path = os.path.join(OUTPUT_PATH, f"stock_factors_{trade_date}.h5")
    completed_ts_codes = load_completed_ts_codes(out_path)
    all_ts_codes = stock_basic["ts_code"].tolist()
    ts_codes = [c for c in all_ts_codes if c not in completed_ts_codes]

    print(f">>> 股票总数: {len(all_ts_codes)}")
    print(f">>> 已存在: {len(completed_ts_codes)}")
    print(f">>> 待爬取: {len(ts_codes)}")

    if not ts_codes:
        print(">>> 今日股票已全部爬取，退出")
        return

    with pd.HDFStore(out_path, mode="a") as store:

        # --- daily_basic（这一步可以一次性，问题不大） ---
        if trade_date is None:
            trade_date = TODAY

        # === 选择财务因子模式 ===
        if fin_mode == "all":
            print(">>> 使用【全量财务因子】模式")
            fin_cols = infer_all_fin_cols()
        elif fin_mode == "selected":
            print(">>> 使用【精选财务因子】模式")
            fin_cols = load_selected_fin_cols(fin_json_path)
        else:
            raise ValueError(f"unknown fin_mode: {fin_mode}")


        # daily_basic schema（和 fetch_daily_basic fields 一致）
        daily_cols = [
            "ts_code","trade_date",
            "close","turnover_rate","volume_ratio",
            "pe","pe_ttm","pb","ps","ps_ttm",
            "dv_ratio","dv_ttm",
            "total_mv","float_mv"
        ]
        _HDF_SCHEMA_CACHE["daily_basic"] = {
            "cols": daily_cols,
            "dtypes": {c: ("object" if c in ("ts_code","trade_date") else "float64") for c in daily_cols}
        }

        # completed schema（就一列，别整花活）
        _HDF_SCHEMA_CACHE["completed"] = {
            "cols": ["ts_code"],
            "dtypes": {"ts_code": "object"}
        }


        print(f">>> financial columns: {len(fin_cols)}")
        print_schema_brief(_HDF_SCHEMA_CACHE)


        daily_basic = fetch_daily_basic(ts_codes, trade_date)
        if daily_basic.empty:
            raise RuntimeError(
                f"[FATAL] daily_basic empty on trade_date={trade_date}"
            )
        safe_append(store, "daily_basic", daily_basic, ["ts_code"])
        del daily_basic
        

        # --- 财务因子：流式写入 ---
        print(">>> 获取财务因子（流式写入）")
        fetch_financial_factors_stream(
            ts_codes=ts_codes,
            store=store,
            fin_cols=fin_cols
        )


    print(">>> 本批次完成")


if __name__ == "__main__":
    main()
