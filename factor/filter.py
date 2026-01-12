
import os
import argparse
import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseStockFilter(ABC):
    """股票筛选策略基类"""

    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class LooseStockFilter(BaseStockFilter):
    """宽松的投资宇宙筛选"""

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        print(">>> applying LOOSE stock filter")

        if "list_status" in df.columns:
            df = df[df["list_status"] == "L"]

        if "name" in df.columns:
            df = df[~df["name"].astype(str).str.contains("ST", na=False)]

        if "total_mv" in df.columns:
            s = pd.to_numeric(df["total_mv"], errors="coerce")
            threshold = s.quantile(0.01)
            df = df[s >= threshold]

        print(f">>> after LOOSE filter: {df['ts_code'].nunique()} stocks")
        return df


class StrictStockFilter(BaseStockFilter):
    """严格的投资宇宙筛选"""

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        print(">>> applying STRICT stock filter")

        if "list_status" in df.columns:
            df = df[df["list_status"] == "L"]

        if "name" in df.columns:
            df = df[~df["name"].astype(str).str.contains("ST", na=False)]

        for col in ["pb", "pe_ttm", "roe"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df[df[col] > 0]

        if "turnover_rate" in df.columns:
            df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce")
            df = df[df["turnover_rate"] > 0.1]

        if "total_mv" in df.columns:
            df["total_mv"] = pd.to_numeric(df["total_mv"], errors="coerce")
            df = df[df["total_mv"] > 1e6]

        print(f">>> after STRICT filter: {df['ts_code'].nunique()} stocks")
        return df


class MultiFactorPurifier:
    def __init__(self, input_path, output_path, stock_filter):
        self.input_path = input_path
        self.output_path = output_path
        self.stock_filter = stock_filter

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    def run(self):
        with pd.HDFStore(self.input_path, mode="r") as store_in,              pd.HDFStore(self.output_path, mode="w") as store_out:

            # 1. 读取 daily_basic 并筛选
            print(">>> loading /daily_basic")
            df_basic = store_in['/daily_basic']
            print(f"    original: {df_basic['ts_code'].nunique()} stocks, {len(df_basic)} rows")

            df_filtered = self.stock_filter.filter(df_basic)
            filtered_stocks = df_filtered["ts_code"].unique().tolist()
            print(f"    after filter: {len(filtered_stocks)} stocks, {len(df_filtered)} rows")

            store_out.put('/daily_basic', df_filtered, format='table', data_columns=["ts_code"])

            # 2. 筛选 financials
            if '/financials' in store_in.keys():
                print(">>> loading /financials")
                df_fin = store_in['/financials']
                print(f"    original: {df_fin['ts_code'].nunique()} stocks, {len(df_fin)} rows")

                df_fin_filtered = df_fin[df_fin["ts_code"].isin(filtered_stocks)]
                print(f"    after filter: {df_fin_filtered['ts_code'].nunique()} stocks, {len(df_fin_filtered)} rows")

                store_out.put('/financials', df_fin_filtered, format='table', data_columns=["ts_code"])

            # 3. 更新 completed
            print(">>> updating /completed")
            df_completed = pd.DataFrame({"ts_code": filtered_stocks})
            print(f"    writing {len(df_completed)} stocks to /completed")

            store_out.put('/completed', df_completed, format='table', data_columns=["ts_code"])


def build_argparser():
    p = argparse.ArgumentParser(description="Multi-key HDF5 factor purifier")

    p.add_argument("--input_path", type=str, default=None, help="input HDF5 path")
    p.add_argument("--output_path", type=str, default="factor/purified.h5", help="output HDF5 path")
    p.add_argument("--mode", type=str, choices=["loose", "strict", "none"], default="loose", help="stock filter mode")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.input_path is None:
        date = "20251219"
        args.input_path = f"A/stock_factors_{date}.h5"

    if args.mode == "loose":
        stock_filter = LooseStockFilter()
    elif args.mode == "strict":
        stock_filter = StrictStockFilter()
    else:
        stock_filter = None

    purifier = MultiFactorPurifier(
        input_path=args.input_path,
        output_path=args.output_path,
        stock_filter=stock_filter,
    )
    purifier.run()
