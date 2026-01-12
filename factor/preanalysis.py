import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from datetime import datetime
import json
import re
import shutil
import argparse

ID_COLS = {
    "ts_code",
    "ann_date",
    "end_date",
    "trade_date",
    "name",
    "industry",
    "market",
}


def resolve_date_tag(date_str: str | None = None) -> str:
    """Resolve run date tag (YYYYMMDD) with priority:
    1) explicit date_str arg
    2) env RUN_DATE or DATE_TAG
    3) today's date
    """
    if date_str:
        return str(date_str)
    env = os.getenv("RUN_DATE") or os.getenv("DATE_TAG")
    if env:
        return str(env)
    return datetime.now().strftime("%Y%m%d")


def today_str(date_str: str | None = None) -> str:
    # backward-compatible helper
    return resolve_date_tag(date_str)


def get_cache_path(prefix="factor/time_factors/time_factors", date_tag: str | None = None):
    tag = resolve_date_tag(date_tag)
    return f"{prefix}_{tag}.h5"

def infer_lookback(g, user_lookback=None, min_q=4, max_q=None):
    """
    根据财报数量自动决定回溯窗口
    """
    if user_lookback is not None:
        return user_lookback

    n = g["end_date"].nunique()

    if max_q is not None and n >= max_q:
        return max_q
    elif min_q is not None and n <= min_q:
        return min_q
    else:
        return n


def fit_trend(y):
    t = np.arange(len(y))
    a, b = np.polyfit(t, y, 1)
    y_hat = a * t + b
    r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
    return a, r2

"""
def build_time_factors(df_raw, lookback=None):
    records = []

    for ts_code, g in df_raw.groupby("ts_code"):
        g = g.sort_values("end_date")

        lb = infer_lookback(g, user_lookback=lookback)
        g_lb = g.tail(lb)

        out = {
            "ts_code": ts_code,
            "lookback_q": lb,   # 很重要，方便你之后分析
        }

        for col in [
            "roe",
            "roe_dt",
            "netprofit_margin",
            "debt_to_assets",
            "ocf_to_or",
        ]:
            y = g_lb[col].dropna().values

            if len(y) == 0:
                out[f"{col}_mean_q"] = np.nan
                out[f"{col}_std_q"] = np.nan
                out[f"{col}_slope_q"] = np.nan
                out[f"{col}_r2_q"] = np.nan
                continue

            out[f"{col}_mean_q"] = np.mean(y)
            out[f"{col}_std_q"] = np.std(y)

            if len(y) >= 2:
                a, r2 = fit_trend(y)
                out[f"{col}_slope_q"] = a
                out[f"{col}_r2_q"] = r2
            else:
                out[f"{col}_slope_q"] = np.nan
                out[f"{col}_r2_q"] = np.nan

        # —— 拼接最新一期的“静态因子” ——
        last = g.iloc[-1]
        for col in [
            "trade_date",
            "close",
            "pe_ttm",
            "pb",
            "total_mv",
            "turnover_rate",
            "name",
            "industry",
            "market",
        ]:
            if col in last:
                out[col] = last[col]

        records.append(out)

    return pd.DataFrame(records)

def load_or_build_time_factors_cached(
    input_path,
    key,
    lookback=None,
    cache_prefix="factor/time_factors/time_factors"
):
    cache_path = get_cache_path(cache_prefix)

    if os.path.exists(cache_path):
        print(f">>> loading cached time factors: {cache_path}")
        return pd.read_hdf(cache_path, key="factors")

    print(">>> building time factors (no cache found)")

    df_raw = pd.read_hdf(input_path, key=key)
    
    df = build_time_factors(df_raw, lookback=lookback)

    print(f">>> saving time factors to {cache_path}")
    df.to_hdf(cache_path, key="factors", mode="w")

    return df
"""

def factor_audit_table(df, cols):
    rows = []

    n = len(df)

    for col in cols:
        s = df[col]

        valid = s.notna().sum()
        zero = (s == 0).sum()
        unique = s.nunique(dropna=True)

        rows.append({
            "factor": col,
            "valid_ratio": valid / n,
            "nan_ratio": 1 - (valid / n),
            # zero_ratio is "zero among valid" (kept for backward compatibility)
            "zero_ratio": zero / valid if valid > 0 else np.nan,
            "zero_all_ratio": zero / n,
            "unique_count": unique,
            "mean": s.mean(),
            "std": s.std(),
        })

    audit = pd.DataFrame(rows).set_index("factor")
    return audit.sort_values("valid_ratio")


def clip_series(s, q=0.01):
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def apply_factor_dict(df, factor_dict=None, id_cols=ID_COLS):
    """
    读取时可选地只保留 factor_dict 中指定的因子（默认 None=全部）。
    factor_dict: dict(因子名->任意) 或 list/tuple/set(因子名)。
    """
    if factor_dict is None:
        return df

    if isinstance(factor_dict, str):
        # 支持传入 JSON 路径
        with open(factor_dict, "r", encoding="utf-8") as f:
            factor_dict = json.load(f)

    if isinstance(factor_dict, dict):
        wanted = list(factor_dict.keys())
    else:
        wanted = list(factor_dict)

    keep = [c for c in df.columns if (c in id_cols) or (c in wanted)]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f">>> [warn] {len(missing)} factors not found in df, ignored (showing first 20): {missing[:20]}")
    return df[keep]

import os
import re

def delete_histogram_pngs(
    out_dir: str,
    date_str=None,
):
    """
    只删除形如：
    factor_histograms_YYYYMMDD_pX.png
    的文件，不碰目录，不碰其它文件
    """
    out_dir = os.path.abspath(out_dir)

    if not os.path.isdir(out_dir):
        print(f">>> [warn] hist dir not found: {out_dir}")
        return

    if date_str is None:
        date_str = today_str(date_str)

    pattern = re.compile(
        rf"^factor_histograms_{date_str}_p\d+\.png$"
    )

    removed = []
    for fname in os.listdir(out_dir):
        if pattern.match(fname):
            fpath = os.path.join(out_dir, fname)
            try:
                os.remove(fpath)
                removed.append(fname)
            except OSError as e:
                print(f">>> [warn] failed to remove {fname}: {e}")

    if removed:
        print(f">>> removed {len(removed)} histogram pngs:")
        for f in removed:
            print("   ", f)
    else:
        print(">>> no matching histogram pngs to remove")


def print_audit_rankings_and_maybe_prune(
    df,
    cols,
    audit,
    topk=20,
    drop_invalid_top_n=0,
    drop_zero_top_n=0,
):
    """
    在 hist 之后输出：无效因子（valid_ratio=0）与含零因子（zero_ratio>0）排名；
    并可选自动删除各自 Top N（默认 0=不删）。
    """
    # 无效因子：全 NaN（valid_ratio==0）
    invalid_candidates = audit[audit["valid_ratio"] <= 0].sort_values("valid_ratio")
    print_tbl_invalid = (
        invalid_candidates.head(topk)
        if not invalid_candidates.empty
        else audit.sort_values("valid_ratio").head(topk)
    )
    print("\n>>> invalid factors (valid_ratio lowest; drop-candidates are valid_ratio==0):")
    print(print_tbl_invalid[["valid_ratio", "zero_ratio", "unique_count", "mean", "std"]].to_string())

    # 含零因子：zero_ratio>0 且有有效值
    zero_candidates = audit[(audit["valid_ratio"] > 0) & (audit["zero_ratio"] > 0)].sort_values("zero_ratio", ascending=False)
    print("\n>>> zero-heavy factors (zero_ratio highest):")
    if zero_candidates.empty:
        print("(none)")
    else:
        print(zero_candidates.head(topk)[["valid_ratio", "zero_ratio", "unique_count", "mean", "std"]].to_string())

    drop = set()
    if drop_invalid_top_n > 0 and not invalid_candidates.empty:
        drop.update(invalid_candidates.index[:drop_invalid_top_n].tolist())
    if drop_zero_top_n > 0 and not zero_candidates.empty:
        drop.update(zero_candidates.index[:drop_zero_top_n].tolist())

    if drop:
        drop = [c for c in drop if c in df.columns]
        print(f"\n>>> auto-dropping {len(drop)} factors after hist: {drop}")
        df = df.drop(columns=drop, errors="ignore")
        cols = [c for c in cols if c not in drop]

    return df, cols


def interactive_hist_prune(df, cols, topk=20):
    """
    直方图绘制结束后，基于缺失/零值情况做一次“人工 + 半自动”丢弃因子。
    支持：
      - 直接输入因子名：roe, pb
      - 输入列序号：0, 12
      - 自动丢弃：null=10 / zero=10
      - 按占比阈值丢弃：nan>=0.8 / valid<=0.2 / zeroall>=0.5 / zerovalid>=0.8
    """
    df = df.copy()
    cols = list(cols)

    help_msg = (
        "\nEnter factor ids/names to drop, or commands: "
        "null=5, zero=5, nan>=0.8, valid<=0.2, zeroall>=0.5, zerovalid>=0.8; "
        "q to quit\n"
        "Examples: null=10, zeroall>=0.6, 3, roe\n"
    )

    while True:
        audit = factor_audit_table(df, cols)

        # 兼容旧版本：如果没补齐列（用户自己改了），这里兜底构造
        if "nan_ratio" not in audit.columns:
            audit["nan_ratio"] = 1 - audit["valid_ratio"]
        if "zero_all_ratio" not in audit.columns:
            # zero_all_ratio 需要总样本数，这里只能用 valid_ratio + nan_ratio 反推不了，只给 NaN
            audit["zero_all_ratio"] = np.nan

        print("\n>>> [hist prune] lowest valid_ratio (highest missing):")
        show_cols = [c for c in ["valid_ratio", "nan_ratio", "zero_all_ratio", "zero_ratio", "unique_count", "mean", "std"] if c in audit.columns]
        print(audit.sort_values("valid_ratio").head(topk)[show_cols].to_string())

        # 零值占比排行（优先 zero_all_ratio，其次 zero_ratio）
        if audit["zero_all_ratio"].notna().any():
            ztbl = audit.sort_values("zero_all_ratio", ascending=False)
            print("\n>>> [hist prune] highest zero_all_ratio:")
            print(ztbl.head(topk)[show_cols].to_string())
        else:
            ztbl = audit.sort_values("zero_ratio", ascending=False)
            print("\n>>> [hist prune] highest zero_ratio (among valid):")
            print(ztbl.head(topk)[show_cols].to_string())

        s = input(help_msg).strip()
        if s.lower() in {"q", "quit", "exit"}:
            break

        tokens = [t.strip() for t in s.split(",") if t.strip()]
        drop = set()

        # 记录自动 drop 请求
        auto_null_n = 0
        auto_zero_n = 0
        th_nan = None
        th_valid = None
        th_zeroall = None
        th_zerovalid = None

        for t in tokens:
            # auto null/zero top-n
            m = re.fullmatch(r"(?:null|nan|invalid)\s*=?\s*(\d+)", t, flags=re.IGNORECASE)
            if m:
                auto_null_n = max(auto_null_n, int(m.group(1)))
                continue

            m = re.fullmatch(r"zero\s*=?\s*(\d+)", t, flags=re.IGNORECASE)
            if m:
                auto_zero_n = max(auto_zero_n, int(m.group(1)))
                continue

            # thresholds
            m = re.fullmatch(r"(?:nan|null)\s*>=\s*([0-9]*\.?[0-9]+)", t, flags=re.IGNORECASE)
            if m:
                th_nan = float(m.group(1))
                continue

            m = re.fullmatch(r"valid\s*<=\s*([0-9]*\.?[0-9]+)", t, flags=re.IGNORECASE)
            if m:
                th_valid = float(m.group(1))
                continue

            m = re.fullmatch(r"zeroall\s*>=\s*([0-9]*\.?[0-9]+)", t, flags=re.IGNORECASE)
            if m:
                th_zeroall = float(m.group(1))
                continue

            m = re.fullmatch(r"zerovalid\s*>=\s*([0-9]*\.?[0-9]+)", t, flags=re.IGNORECASE)
            if m:
                th_zerovalid = float(m.group(1))
                continue

            # direct factor id/name
            if t.isdigit():
                idx = int(t)
                if 0 <= idx < len(cols):
                    drop.add(cols[idx])
                else:
                    print(f"!!! invalid id: {idx}")
            elif t in cols:
                drop.add(t)
            else:
                print(f"!!! unknown factor: {t}")

        # apply auto null
        if auto_null_n > 0:
            # 优先砍全 NaN 的，其次按 missing 高的
            invalid = audit[audit["valid_ratio"] <= 0].index.tolist()
            if invalid:
                take = invalid[:auto_null_n]
            else:
                take = audit.sort_values("valid_ratio").head(auto_null_n).index.tolist()
            drop.update(take)
            print(f">>> auto-dropping null-heavy top {auto_null_n}: {take}")

        # apply auto zero
        if auto_zero_n > 0:
            if audit["zero_all_ratio"].notna().any():
                take = audit.sort_values("zero_all_ratio", ascending=False).head(auto_zero_n).index.tolist()
            else:
                take = audit.sort_values("zero_ratio", ascending=False).head(auto_zero_n).index.tolist()
            drop.update(take)
            print(f">>> auto-dropping zero-heavy top {auto_zero_n}: {take}")

        # apply thresholds
        if th_nan is not None:
            take = audit[audit["nan_ratio"] >= th_nan].index.tolist()
            drop.update(take)
            print(f">>> threshold drop nan_ratio >= {th_nan}: n={len(take)}")

        if th_valid is not None:
            take = audit[audit["valid_ratio"] <= th_valid].index.tolist()
            drop.update(take)
            print(f">>> threshold drop valid_ratio <= {th_valid}: n={len(take)}")

        if th_zeroall is not None and audit["zero_all_ratio"].notna().any():
            take = audit[audit["zero_all_ratio"] >= th_zeroall].index.tolist()
            drop.update(take)
            print(f">>> threshold drop zero_all_ratio >= {th_zeroall}: n={len(take)}")

        if th_zerovalid is not None:
            take = audit[audit["zero_ratio"].notna() & (audit["zero_ratio"] >= th_zerovalid)].index.tolist()
            drop.update(take)
            print(f">>> threshold drop zero_ratio (among valid) >= {th_zerovalid}: n={len(take)}")

        drop = [c for c in drop if c in df.columns]

        if not drop:
            print(">>> nothing dropped")
            continue

        print(">>> dropping:")
        for f in drop:
            print("   ", f)

        df = df.drop(columns=drop, errors="ignore")
        cols = [c for c in cols if c not in set(drop)]

    return df, cols


def plot_factor_histograms(
    df,
    cols=None,
    bins=50,
    max_cols=4,
    max_rows=4,          # 每页最多多少行
    figsize_per_subplot=(4, 3),
    clip_q=0.01,
    out_dir="factor",
    date_tag: str | None = None,
):
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns

    os.makedirs(out_dir, exist_ok=True)

    ncols = max_cols
    nrows = max_rows
    page_size = ncols * nrows
    N = len(df)

    total_pages = math.ceil(len(cols) / page_size)

    for page in range(total_pages):
        start = page * page_size
        end = min((page + 1) * page_size, len(cols))
        cols_page = cols[start:end]

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_per_subplot[0] * ncols,
                     figsize_per_subplot[1] * nrows),
            gridspec_kw={"hspace": 1.8, "wspace": 0.3}
        )

        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(cols_page):
            ax = axes[i]
            s = df[col].dropna()

            if len(s) == 0:
                ax.set_title(f"{col}\n(no valid data)")
                ax.axis("off")
                continue

            s_plot = clip_series(s, q=clip_q)
            zero_ratio = (s == 0).mean()

            # log 判断
            if s_plot.min() > 0 and s_plot.max() / s_plot.min() > 100:
                s_plot = np.log1p(s_plot)
                xlabel = "log1p"
            else:
                xlabel = ""

            ax.hist(s_plot, bins=bins, alpha=0.75)
            ax.set_xlabel(xlabel)

            ax.set_title(
                f"{col}\n"
                f"valid={len(s)}, nan={N-len(s)}, zero%={zero_ratio:.2f}",
                fontsize=9
            )

        # 多余子图关掉
        for j in range(len(cols_page), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"Factor Histograms  (page {page+1}/{total_pages})",
            fontsize=14
        )

        fname = os.path.join(
            out_dir,
            f"factor_histograms_{resolve_date_tag(date_tag)}_p{page+1}.png"
        )
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f">>> saved {fname}")

def corr_degree_table(df, cols, threshold=0.6, method="pearson"):
    """
    统计每个因子与其它因子 |corr|>=threshold 的次数（degree），并输出排序表
    """
    x = df[cols].copy()

    # corr 需要数值型并且 NaN 会传播，这里让 pandas 自己处理 pairwise
    corr_abs = x.corr(method=method).abs()

    # 自相关=1，别算进去
    degree = (corr_abs >= threshold).sum(axis=1) - 1

    # 还可以给个“最强相关对象”和“最大相关系数”方便你直接下刀
    corr_abs_no_diag = corr_abs.copy()
    np.fill_diagonal(corr_abs_no_diag.values, np.nan)
    max_corr = corr_abs_no_diag.max(axis=1)
    max_with = corr_abs_no_diag.idxmax(axis=1)

    out = pd.DataFrame({
        "degree": degree.astype(int),
        "max_abs_corr": max_corr,
        "max_with": max_with,
    }).sort_values(["degree", "max_abs_corr"], ascending=False)

    return out, corr_abs


def compute_corr_and_pairs(df, cols, threshold, method="pearson"):
    corr = df[cols].corr(method=method)
    factors = corr.columns.tolist()

    pairs = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            c = corr.iloc[i, j]
            if pd.notna(c) and abs(c) >= threshold:
                pairs.append((i, j, factors[i], factors[j], float(c)))

    pairs.sort(key=lambda x: abs(x[4]), reverse=True)
    return corr, factors, pairs


def _connected_components_from_pairs(factors, pairs):
    # Union-Find
    parent = {f: f for f in factors}
    rank = {f: 0 for f in factors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for _, _, f1, f2, _ in pairs:
        union(f1, f2)

    comps = {}
    for f in factors:
        r = find(f)
        comps.setdefault(r, set()).add(f)

    # 只返回 size>=2 的连通分量（size=1 没啥可剪）
    components = [c for c in comps.values() if len(c) >= 2]
    components.sort(key=lambda s: len(s), reverse=True)
    return components


def _factor_quality_score(audit_row):
    """
    越大越好：
      - valid_ratio 越大越好（nan_ratio 越小越好，本质同一个信息）
      - zero_all_ratio 越小越好（全样本为0越多越可疑）
      - unique_count 越大越好（常数/离散很少的因子不行）
    """
    if audit_row is None:
        return -1e9

    valid = float(audit_row.get("valid_ratio", np.nan))
    zero_all = float(audit_row.get("zero_all_ratio", np.nan))
    uniq = float(audit_row.get("unique_count", np.nan))

    # 兜底：缺列就当很差
    if np.isnan(valid):
        valid = 0.0
    if np.isnan(zero_all):
        zero_all = 1.0
    if np.isnan(uniq):
        uniq = 0.0

    # 一个简单但实用的打分（你以后想加 IC/IR 也能接进来）
    # valid 最重要，其次惩罚全0，其次鼓励 unique
    return 3.0 * valid - 1.5 * zero_all + 0.001 * uniq


def _pick_best_in_component(component, audit):
    """
    在一个连通分量里选“最好保留”的因子
    """
    best = None
    best_score = -1e18
    for f in component:
        row = audit.loc[f] if (audit is not None and f in audit.index) else None
        sc = _factor_quality_score(row)
        if sc > best_score:
            best_score = sc
            best = f
    return best, best_score


def plot_corr(corr, pairs, threshold, save_path=os.path.join("factor", "correlation.png")):
    factors = corr.columns.tolist()
    n = len(factors)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(factors, rotation=90, fontsize=8)
    ax.set_yticklabels(factors, fontsize=8)

    for i, j, *_ in pairs:
        ax.add_patch(
            plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                          fill=False, edgecolor="black", linewidth=1.2)
        )

    ax.set_title(f"|corr| ≥ {threshold}")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interactive_prune(
    df,
    cols,
    threshold=0.6,
    method="pearson",
    save_path=os.path.join("factor", "correlation.png"),
):
    """
    新启发式：
      - 不再按 degree（出现次数）排序主导删除
      - 改为：构建相关图 -> 连通分量 -> 按“最大连通分量”逐个处理
      - 对于要处理的连通分量：只保留一个“最好的因子”，其余全删
        “最好”依据 hist 的 audit：valid_ratio 高、zero_all_ratio 低、unique_count 高

    交互命令：
      - cc=3      删除“前3个最大连通分量”中除最佳外的所有因子
      - cc=all    删除所有连通分量中除最佳外的所有因子
      - 也支持手动输入因子名/序号： 3, roe_dt
      - q         退出
    """
    df = df.copy()
    cols = list(cols)

    help_msg = (
        "\nEnter factor ids/names to drop, or commands:\n"
        "  cc=3       prune top-3 largest connected components (keep 1 best each)\n"
        "  cc=all     prune all connected components (keep 1 best each)\n"
        "  q          quit\n"
        "Examples: cc=2   |   3, netprofit_margin_std_q\n"
    )

    while True:
        corr, factors, pairs = compute_corr_and_pairs(df, cols, threshold, method=method)

        if not pairs:
            print(">>> no high correlation left")
            # 保存最终图（覆盖写）
            plot_corr(corr, [], threshold, save_path=save_path)
            break

        # 这里用“当前 df”的 audit 作为质量评分依据
        audit = factor_audit_table(df, cols)

        components = _connected_components_from_pairs(factors, pairs)
        if not components:
            print(">>> no connected components (unexpected), stop")
            plot_corr(corr, pairs, threshold, save_path=save_path)
            break

        print(f"\n>>> connected components (|corr|>={threshold}, method={method}) sorted by size:")
        show_k = min(10, len(components))
        for idx in range(show_k):
            comp = components[idx]
            best, best_sc = _pick_best_in_component(comp, audit)
            print(f"[cc#{idx}] size={len(comp)} keep={best} score={best_sc:.4f}")
            # 顺带打印一下这个分量里前几名候选（按质量）
            ranked = sorted(
                [(f, _factor_quality_score(audit.loc[f])) for f in comp],
                key=lambda x: x[1],
                reverse=True
            )
            top_show = ", ".join([f"{f}({sc:.3f})" for f, sc in ranked[:8]])
            print(f"      top candidates: {top_show}")

        # 画图并覆盖保存
        plot_corr(corr, pairs, threshold, save_path=save_path)
        print(f">>> saved correlation plot (overwrite): {save_path}")

        s = input(help_msg).strip()
        if s.lower() in {"q", "quit", "exit"}:
            break

        drop = set()
        tokens = [t.strip() for t in s.split(",") if t.strip()]

        # 是否触发 cc 剪枝
        cc_n = None
        cc_all = False

        for t in tokens:
            m = re.fullmatch(r"cc\s*=\s*(all|\d+)", t, flags=re.IGNORECASE)
            if m:
                v = m.group(1).lower()
                if v == "all":
                    cc_all = True
                else:
                    cc_n = int(v)
                continue

            # 手动因子序号/名称
            if t.isdigit():
                idx = int(t)
                if 0 <= idx < len(factors):
                    drop.add(factors[idx])
                else:
                    print(f"!!! invalid id: {idx}")
            elif t in factors:
                drop.add(t)
            else:
                print(f"!!! unknown token/factor: {t}")

        # 连通图剪枝：删“最大连通分量”的其它因子，只留 best
        if cc_all or (cc_n is not None and cc_n > 0):
            k = len(components) if cc_all else min(cc_n, len(components))
            for comp in components[:k]:
                best, _ = _pick_best_in_component(comp, audit)
                for f in comp:
                    if f != best:
                        drop.add(f)
            print(f">>> cc-prune triggered: processed {k} components; drop n={len(drop)}")

        drop = [c for c in drop if c in df.columns]
        if not drop:
            print(">>> nothing dropped")
            continue

        print(">>> dropping:")
        for f in drop:
            print("   ", f)

        df = df.drop(columns=drop, errors="ignore")
        cols = [c for c in cols if c not in set(drop)]

    with open("columns.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, indent=2, ensure_ascii=False)

    return df, cols

def run_pca_after_prune(
    df,
    cols,
    out_dir="factor",
    n_components=20,
    fillna="median",   # "median" / "mean" / "zero"
    clip_q=None,       # 例如 0.01；None=不clip
    standardize=True,  # True=StandardScaler
    date_tag: str | None = None,
):
    """
    在相关性筛选后的截面数据上做 PCA，并输出：
      - explained_variance_ratio
      - cumulative variance
      - loadings（因子在PC上的权重）
      - scree plot
    """
    os.makedirs(out_dir, exist_ok=True)

    x = df[cols].copy()

    # 可选：对每个因子做横截面去极值
    if clip_q is not None:
        for c in cols:
            s = x[c]
            if pd.api.types.is_numeric_dtype(s):
                lo, hi = s.quantile(clip_q), s.quantile(1 - clip_q)
                x[c] = s.clip(lo, hi)

    # 填 NaN（PCA/Scaler 都不吃 NaN）
    if fillna == "median":
        x = x.fillna(x.median(numeric_only=True))
    elif fillna == "mean":
        x = x.fillna(x.mean(numeric_only=True))
    elif fillna == "zero":
        x = x.fillna(0.0)
    else:
        raise ValueError(f"unknown fillna={fillna}")

    # 标准化
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(x.values)
    else:
        X = x.values

    k = min(n_components, X.shape[1])
    pca = PCA(n_components=k, random_state=0)
    Z = pca.fit_transform(X)

    # 解释方差
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    summary = pd.DataFrame({
        "pc": [f"PC{i+1}" for i in range(len(evr))],
        "explained_var_ratio": evr,
        "cum_explained_var_ratio": cum,
    })

    summary_path = os.path.join(out_dir, f"pca_summary_{resolve_date_tag(date_tag)}.csv")
    summary.to_csv(summary_path, index=False)
    print(f">>> saved PCA summary: {summary_path}")

    # loadings：rows=因子，cols=PC
    loadings = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    loadings_path = os.path.join(out_dir, f"pca_loadings_{resolve_date_tag(date_tag)}.csv")
    loadings.to_csv(loadings_path)
    print(f">>> saved PCA loadings: {loadings_path}")

    # 打印前几个PC的解释率
    print("\n>>> PCA explained variance (top components):")
    print(summary.head(min(10, len(summary))).to_string(index=False))

    # 每个PC打印绝对值最大的若干因子”
    top_m = 10
    print("\n>>> PCA top loadings per PC (abs top 10 factors):")
    for j in range(loadings.shape[1]):
        pc = loadings.columns[j]
        top = loadings[pc].abs().sort_values(ascending=False).head(top_m).index.tolist()
        # 同时给出符号
        signed = [f"{f}({loadings.loc[f, pc]:+.3f})" for f in top]
        print(f"{pc}: " + ", ".join(signed))

    # scree plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(evr) + 1), evr, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, f"pca_scree_{resolve_date_tag(date_tag)}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">>> saved PCA scree plot: {fig_path}")

    # 可选：把PC得分也存一下（用于后续聚类/可视化）
    scores = pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(Z.shape[1])], index=df.index)
    scores_path = os.path.join(out_dir, f"pca_scores_{resolve_date_tag(date_tag)}.csv")
    scores.to_csv(scores_path)
    print(f">>> saved PCA scores: {scores_path}")

    return pca, summary, loadings, scores



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="Run date tag YYYYMMDD; default env RUN_DATE/DATE_TAG or today")

    # ===== filter stage (run BEFORE preanalysis by default) =====
    p.add_argument("--filter-first", action="store_true", default=True,
                   help="Run filter.py logic first to produce purified.h5 before preanalysis (default: True)")
    p.add_argument("--no-filter-first", dest="filter_first", action="store_false",
                   help="Disable the filter stage and run preanalysis directly on --input")

    p.add_argument("--filter-input", default=None,
                   help="Filter input HDF5 path. Default: A/stock_factors_{date}.h5 (date from --date/env/today)")
    p.add_argument("--filter-output", default="factor/purified.h5",
                   help="Filter output HDF5 path (default: factor/purified.h5)")
    p.add_argument("--filter-mode", choices=["loose", "strict", "none"], default="loose",
                   help="Filter mode: loose/strict/none (default: loose)")

    # ===== preanalysis stage =====
    p.add_argument("--input", default="factor/purified.h5", help="Input HDF5 path (used if --no-filter-first)")
    p.add_argument("--key", default="financials", help="HDF5 key for factors")
    p.add_argument("--hist-dir", default="factor", help="Histogram output dir")
    p.add_argument("--factor-dict", default=None, help="Optional factor dict JSON path (None=all)")
    p.add_argument("--corr-method", default="pearson", help="Correlation method: pearson/spearman/kendall")
    p.add_argument("--corr-threshold", type=float, default=0.6, help="Correlation threshold")
    return p.parse_args()



def run_filter_stage(date_tag: str, args):
    """Run filter step (same-directory filter.py) to build purified factor file."""
    # Ensure we can import sibling module when executed from elsewhere
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)

    try:
        import filter as filter_mod  # filter.py in the same directory
    except Exception as e:
        raise RuntimeError(f"Failed to import filter.py from {this_dir}: {e}") from e

    input_path = args.filter_input
    if input_path is None:
        input_path = f"A/stock_factors_{date_tag}.h5"

    output_path = args.filter_output

    # Build stock_filter object compatible with filter.py's MultiFactorPurifier
    mode = (args.filter_mode or "loose").lower()

    if mode == "loose":
        stock_filter = filter_mod.LooseStockFilter()
    elif mode == "strict":
        stock_filter = filter_mod.StrictStockFilter()
    elif mode == "none":
        # No-op filter: keep everything
        class _NoOpFilter(filter_mod.BaseStockFilter):
            def filter(self, df: pd.DataFrame) -> pd.DataFrame:
                print(">>> applying NONE stock filter (no-op)")
                if "ts_code" in df.columns:
                    print(f">>> after NONE filter: {df['ts_code'].nunique()} stocks")
                return df
        stock_filter = _NoOpFilter()
    else:
        raise ValueError(f"Unknown filter_mode={args.filter_mode}")

    purifier = filter_mod.MultiFactorPurifier(
        input_path=input_path,
        output_path=output_path,
        stock_filter=stock_filter,
    )
    print(f">>> [stage:filter] input={input_path} -> output={output_path} (mode={mode})")
    purifier.run()
    return output_path

if __name__ == "__main__":
    args = parse_args()
    date_tag = resolve_date_tag(args.date)

    # If enabled, run filter stage first (so you can just run preanalysis once).
    if getattr(args, "filter_first", True):
        INPUT_PATH = run_filter_stage(date_tag, args)
    else:
        INPUT_PATH = args.input
    INPUT_KEY = args.key
    HIST_DIR = args.hist_dir
    # 可选：仅加载指定因子（dict 或 JSON 路径）；None=加载全部
    FACTOR_DICT = args.factor_dict

    df = pd.read_hdf(INPUT_PATH, key=INPUT_KEY)
    df = apply_factor_dict(df, FACTOR_DICT)
    print(f"shape:{df.shape}")
    print(f"columns: \n{df.columns}")
    print(f"dtypes: \n{df.dtypes}")
    numeric_cols = [
    c for c in df.columns
    if c not in ID_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]
    audit = factor_audit_table(df, numeric_cols)
    print(audit)

    plot_factor_histograms(
        df,
        cols=numeric_cols,
        bins=60,
        date_tag=date_tag,
    )
    # hist 后输出无效/含零因子排名，并可选自动删除各自 Top N（默认 0=不删）
    AUTO_DROP_INVALID_TOP_N = 0
    AUTO_DROP_ZERO_TOP_N = 0
    df, numeric_cols = print_audit_rankings_and_maybe_prune(
        df,
        numeric_cols,
        audit,
        topk=20,
        drop_invalid_top_n=AUTO_DROP_INVALID_TOP_N,
        drop_zero_top_n=AUTO_DROP_ZERO_TOP_N,
    )

    # hist 后交互式丢弃（手动输入因子名/序号，或按 null/zero top_n / 占比阈值自动丢弃）
    df, numeric_cols = interactive_hist_prune(df, numeric_cols, topk=20)

    #绘制丢弃后的各个因子直方图
    delete_histogram_pngs(HIST_DIR, date_str=date_tag)
    plot_factor_histograms(
        df,
        cols=numeric_cols,
        bins=60,
        date_tag=date_tag,
    )

    # 相关性计算方法：'pearson'(默认) / 'spearman' / 'kendall' / 自定义函数
    CORR_METHOD = args.corr_method

    # 这里 corr 这行其实可以不要了（interactive_prune 里会算并保存图）
    # corr = df[numeric_cols].corr(method=CORR_METHOD)

    df, numeric_cols = interactive_prune(df, numeric_cols, threshold=args.corr_threshold, method=CORR_METHOD)

    # 保存最终剩余因子列表（不影响 interactive_prune 里原本的 columns.json）
    # ===== PCA (after correlation pruning) =====
    pca, pca_summary, pca_loadings, pca_scores = run_pca_after_prune(
        df,
        numeric_cols,
        out_dir="factor",
        n_components=min(20, len(numeric_cols)),
        fillna="median",
        clip_q=0.01,       # 你想更“干净”就开；不想动分布就设 None
        standardize=True,
        date_tag=date_tag,
    )
    os.makedirs("factor", exist_ok=True)
    out_json = os.path.join("factor", f"remaining_factors_{resolve_date_tag(date_tag)}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(numeric_cols, f, indent=2, ensure_ascii=False)
    print(f">>> saved remaining factors: {out_json} (n={len(numeric_cols)})")

