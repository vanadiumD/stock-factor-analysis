#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ae_extra_analysis_simple.py (extended)

Reuse outputs from ae_cluster.py:
  - ae_latent.csv
  - clusters_AE_*.csv

Add actionable picks:
  1) cluster representatives (closest to centroid)
  2) latent extremes: per latent dimension Top-N / Bottom-N

Outputs under:
  <out_dir>/extra_pick_simple/
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def resolve_run_date(date_str: str) -> str:
    s = str(date_str).strip()
    if s == "":
        return datetime.datetime.now().strftime("%Y%m%d")
    return s


def normalize_stock_code(s) -> str:
    return str(s).strip()


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def auto_find_clusters_csv(out_dir: str) -> str:
    cands = [f for f in os.listdir(out_dir) if f.startswith("clusters_AE_") and f.endswith(".csv")]
    if not cands:
        raise FileNotFoundError(f"No clusters_AE_*.csv found in {out_dir}")
    cands_full = [os.path.join(out_dir, f) for f in cands]
    cands_full.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands_full[0]


def pick_cluster_representatives(latent_df: pd.DataFrame, labels: np.ndarray, n_per_cluster: int) -> pd.DataFrame:
    """
    Per cluster: pick n points closest to centroid in latent space.
    score_d2 = squared distance to centroid (smaller => more representative).
    """
    Z = latent_df.values
    rows = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        Zc = Z[idx]
        center = Zc.mean(axis=0, keepdims=True)
        d2 = ((Zc - center) ** 2).sum(axis=1)
        order = np.argsort(d2)[: min(n_per_cluster, idx.size)]
        for j in order:
            code = latent_df.index[idx[j]]
            rows.append({
                "stock_code": code,
                "cluster": int(c),
                "method": "cluster_rep",
                "tag": f"cluster_{int(c)}_rep",
                "score": float(d2[j]),  # smaller is better for this method
                "dim": "",
                "value": np.nan,
                "rank_in_group": int(j) + 1,
            })
    return pd.DataFrame(rows).sort_values(["cluster", "score"]).reset_index(drop=True)


def pick_latent_extremes(latent_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    For each latent dim:
      - Top-N (largest)
      - Bottom-N (smallest)
    score/value = the latent value on that dim.
    """
    rows = []
    for dim in latent_df.columns:
        s = latent_df[dim]
        top = s.nlargest(n)
        bot = s.nsmallest(n)

        # top
        for r, (code, val) in enumerate(top.items(), start=1):
            rows.append({
                "stock_code": code,
                "cluster": np.nan,
                "method": "latent_extreme",
                "tag": f"{dim}_top",
                "score": float(val),     # larger is more extreme for top
                "dim": dim,
                "value": float(val),
                "rank_in_group": r,
            })

        # bottom
        for r, (code, val) in enumerate(bot.items(), start=1):
            rows.append({
                "stock_code": code,
                "cluster": np.nan,
                "method": "latent_extreme",
                "tag": f"{dim}_bottom",
                "score": float(val),     # smaller is more extreme for bottom
                "dim": dim,
                "value": float(val),
                "rank_in_group": r,
            })

    out = pd.DataFrame(rows)
    # readability: sort by dim then top/bottom then rank
    out["tag_type"] = out["tag"].apply(lambda x: "top" if x.endswith("_top") else "bottom")
    out = out.sort_values(["dim", "tag_type", "rank_in_group"]).drop(columns=["tag_type"]).reset_index(drop=True)
    return out


def merge_picks(picks_list):
    df = pd.concat([d for d in picks_list if d is not None and len(d) > 0], ignore_index=True)
    if df.empty:
        return df

    # per-stock tags summary
    tag_sum = df.groupby("stock_code")["tag"].apply(lambda x: "|".join(sorted(set(x)))).rename("tags")
    out = df.merge(tag_sum, on="stock_code", how="left")

    # nice ordering: method then cluster then stock
    method_order = {"cluster_rep": 0, "latent_extreme": 1}
    out["_m"] = out["method"].map(lambda x: method_order.get(x, 99))
    out = out.sort_values(["_m", "cluster", "stock_code", "tag"]).drop(columns=["_m"]).reset_index(drop=True)
    return out


def plot_latent2_highlight(latent_df, labels, picks_all_df, out_path, title, annotate=True):
    """
    Plot z1 vs z2 (first two latent dims).
    Highlight any stock that appears in picks_all_df.
    """
    if latent_df.shape[1] < 2:
        raise ValueError("latent_df must have at least 2 dims to plot (need at least 2 columns).")

    x = latent_df.iloc[:, 0].values
    y = latent_df.iloc[:, 1].values

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, c=labels, s=8, alpha=0.55)

    pick_codes = sorted(set(picks_all_df["stock_code"].astype(str).tolist()))
    pick_codes = [c for c in pick_codes if c in latent_df.index]
    pick_idx = [latent_df.index.get_loc(c) for c in pick_codes]

    if pick_idx:
        plt.scatter(x[pick_idx], y[pick_idx], s=70, marker="x")
        if annotate:
            for code in pick_codes:
                i = latent_df.index.get_loc(code)
                plt.text(x[i], y[i], code, fontsize=7)

    plt.title(title)
    plt.xlabel(latent_df.columns[0])
    plt.ylabel(latent_df.columns[1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="run date YYYYMMDD (default: today)")
    ap.add_argument("--out_dir", default="", help="ae_cluster output dir (override date-based path)")

    ap.add_argument("--rep_n", type=int, default=10, help="representatives per cluster (default=10)")
    ap.add_argument("--extreme_n", type=int, default=10, help="top/bottom N per latent dim (default=10)")

    ap.add_argument("--no_annotate", action="store_true", help="do not draw stock_code text labels")

    ap.add_argument("--latent", default="", help="latent csv path (default: <out_dir>/ae_latent.csv)")
    ap.add_argument("--clusters", default="", help="clusters csv path (default: auto find clusters_AE_*.csv in out_dir)")
    return ap.parse_args()


def main():
    args = parse_args()
    run_date = resolve_run_date(args.date)

    out_dir = args.out_dir if args.out_dir else f"factor/ae_out_{run_date}"
    os.makedirs(out_dir, exist_ok=True)

    latent_path = args.latent.strip() or os.path.join(out_dir, "ae_latent.csv")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(f"latent csv not found: {latent_path}")

    clusters_path = args.clusters.strip()
    if not clusters_path:
        clusters_path = auto_find_clusters_csv(out_dir)
    if not os.path.exists(clusters_path):
        raise FileNotFoundError(f"clusters csv not found: {clusters_path}")

    pick_dir = ensure_dir(os.path.join(out_dir, "extra_pick_simple"))
    print(f">>> out_dir      = {out_dir}")
    print(f">>> latent_path  = {latent_path}")
    print(f">>> clusters_path= {clusters_path}")
    print(f">>> pick_dir     = {pick_dir}")

    latent_df = pd.read_csv(latent_path, index_col=0)
    latent_df.index = latent_df.index.map(normalize_stock_code)

    cdf = pd.read_csv(clusters_path, index_col=0)
    cdf.index = cdf.index.map(normalize_stock_code)
    cluster_col = "cluster" if "cluster" in cdf.columns else cdf.columns[0]

    common = latent_df.index.intersection(cdf.index)
    if len(common) == 0:
        raise RuntimeError("No overlap between latent and clusters indices.")

    latent_df = latent_df.loc[common]
    labels = cdf.loc[common, cluster_col].to_numpy()

    # 1) cluster reps
    picks_rep = pick_cluster_representatives(latent_df, labels, n_per_cluster=args.rep_n)
    path_rep = os.path.join(pick_dir, f"picks_cluster_representatives_n{args.rep_n}.csv")
    picks_rep.to_csv(path_rep, index=False)
    print(f">>> saved: {path_rep} | rows={len(picks_rep)}")

    # 2) latent extremes
    picks_ext = pick_latent_extremes(latent_df, n=args.extreme_n)
    path_ext = os.path.join(pick_dir, f"picks_latent_extremes_n{args.extreme_n}.csv")
    picks_ext.to_csv(path_ext, index=False)
    print(f">>> saved: {path_ext} | rows={len(picks_ext)}")

    # merge
    picks_all = merge_picks([picks_rep, picks_ext])
    path_all = os.path.join(pick_dir, "picks_all.csv")
    picks_all.to_csv(path_all, index=False)
    print(f">>> saved: {path_all} | rows={len(picks_all)} | unique_stocks={picks_all['stock_code'].nunique()}")

    # plot
    plot_path = os.path.join(pick_dir, f"plot_latent2_highlight_all.png")
    plot_latent2_highlight(
        latent_df=latent_df,
        labels=labels,
        picks_all_df=picks_all,
        out_path=plot_path,
        title=f"AE latent2 (z1,z2) + picks (rep_n={args.rep_n}, extreme_n={args.extreme_n})",
        annotate=(not args.no_annotate),
    )
    print(f">>> saved: {plot_path}")

    summary = {
        "out_dir": out_dir,
        "latent": latent_path,
        "clusters": clusters_path,
        "rep_n": int(args.rep_n),
        "extreme_n": int(args.extreme_n),
        "n_stocks": int(len(latent_df)),
        "z_dim": int(latent_df.shape[1]),
        "n_clusters": int(len(np.unique(labels))),
        "outputs": {
            "cluster_reps": os.path.basename(path_rep),
            "latent_extremes": os.path.basename(path_ext),
            "merged": os.path.basename(path_all),
            "plot": os.path.basename(plot_path),
        }
    }
    with open(os.path.join(pick_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(">>> DONE.")


if __name__ == "__main__":
    main()
