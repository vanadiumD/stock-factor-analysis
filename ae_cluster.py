
import os
import json
import random
import warnings
import argparse
import glob
import re
import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------
# Config (defaults; can override via CLI)
# ----------------------------
H5_PATH = "factor/purified.h5"
REMAIN_JSON = None  # default resolved in main by --date

OUT_DIR = None  # default resolved in main by --date

# AE settings
LATENT_DIM = 8          # 建议 4~10；根据 PCA 显示的有效维度选取
HIDDEN1 = 16
HIDDEN2 = 8
EPOCHS = 2000
BATCH_SIZE = 512
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42

# clustering
CLUSTER_METHOD = "kmeans"   # "kmeans" or "gmm"
N_CLUSTERS = 8              # 先 6~12 扫一下更好

# visualization
VIS_METHOD = "umap"         # "umap" / "tsne" / "latent2"

# PCA artifacts location (produced by preanalysis / run_pca_after_prune)
PCA_DIR = "factor"
# ----------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def safe_read_hdf_df(path: str, key: str) -> pd.DataFrame:
    """Try reading a DataFrame from HDF5 key with a few common variants."""
    candidates = [key, f"/{key}", f"{key}/", f"/{key}/"]
    last_err = None
    for k in candidates:
        try:
            return pd.read_hdf(path, k)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read HDF5 key={key}. Last error: {last_err}")


def safe_read_hdf_obj(path: str, key: str):
    """Read possibly non-DF object from HDFStore."""
    candidates = [key, f"/{key}", f"{key}/", f"/{key}/"]
    last_err = None
    for k in candidates:
        try:
            with pd.HDFStore(path, mode="r") as store:
                if k in store:
                    return store.get(k)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read HDF5 object key={key}. Last error: {last_err}")


def load_remaining_factors(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # 兼容：可能是 list，也可能是 {"remaining_factors":[...]} 这种
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for cand in ["remaining_factors", "factors", "cols", "columns"]:
            if cand in obj and isinstance(obj[cand], list):
                return obj[cand]
    raise ValueError(f"Unrecognized JSON format in {json_path}")


def normalize_stock_code(s: str) -> str:
    return str(s).strip()


def build_feature_matrix(h5_path: str, remaining_factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      X_df: index=stock_code, columns=remaining_factors
      daily_basic_df: daily_basic dataframe (optional use)
    """
    fin = safe_read_hdf_df(h5_path, "financials")

    try:
        db = safe_read_hdf_df(h5_path, "daily_basic")
    except Exception:
        db = pd.DataFrame()

    try:
        completed = safe_read_hdf_obj(h5_path, "completed")
    except Exception:
        completed = None

    completed_list = None
    if completed is not None:
        if isinstance(completed, (pd.Series, list, np.ndarray)):
            completed_list = [normalize_stock_code(x) for x in list(completed)]
        elif isinstance(completed, pd.DataFrame):
            if completed.shape[1] >= 1:
                completed_list = [normalize_stock_code(x) for x in completed.iloc[:, 0].tolist()]
        else:
            try:
                completed_list = [normalize_stock_code(x) for x in list(completed)]
            except Exception:
                completed_list = None

    if fin.index.name is None or fin.index.dtype == int:
        for cand in ["stock_code", "ts_code", "code"]:
            if cand in fin.columns:
                fin = fin.set_index(cand)
                break
    fin.index = fin.index.map(normalize_stock_code)

    if not db.empty:
        if db.index.name is None or db.index.dtype == int:
            for cand in ["stock_code", "ts_code", "code"]:
                if cand in db.columns:
                    db = db.set_index(cand)
                    break
        db.index = db.index.map(normalize_stock_code)

    # IMPORTANT: keep EXACT order from remaining_factors (json order)
    cols_exist = [c for c in remaining_factors if c in fin.columns]
    missing = [c for c in remaining_factors if c not in fin.columns]
    if missing:
        print(f"[warn] {len(missing)} factors not found in financials, e.g.: {missing[:10]}")

    X = fin[cols_exist].copy()

    if completed_list is not None and len(completed_list) > 0:
        keep = sorted(set(completed_list) & set(X.index))
        X = X.loc[keep]
        if not db.empty:
            db = db.loc[db.index.intersection(keep)]

    print(f">>> X shape (stocks x factors) = {X.shape}")
    return X, db


# ----------------------------
# AE model (PyTorch)
# ----------------------------
def build_ae_model(in_dim: int):
    import torch.nn as nn

    class AE(nn.Module):
        def __init__(self, in_dim: int, h1: int, h2: int, z_dim: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, h1),
                nn.ReLU(),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.Linear(h2, z_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, h2),
                nn.ReLU(),
                nn.Linear(h2, h1),
                nn.ReLU(),
                nn.Linear(h1, in_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

    return AE(in_dim=in_dim, h1=HIDDEN1, h2=HIDDEN2, z_dim=LATENT_DIM)


def train_or_load_ae_get_latent(X: np.ndarray, out_dir: str, model_path: str | None):
    """
    model_path:
      - None or "none": train and save to out_dir/ae_model.pt
      - otherwise: load model weights from given path, skip training
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> device: {device}")

    model = build_ae_model(in_dim=X.shape[1]).to(device)

    is_train = (model_path is None) or (str(model_path).lower() == "none")
    if not is_train:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AE model path not found: {model_path}")
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        print(f">>> loaded AE model weights: {model_path}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=min(BATCH_SIZE, len(ds)), shuffle=True, drop_last=False)

        best_loss = float("inf")
        best_state = None
        avg_loss = []

        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []
            for (xb,) in dl:
                xb = xb.to(device)
                x_hat, _ = model(xb)
                loss = loss_fn(x_hat, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())

            avg = float(np.mean(losses))
            if avg < best_loss:
                best_loss = avg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 20 == 0 or epoch == 1:
                print(f"epoch {epoch:4d} | loss {avg:.6f}")
            avg_loss.append(avg)

        model.load_state_dict(best_state)
        model.eval()

        save_path = os.path.join(out_dir, "ae_model.pt")
        torch.save(model.state_dict(), save_path)
        print(f">>> saved AE model: {save_path}")

        meta = {
            "latent_dim": LATENT_DIM,
            "hidden1": HIDDEN1,
            "hidden2": HIDDEN2,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        }
        with open(os.path.join(out_dir, "ae_model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        X_hat, Z = model(X_t)
        X_hat = X_hat.detach().cpu().numpy()
        Z = Z.detach().cpu().numpy()

    rec_err = np.mean((X - X_hat) ** 2, axis=1)
    return Z, rec_err


# ----------------------------
# Clustering / Embedding / Plot
# ----------------------------
def cluster_latent(Z: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        return km.fit_predict(Z)
    if method == "gmm":
        gmm = GaussianMixture(n_components=n_clusters, random_state=SEED, covariance_type="full")
        return gmm.fit_predict(Z)
    raise ValueError(f"unknown cluster method: {method}")


def embed_2d(Z: np.ndarray, method: str) -> np.ndarray:
    if method == "latent2":
        if Z.shape[1] < 2:
            raise ValueError("latent2 requires latent dim >= 2")
        return Z[:, :2]

    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=20, min_dist=0.1)
            return reducer.fit_transform(Z)
        except Exception as e:
            print(f"[warn] UMAP not available ({e}), fallback to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        ts = TSNE(n_components=2, random_state=SEED, perplexity=min(30, max(5, len(Z) // 50)))
        return ts.fit_transform(Z)

    raise ValueError(f"unknown embed method: {method}")


def plot_scatter(emb2d: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, s=8)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f">>> saved plot: {out_path}")


def plot_scatter_sizecolor(emb2d: np.ndarray, color_values: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(8, 6))
    plt.scatter(emb2d[:, 0], emb2d[:, 1], c=color_values, s=8)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f">>> saved plot: {out_path}")


def compute_cluster_metrics(Z: np.ndarray, labels: np.ndarray) -> dict:
    # guard: silhouette requires >=2 clusters and no empty
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    try:
        sil = float(silhouette_score(Z, labels))
    except Exception:
        sil = np.nan
    try:
        ch = float(calinski_harabasz_score(Z, labels))
    except Exception:
        ch = np.nan
    try:
        db = float(davies_bouldin_score(Z, labels))
    except Exception:
        db = np.nan
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}



# ----------------------------
# PCA artifacts reuse + order check (with fallback recompute)
# ----------------------------
def resolve_date_tag(date_str: str | None = None) -> str:
    if date_str:
        return str(date_str)
    env = os.getenv("RUN_DATE") or os.getenv("DATE_TAG")
    if env:
        return str(env)
    return resolve_date_tag(date_str)


def today_str(date_str: str | None = None):
    return resolve_date_tag(date_str)


def _latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=lambda p: os.path.getmtime(p))


def load_latest_pca_artifacts(pca_dir: str, date_tag: str | None = None):
    """
    Expect files like:
      - pca_loadings_YYYYMMDD.csv
      - pca_scores_YYYYMMDD.csv
      - pca_summary_YYYYMMDD.csv
      - pca_scree_YYYYMMDD.png

    Selection priority:
      1) If date_tag provided and ALL four exact files exist, use that set.
      2) Otherwise fall back to latest-by-mtime for each artifact type.
    Returns (loadings_path, scores_path, summary_path, scree_path, debug_info)
    """
    patterns = {
        "loadings": os.path.join(pca_dir, "pca_loadings_*.csv"),
        "scores": os.path.join(pca_dir, "pca_scores_*.csv"),
        "summary": os.path.join(pca_dir, "pca_summary_*.csv"),
        "scree": os.path.join(pca_dir, "pca_scree_*.png"),
    }

    if date_tag:
        exact = {
            "loadings": os.path.join(pca_dir, f"pca_loadings_{date_tag}.csv"),
            "scores": os.path.join(pca_dir, f"pca_scores_{date_tag}.csv"),
            "summary": os.path.join(pca_dir, f"pca_summary_{date_tag}.csv"),
            "scree": os.path.join(pca_dir, f"pca_scree_{date_tag}.png"),
        }
        if all(os.path.exists(exact[k]) for k in ["loadings", "scores", "summary", "scree"]):
            dbg = {
                "pca_dir": pca_dir,
                "patterns": patterns,
                "found_counts": {k: len(glob.glob(v)) for k, v in patterns.items()},
                "latest": exact,
                "preferred_date_tag": date_tag,
                "selection": "exact_date_tag",
            }
            return exact["loadings"], exact["scores"], exact["summary"], exact["scree"], dbg

    loadings_path = _latest_file(patterns["loadings"])
    scores_path = _latest_file(patterns["scores"])
    summary_path = _latest_file(patterns["summary"])
    scree_path = _latest_file(patterns["scree"])

    dbg = {
        "pca_dir": pca_dir,
        "patterns": patterns,
        "found_counts": {k: len(glob.glob(v)) for k, v in patterns.items()},
        "latest": {
            "loadings": loadings_path,
            "scores": scores_path,
            "summary": summary_path,
            "scree": scree_path,
        },
        "preferred_date_tag": date_tag,
        "selection": "latest_mtime",
    }
    return loadings_path, scores_path, summary_path, scree_path, dbg


def check_saved_pca_order(remaining_factors: list[str], pca_loadings_path: str) -> tuple[bool, dict]:
    """Return (ok, info). Never exits."""
    info = {"ok": False, "reason": "", "mismatch_at": None}
    if pca_loadings_path is None or (not os.path.exists(pca_loadings_path)):
        info["reason"] = "missing_loadings"
        return False, info

    loadings = pd.read_csv(pca_loadings_path, index_col=0)
    pca_factor_order = loadings.index.tolist()

    if pca_factor_order == remaining_factors:
        info["ok"] = True
        info["reason"] = "match"
        return True, info

    # diff
    m = min(len(pca_factor_order), len(remaining_factors))
    mismatch_at = None
    for i in range(m):
        if pca_factor_order[i] != remaining_factors[i]:
            mismatch_at = i
            break
    info["reason"] = "order_mismatch"
    info["mismatch_at"] = mismatch_at
    info["json_len"] = len(remaining_factors)
    info["pca_len"] = len(pca_factor_order)
    if mismatch_at is not None:
        info["json_val"] = remaining_factors[mismatch_at]
        info["pca_val"] = pca_factor_order[mismatch_at]
    return False, info


def pca_fit_and_benchmark(
    X_scaled_df: pd.DataFrame,
    remaining_factors: list[str],
    out_dir: str,
    n_components: int,
    date_tag: str,
):
    """
    Fit PCA on current standardized data for benchmark (guaranteed consistent).
    Outputs: Z_df, recon_err_series, labels, meta
    """
    from sklearn.decomposition import PCA

    X_use = X_scaled_df.loc[:, remaining_factors].copy()
    k = min(n_components, X_use.shape[1])

    pca = PCA(n_components=k, random_state=0)
    Z = pca.fit_transform(X_use.values)
    X_hat = pca.inverse_transform(Z)  # reconstruction in standardized space
    rec_err = np.mean((X_use.values - X_hat) ** 2, axis=1)

    Z_df = pd.DataFrame(Z, index=X_use.index, columns=[f"PC{i+1}" for i in range(k)])
    rec_err_s = pd.Series(rec_err, index=X_use.index, name="pca_recon_mse")

    # save minimal benchmark PCA artifacts into output dir
    pca_out = os.path.join(out_dir, "pca_benchmark_recomputed")
    os.makedirs(pca_out, exist_ok=True)

    # summary
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    summary = pd.DataFrame({
        "pc": [f"PC{i+1}" for i in range(len(evr))],
        "explained_var_ratio": evr,
        "cum_explained_var_ratio": cum,
    })
    tag = resolve_date_tag(date_tag)
    summary_path = os.path.join(pca_out, f"pca_summary_{tag}.csv")
    summary.to_csv(summary_path, index=False)

    # loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        index=remaining_factors,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
    )
    loadings_path = os.path.join(pca_out, f"pca_loadings_{tag}.csv")
    loadings.to_csv(loadings_path)

    # scores
    scores_path = os.path.join(pca_out, f"pca_scores_{tag}.csv")
    Z_df.to_csv(scores_path)

    # scree
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(evr) + 1), evr, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Scree Plot (benchmark recomputed)")
    plt.tight_layout()
    scree_path = os.path.join(pca_out, f"pca_scree_{tag}.png")
    plt.savefig(scree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # clustering on Z
    labels = cluster_latent(Z, CLUSTER_METHOD, N_CLUSTERS)

    # visualization
    emb2d = embed_2d(Z, VIS_METHOD)
    plot_scatter(
        emb2d, labels,
        title=f"PCA (recomputed) scores (k={k}) + {CLUSTER_METHOD} k={N_CLUSTERS} ({VIS_METHOD} 2D)",
        out_path=os.path.join(out_dir, f"latent_cluster_PCAk{k}.png")
    )
    plot_scatter_sizecolor(
        emb2d, rec_err,
        title=f"PCA (recomputed) scores colored by recon error (k={k}) ({VIS_METHOD} 2D)",
        out_path=os.path.join(out_dir, f"latent_recon_error_PCAk{k}.png")
    )

    # save pca cluster & profile
    pd.DataFrame({"cluster": labels}, index=X_use.index).to_csv(
        os.path.join(out_dir, f"clusters_PCAk{k}_{CLUSTER_METHOD}_k{N_CLUSTERS}.csv")
    )
    prof = pd.DataFrame(X_use.values, index=X_use.index, columns=remaining_factors)
    prof["cluster"] = labels
    prof.groupby("cluster").mean(numeric_only=True).to_csv(
        os.path.join(out_dir, f"cluster_profile_mean_PCAk{k}.csv")
    )

    meta = {
        "mode": "recomputed",
        "pca_out_dir": pca_out,
        "loadings_path": loadings_path,
        "scores_path": scores_path,
        "summary_path": summary_path,
        "scree_path": scree_path,
        "k": k,
    }
    return Z_df, rec_err_s, labels, meta


def pca_benchmark_auto(
    X_scaled_df: pd.DataFrame,
    remaining_factors: list[str],
    out_dir: str,
    n_components: int,
    pca_dir: str,
    date_tag: str,
):
    """
    Try reuse saved PCA artifacts under pca_dir if (1) files exist and (2) factor order matches JSON.
    Otherwise print detailed reasons and fallback to recomputing PCA on current X.
    """
    loadings_path, scores_path, summary_path, scree_path, dbg = load_latest_pca_artifacts(pca_dir, date_tag=date_tag)

    # more specific debug output if no coverage
    if loadings_path is None or scores_path is None:
        print("[warn] Saved PCA artifacts not usable (missing files). Will recompute PCA for benchmark.")
        print(f"       PCA_DIR={pca_dir}")
        print(f"       Patterns & found counts: {dbg['found_counts']}")
        print(f"       Latest candidates: {dbg['latest']}")
        # show a few matching filenames for diagnostics
        for k, pat in dbg["patterns"].items():
            files = sorted(glob.glob(pat))
            if files:
                print(f"       Examples for {k}: {files[-3:]}")
        return pca_fit_and_benchmark(X_scaled_df, remaining_factors, out_dir, n_components, date_tag)

    ok, info = check_saved_pca_order(remaining_factors, loadings_path)
    if not ok:
        print("[warn] Saved PCA factor order does NOT match JSON/AE order. Will recompute PCA for benchmark.")
        print(f"       PCA loadings used: {loadings_path}")
        print(f"       Reason: {info.get('reason')}")
        if info.get("reason") == "order_mismatch":
            print(f"       JSON length={info.get('json_len')} | PCA length={info.get('pca_len')}")
            if info.get("mismatch_at") is not None:
                i = info["mismatch_at"]
                print(f"       First mismatch at position {i}:")
                print(f"         JSON : {info.get('json_val')}")
                print(f"         PCA  : {info.get('pca_val')}")
        return pca_fit_and_benchmark(X_scaled_df, remaining_factors, out_dir, n_components, date_tag)

    # order matches, reuse scores/loadings
    print(">>> Saved PCA artifacts found and factor order matches. Reusing for benchmark.")
    loadings = pd.read_csv(loadings_path, index_col=0)
    scores = pd.read_csv(scores_path, index_col=0)

    common = X_scaled_df.index.intersection(scores.index)
    if len(common) == 0:
        print("[warn] No overlapping stocks between current data and saved PCA scores. Recompute PCA.")
        return pca_fit_and_benchmark(X_scaled_df, remaining_factors, out_dir, n_components, date_tag)

    X_use = X_scaled_df.loc[common, remaining_factors].copy()
    scores = scores.loc[common].copy()

    # Determine usable k
    pc_cols = [c for c in scores.columns if re.match(r"^PC\d+$", str(c))]
    if len(pc_cols) == 0:
        pc_cols = list(scores.columns)

    k = min(n_components, len(pc_cols))
    pc_cols = pc_cols[:k]

    # Reconstruct in standardized space (approx center as current mean; should match if preprocess same)
    components = loadings[[f"PC{i+1}" for i in range(min(k, loadings.shape[1]))]].T.values  # (k, n_features)
    Z = scores[pc_cols].values  # (n_samples, k)
    X_mean = X_use.values.mean(axis=0, keepdims=True)
    X_hat = Z @ components[:k, :] + X_mean
    rec_err = np.mean((X_use.values - X_hat) ** 2, axis=1)

    Z_df = pd.DataFrame(Z, index=common, columns=pc_cols)
    rec_err_s = pd.Series(rec_err, index=common, name="pca_recon_mse")

    labels = cluster_latent(Z, CLUSTER_METHOD, N_CLUSTERS)

    emb2d = embed_2d(Z, VIS_METHOD)
    plot_scatter(
        emb2d, labels,
        title=f"PCA scores (saved) (k={k}) + {CLUSTER_METHOD} k={N_CLUSTERS} ({VIS_METHOD} 2D)",
        out_path=os.path.join(out_dir, f"latent_cluster_PCAk{k}.png")
    )
    plot_scatter_sizecolor(
        emb2d, rec_err,
        title=f"PCA scores (saved) colored by recon error (k={k}) ({VIS_METHOD} 2D)",
        out_path=os.path.join(out_dir, f"latent_recon_error_PCAk{k}.png")
    )

    pd.DataFrame({"cluster": labels}, index=common).to_csv(
        os.path.join(out_dir, f"clusters_PCAk{k}_{CLUSTER_METHOD}_k{N_CLUSTERS}.csv")
    )
    prof = pd.DataFrame(X_use.values, index=common, columns=remaining_factors)
    prof["cluster"] = labels
    prof.groupby("cluster").mean(numeric_only=True).to_csv(
        os.path.join(out_dir, f"cluster_profile_mean_PCAk{k}.csv")
    )

    meta = {
        "mode": "reused",
        "pca_dir": pca_dir,
        "loadings_path": loadings_path,
        "scores_path": scores_path,
        "summary_path": summary_path,
        "scree_path": scree_path,
        "k": k,
        "debug": dbg,
    }
    return Z_df, rec_err_s, labels, meta
# ----------------------------
def plot_benchmark(bm_df: pd.DataFrame, out_path: str):
    """
    bm_df rows: ["AE","PCA"], cols include recon_mse_mean, silhouette, calinski_harabasz, davies_bouldin
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    # left axis: recon mse mean
    ax.plot(bm_df.index, bm_df["recon_mse_mean"], marker="o")
    ax.set_ylabel("Mean Recon MSE (lower better)")
    ax.set_title("AE vs PCA Benchmark")

    ax2 = ax.twinx()
    ax2.plot(bm_df.index, bm_df["silhouette"], marker="o")
    ax2.set_ylabel("Silhouette (higher better)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">>> saved benchmark plot: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="Run date tag YYYYMMDD; default env RUN_DATE/DATE_TAG or today")
    p.add_argument("--h5", default=H5_PATH, help="HDF5 path (default: factor/purified.h5)")
    p.add_argument("--json", default=None, help="Remaining factors json path (default: factor/remaining_factors_{date}.json)")
    p.add_argument("--out", default=None, help="Output directory (default: factor/ae_out_{date})")
    p.add_argument("--pca-dir", default=PCA_DIR, help="Directory holding PCA artifacts from preanalysis")
    p.add_argument("--model", default="none", help='AE model path; "none" to retrain (default)')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)

    date_tag = resolve_date_tag(args.date)
    json_path = args.json or f"factor/remaining_factors_{date_tag}.json"
    out_dir = args.out or f"factor/ae_out_{date_tag}"
    pca_dir = args.pca_dir

    os.makedirs(out_dir, exist_ok=True)

    remaining = load_remaining_factors(json_path)

    # Build matrix with EXACT json order
    X_df, db_df = build_feature_matrix(args.h5, remaining)
    # make sure X_df columns order == remaining (after dropping missing factors)
    X_df = X_df.reindex(columns=[c for c in remaining if c in X_df.columns])

    # basic cleaning
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    X_filled = X_num.fillna(X_num.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled.values)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_df.index, columns=X_df.columns)

    scaler_path = os.path.join(out_dir, "scaler_params.npz")
    np.savez(scaler_path, mean_=scaler.mean_, scale_=scaler.scale_, columns=np.array(X_df.columns))
    print(f">>> saved scaler params: {scaler_path}")

    # ---- AE train/load -> latent
    model_path = None if str(args.model).lower() == "none" else args.model
    Z_ae, rec_err_ae = train_or_load_ae_get_latent(X_scaled, out_dir, model_path)

    latent_df = pd.DataFrame(Z_ae, index=X_df.index, columns=[f"z{i+1}" for i in range(Z_ae.shape[1])])
    latent_df.to_csv(os.path.join(out_dir, "ae_latent.csv"))
    pd.DataFrame({"recon_mse": rec_err_ae}, index=X_df.index).to_csv(os.path.join(out_dir, "recon_error.csv"))

    # ---- AE clustering & viz
    labels_ae = cluster_latent(Z_ae, CLUSTER_METHOD, N_CLUSTERS)
    pd.DataFrame({"cluster": labels_ae}, index=X_df.index).to_csv(
        os.path.join(out_dir, f"clusters_AE_{CLUSTER_METHOD}_k{N_CLUSTERS}.csv")
    )

    prof = X_filled.copy()
    prof["cluster"] = labels_ae
    prof.groupby("cluster").mean(numeric_only=True).to_csv(os.path.join(out_dir, "cluster_profile_mean_AE.csv"))

    emb2d_ae = embed_2d(Z_ae, VIS_METHOD)
    plot_scatter(
        emb2d_ae, labels_ae,
        title=f"AE latent ({LATENT_DIM}d) + {CLUSTER_METHOD} k={N_CLUSTERS} ({VIS_METHOD} 2D)",
        out_path=os.path.join(out_dir, "latent_cluster_AE.png")
    )
    plot_scatter_sizecolor(
        emb2d_ae, rec_err_ae,
        title=f"AE latent colored by recon error ({VIS_METHOD} 2D)",
        out_path=os.path.join(out_dir, "latent_recon_error_AE.png")
    )

    # optional: visualize a daily_basic field if exists
    if not db_df.empty:
        cand_cols = [c for c in ["pct_chg", "close", "vol", "amount", "turnover_rate"] if c in db_df.columns]
        if cand_cols:
            c0 = cand_cols[0]
            v = pd.to_numeric(db_df[c0], errors="coerce").reindex(X_df.index).fillna(0.0).values
            plot_scatter_sizecolor(
                emb2d_ae, v,
                title=f"AE latent colored by daily_basic.{c0} ({VIS_METHOD} 2D)",
                out_path=os.path.join(out_dir, f"latent_daily_basic_{c0}.png")
            )

    # ---- PCA benchmark (reuse saved PCA only if order matches; else exit inside)
    Z_pca_df, rec_err_pca_s, labels_pca, pca_meta = pca_benchmark_auto(
        X_scaled_df=X_scaled_df,
        remaining_factors=list(X_df.columns),   # actual cols used (after drop missing)
        out_dir=out_dir,
        n_components=LATENT_DIM,
        pca_dir=pca_dir,
        date_tag=date_tag,
    )

    # ---- Benchmark table
    ae_metrics = compute_cluster_metrics(Z_ae, labels_ae)
    pca_metrics = compute_cluster_metrics(Z_pca_df.values, labels_pca)

    bm = pd.DataFrame(index=["AE", "PCA"])
    bm.loc["AE", "recon_mse_mean"] = float(np.mean(rec_err_ae))
    bm.loc["PCA", "recon_mse_mean"] = float(rec_err_pca_s.mean())
    bm.loc["AE", "recon_mse_p90"] = float(np.quantile(rec_err_ae, 0.9))
    bm.loc["PCA", "recon_mse_p90"] = float(rec_err_pca_s.quantile(0.9))

    for k, v in ae_metrics.items():
        bm.loc["AE", k] = v
    for k, v in pca_metrics.items():
        bm.loc["PCA", k] = v

    bm_path = os.path.join(out_dir, "benchmark_ae_vs_pca.csv")
    bm.to_csv(bm_path)
    print(f">>> saved benchmark table: {bm_path}")
    print("\n>>> Benchmark summary:")
    print(bm.to_string())

    plot_benchmark(bm, os.path.join(out_dir, "benchmark_ae_vs_pca.png"))

    # Save PCA meta pointers for traceability
    with open(os.path.join(out_dir, "pca_benchmark_inputs.json"), "w", encoding="utf-8") as f:
        json.dump(pca_meta, f, indent=2, ensure_ascii=False)

    print("\n>>> DONE.")
    print(f">>> outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
