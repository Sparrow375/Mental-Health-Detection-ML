"""
compare_dim_reduction.py
========================
Extends compare_clusterers.py with a full grid of:

Usage:
    py tests/compare_dim_reduction.py [path/to/sample.json]

If no argument is supplied the script defaults to sampleS.json in the
project root.  Results are saved to tests/results/<json-stem>/.

    Feature Reduction Methods  x  Clustering Algorithms
    ─────────────────────────────────────────────────────
    Reduction methods  (applied to the 12-feature L1 space):
        1. No reduction    (original 12-D, StandardScaler only)
        2. PCA  (2 / 3 / 4 components)
        3. UMAP (2 / 3 components)  ← requires `pip install umap-learn`
        4. t-SNE (2 components)     ← for visualisation; not used for fit
        5. SelectKBest  (variance-based top-k, k=6/8)
        6. VarianceThreshold        (drop near-zero-variance features)
        7. Recursive Feature Elimination via Random-Forest importance (top-6/8)
        8. Kernel PCA  (rbf, 2 / 4 components)

    Clustering algorithms  (same as compare_clusterers.py):
        K-Means  ·  Hierarchical (Ward)  ·  DBSCAN  ·  Mean-Shift  ·  Spectral

Results
-------
• A combined metric table (Silhouette + Davies-Bouldin + Calinski-Harabasz).
• One figure per reduction method  →  5-subplot cluster comparison.
• A heatmap figure  →  Silhouette scores across (method × algorithm).
• Summary JSON saved to tests/results/dim_reduction_results.json.
"""

import sys, json, os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

warnings.filterwarnings("ignore")

# ── project path ─────────────────────────────────────────────────────────────
PROJECT_ROOT = r"f:\Avaneesh\projects\MH detector\Mental-Health-Detection-ML"
sys.path.insert(0, PROJECT_ROOT)

from system1.feature_meta import L1_CLUSTERING_FEATURES, FEATURE_META

# ── scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, MeanShift,
    SpectralClustering, estimate_bandwidth,
)
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# optional UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[WARN] umap-learn not installed → UMAP methods will be skipped.\n"
          "       Install with: pip install umap-learn")

# ── CLI arg: optional path to a JSON data file ───────────────────────────────
if len(sys.argv) > 1:
    _arg = sys.argv[1]
    sample_path = _arg if os.path.isabs(_arg) else os.path.join(PROJECT_ROOT, _arg)
else:
    sample_path = os.path.join(PROJECT_ROOT, "sampleS.json")

_sample_stem = os.path.splitext(os.path.basename(sample_path))[0]

# ── output dir ────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(PROJECT_ROOT, "tests", "results", _sample_stem)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def map_features(metrics_dict: dict) -> dict:
    """Map JSON field names → L1 feature names."""
    mapping = {
        "sleep_duration_hours":    "sleepDurationHours",
        "wake_time_hour":          "wakeTimeHour",
        "sleep_time_hour":         "sleepTimeHour",
        "daily_displacement_km":   "displacementKm",
        "location_entropy":        "locationEntropy",
        "places_visited":          "placesVisited",
        "calls_per_day":           "callsPerDay",
        "screen_time_hours":       "screenTimeHours",
        "unlock_count":            "unlockCount",
        "social_app_ratio":        "socialRatio",
        "dark_duration_hours":     "darkDurationHours",
    }
    out = {l1: metrics_dict.get(json_key, 0.0) for l1, json_key in mapping.items()}
    out["conversation_duration_hours"] = metrics_dict.get("callDurationMins", 0.0) / 60.0
    return out


with open(sample_path, "r") as f:
    data = json.load(f)

daily_features, dates = [], []
for day in data.get("daily_history", []):
    daily_features.append(map_features(day["metrics"]))
    dates.append(day["date"])

df = pd.DataFrame(daily_features)
X_raw = df[L1_CLUSTERING_FEATURES].values.astype(float)
feature_names = list(L1_CLUSTERING_FEATURES)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

n_samples, n_features = X_scaled.shape
print(f"Loaded {n_samples} days × {n_features} features from {sample_path}")

# =============================================================================
# 2.  DIMENSIONALITY / FEATURE REDUCTION METHODS
# =============================================================================

# Clinical weights from FEATURE_META for weighted scoring (optional use)
clinical_weights = np.array([
    FEATURE_META.get(f, {}).get("weight", 1.0) for f in L1_CLUSTERING_FEATURES
])

def _safe_silhouette(X, labels):
    """Return silhouette score, or NaN if not computable."""
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    if n_cl < 2 or n_cl >= len(labels):
        return np.nan
    try:
        return silhouette_score(X, labels)
    except Exception:
        return np.nan

def _safe_db(X, labels):
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    if n_cl < 2:
        return np.nan
    try:
        return davies_bouldin_score(X, labels)
    except Exception:
        return np.nan

def _safe_ch(X, labels):
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    if n_cl < 2:
        return np.nan
    try:
        return calinski_harabasz_score(X, labels)
    except Exception:
        return np.nan


# Each entry: (display_name, 2-D projection for plot, high-D matrix for clustering)
reduction_methods = {}

# 2-A  No reduction
reduction_methods["Original (12-D)"] = {
    "X_cluster": X_scaled,
    "X_2d": PCA(n_components=2).fit_transform(X_scaled),
    "description": "All 12 L1 features, StandardScaler",
    "dim": 12,
}

# 2-B  PCA variants
for nc in [2, 3, 4]:
    pca = PCA(n_components=nc, random_state=42)
    Xp = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_.sum()
    key = f"PCA ({nc}D, {var:.0%} var)"
    reduction_methods[key] = {
        "X_cluster": Xp,
        "X_2d": Xp[:, :2],
        "description": f"PCA → {nc} components ({var:.1%} variance explained)",
        "dim": nc,
    }

# 2-C  Kernel PCA (RBF)
for nc in [2, 4]:
    kpca = KernelPCA(n_components=nc, kernel="rbf", random_state=42)
    Xk = kpca.fit_transform(X_scaled)
    key = f"Kernel PCA (rbf, {nc}D)"
    reduction_methods[key] = {
        "X_cluster": Xk,
        "X_2d": Xk[:, :2],
        "description": f"Kernel PCA (RBF) → {nc} components",
        "dim": nc,
    }

# 2-D  t-SNE (2D only — visualisation; also used for clustering here)
if n_samples > 3:
    perp = min(5, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
    Xt = tsne.fit_transform(X_scaled)
    reduction_methods["t-SNE (2D)"] = {
        "X_cluster": Xt,
        "X_2d": Xt,
        "description": f"t-SNE → 2D (perplexity={perp})",
        "dim": 2,
    }

# 2-E  UMAP
if HAS_UMAP:
    for nc in [2, 3]:
        um = umap.UMAP(n_components=nc, random_state=42, n_neighbors=min(5, n_samples-1))
        Xu = um.fit_transform(X_scaled)
        key = f"UMAP ({nc}D)"
        reduction_methods[key] = {
            "X_cluster": Xu,
            "X_2d": Xu[:, :2],
            "description": f"UMAP → {nc} components",
            "dim": nc,
        }

# 2-F  VarianceThreshold
vt = VarianceThreshold(threshold=0.1)
Xvt = vt.fit_transform(X_scaled)
selected_vt = np.array(feature_names)[vt.get_support()]
proj_vt = PCA(n_components=2).fit_transform(Xvt) if Xvt.shape[1] >= 2 else Xvt
reduction_methods[f"VarianceThreshold ({Xvt.shape[1]}D)"] = {
    "X_cluster": Xvt,
    "X_2d": proj_vt,
    "description": f"Removed near-zero-var feats → {Xvt.shape[1]} remain: {list(selected_vt)}",
    "dim": Xvt.shape[1],
    "selected_features": list(selected_vt),
}

# 2-G  SelectKBest  (using variance as proxy score via f_classif with dummy labels)
#      We use a weak label derived from K-Means (k=2) as a proxy target for f_classif
_km_proxy = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
_proxy_labels = _km_proxy.labels_
for k_best in [6, 8]:
    skb = SelectKBest(f_classif, k=min(k_best, n_features))
    Xsk = skb.fit_transform(X_scaled, _proxy_labels)
    sel_feats = np.array(feature_names)[skb.get_support()]
    proj_sk = PCA(n_components=2).fit_transform(Xsk) if Xsk.shape[1] >= 2 else Xsk
    key = f"SelectKBest (top-{k_best})"
    reduction_methods[key] = {
        "X_cluster": Xsk,
        "X_2d": proj_sk,
        "description": f"SelectKBest f_classif top-{k_best}: {list(sel_feats)}",
        "dim": Xsk.shape[1],
        "selected_features": list(sel_feats),
    }

# 2-H  RFE via Random-Forest importance
for k_rfe in [6, 8]:
    _rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(_rf, n_features_to_select=min(k_rfe, n_features), step=1)
    rfe.fit(X_scaled, _proxy_labels)
    Xrfe = X_scaled[:, rfe.support_]
    sel_feats_rfe = np.array(feature_names)[rfe.support_]
    proj_rfe = PCA(n_components=2).fit_transform(Xrfe) if Xrfe.shape[1] >= 2 else Xrfe
    key = f"RFE/RF (top-{k_rfe})"
    reduction_methods[key] = {
        "X_cluster": Xrfe,
        "X_2d": proj_rfe,
        "description": f"RFE (RF importance) top-{k_rfe}: {list(sel_feats_rfe)}",
        "dim": Xrfe.shape[1],
        "selected_features": list(sel_feats_rfe),
    }

# 2-I  Clinically Weighted PCA  (scale each feature by its clinical weight, then PCA-2)
X_weighted = X_scaled * clinical_weights
cwpca = PCA(n_components=2, random_state=42)
Xcw = cwpca.fit_transform(X_weighted)
var_cw = cwpca.explained_variance_ratio_.sum()
reduction_methods[f"Clinical-Weighted PCA (2D, {var_cw:.0%})"] = {
    "X_cluster": Xcw,
    "X_2d": Xcw,
    "description": f"Features weighted by clinical importance then PCA-2 ({var_cw:.1%} var)",
    "dim": 2,
}

print(f"\n{len(reduction_methods)} reduction methods configured.\n")

# =============================================================================
# 3.  CLUSTERING ALGORITHMS
# =============================================================================

def run_clusterers(X):
    """Run all 5 clusterers on matrix X. Returns dict {name: labels}."""
    results = {}

    # K-Means — best K ∈ {2,3,4}
    best_k, best_s = 2, -1
    for k in range(2, 5):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        s = _safe_silhouette(X, lbl)
        if not np.isnan(s) and s > best_s:
            best_s, best_k = s, k
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    results[f"K-Means (k={best_k})"] = km_final.fit_predict(X)

    # Hierarchical Ward
    ac = AgglomerativeClustering(n_clusters=None, distance_threshold=3.0)
    results["Hierarchical (Ward)"] = ac.fit_predict(X)

    # DBSCAN
    db = DBSCAN(eps=1.5, min_samples=2)
    results["DBSCAN (ε=1.5)"] = db.fit_predict(X)

    # Mean-Shift
    bw = estimate_bandwidth(X, quantile=0.3, n_samples=len(X))
    if bw == 0:
        bw = 1.0
    ms = MeanShift(bandwidth=bw)
    results["Mean-Shift"] = ms.fit_predict(X)

    # Spectral
    sc = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=42)
    results["Spectral (k=2)"] = sc.fit_predict(X)

    return results

# =============================================================================
# 4.  RUN EVERYTHING + COLLECT METRICS
# =============================================================================

all_metrics = []   # rows for summary table
all_results = {}   # {reduction_method: {clusterer: {labels, metrics}}}

for red_name, red_info in reduction_methods.items():
    Xc = red_info["X_cluster"]
    clusterer_results = run_clusterers(Xc)
    all_results[red_name] = {}

    for cl_name, labels in clusterer_results.items():
        sil = _safe_silhouette(Xc, labels)
        db_s = _safe_db(Xc, labels)
        ch_s = _safe_ch(Xc, labels)
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)

        all_results[red_name][cl_name] = {
            "labels": labels.tolist(),
            "n_clusters": n_cl,
            "n_outliers": int(np.sum(labels == -1)),
            "silhouette": round(float(sil), 4) if not np.isnan(sil) else None,
            "davies_bouldin": round(float(db_s), 4) if not np.isnan(db_s) else None,
            "calinski_harabasz": round(float(ch_s), 2) if not np.isnan(ch_s) else None,
        }

        all_metrics.append({
            "Reduction": red_name,
            "Dim": red_info["dim"],
            "Clusterer": cl_name,
            "Clusters": n_cl,
            "Outliers": int(np.sum(labels == -1)),
            "Silhouette ↑": round(float(sil), 4) if not np.isnan(sil) else np.nan,
            "Davies-Bouldin ↓": round(float(db_s), 4) if not np.isnan(db_s) else np.nan,
            "Calinski-Harabasz ↑": round(float(ch_s), 2) if not np.isnan(ch_s) else np.nan,
        })

metrics_df = pd.DataFrame(all_metrics)
print("\n" + "="*90)
print("METRICS TABLE")
print("="*90)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(metrics_df.to_string(index=False))

# =============================================================================
# 5.  SAVE JSON RESULTS
# =============================================================================

json_payload = {
    "n_samples": n_samples,
    "n_base_features": n_features,
    "base_features": feature_names,
    "dates": dates,
    "reduction_methods": {
        k: {
            "description": v["description"],
            "dim": v["dim"],
            "selected_features": v.get("selected_features"),
        }
        for k, v in reduction_methods.items()
    },
    "results": all_results,
}

json_path = os.path.join(OUT_DIR, "dim_reduction_results.json")
with open(json_path, "w") as f:
    json.dump(json_payload, f, indent=2)
print(f"\nJSON results saved → {json_path}")

# =============================================================================
# 6.  HEATMAP — Silhouette scores across (reduction × clusterer)
# =============================================================================

pivot_sil = metrics_df.pivot_table(
    index="Reduction", columns="Clusterer", values="Silhouette ↑", aggfunc="mean"
)
pivot_cl = metrics_df.pivot_table(
    index="Reduction", columns="Clusterer", values="Clusters", aggfunc="mean"
)

fig_heat, axes_heat = plt.subplots(1, 2, figsize=(22, max(6, len(pivot_sil) * 0.55 + 2)))
fig_heat.patch.set_facecolor("#0f1117")

cmap_sil = sns.diverging_palette(10, 130, as_cmap=True)
sns.heatmap(
    pivot_sil, ax=axes_heat[0],
    annot=True, fmt=".3f", cmap=cmap_sil, center=0,
    linewidths=0.5, linecolor="#222", cbar_kws={"label": "Silhouette Score"},
    annot_kws={"size": 8},
)
axes_heat[0].set_title("Silhouette Score  (↑ better)", color="white", pad=12, fontsize=12)
axes_heat[0].set_facecolor("#0f1117")
axes_heat[0].tick_params(colors="white", labelsize=7)
axes_heat[0].set_xlabel("Clusterer", color="white")
axes_heat[0].set_ylabel("Reduction Method", color="white")

cmap_cl = sns.light_palette("#4fc3f7", as_cmap=True)
sns.heatmap(
    pivot_cl, ax=axes_heat[1],
    annot=True, fmt=".0f", cmap=cmap_cl,
    linewidths=0.5, linecolor="#222",
    annot_kws={"size": 8},
)
axes_heat[1].set_title("Number of Clusters Found", color="white", pad=12, fontsize=12)
axes_heat[1].set_facecolor("#0f1117")
axes_heat[1].tick_params(colors="white", labelsize=7)
axes_heat[1].set_xlabel("Clusterer", color="white")
axes_heat[1].set_ylabel("", color="white")

plt.suptitle(
    "L1 Behavioral Clustering  ·  Dimensionality Reduction × Clustering Algorithm\n"
    f"({n_samples} days, {n_features} base features)",
    color="white", fontsize=14, y=1.01,
)
plt.tight_layout()
heatmap_path = os.path.join(OUT_DIR, "heatmap_dim_reduction.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight", facecolor=fig_heat.get_facecolor())
plt.close(fig_heat)
print(f"Heatmap saved → {heatmap_path}")

# =============================================================================
# 7.  PER-REDUCTION CLUSTER PLOTS
# =============================================================================

CLUST_COLORS = plt.get_cmap("tab10")
DATE_LABELS  = [f"{d[8:]}-{d[5:7]}" for d in dates]

for red_name, red_info in reduction_methods.items():
    clusterer_data = all_results[red_name]
    n_methods = len(clusterer_data)
    X_2d = red_info["X_2d"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.patch.set_facecolor("#0f1117")
    axes = axes.flatten()

    for i, (cl_name, cl_data) in enumerate(clusterer_data.items()):
        ax = axes[i]
        ax.set_facecolor("#1a1d2e")
        labels = np.array(cl_data["labels"])
        n_cl   = cl_data["n_clusters"]
        sil    = cl_data["silhouette"]
        sil_str = f"{sil:.3f}" if sil is not None else "N/A"

        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels, cmap="tab10", s=120,
            alpha=0.85, edgecolor="white", linewidths=0.5,
        )
        for idx, lbl in enumerate(DATE_LABELS):
            ax.annotate(
                lbl, (X_2d[idx, 0], X_2d[idx, 1]),
                fontsize=5.5, alpha=0.75, color="white",
                xytext=(3, 3), textcoords="offset points",
            )

        ax.set_title(
            f"{cl_name}\nClusters: {n_cl}  ·  Silhouette: {sil_str}",
            color="white", fontsize=9, pad=6,
        )
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#333")
        ax.grid(True, linestyle="--", alpha=0.2, color="#555")
        handles, lbls = scatter.legend_elements()
        ax.legend(handles, lbls, title="Label", loc="best",
                  fontsize=7, title_fontsize=7,
                  labelcolor="white",
                  facecolor="#0f1117", edgecolor="#444")

    # last subplot → info card
    ax_info = axes[5]
    ax_info.set_facecolor("#0f1117")
    ax_info.axis("off")
    info_text = (
        f"Reduction: {red_name}\n\n"
        f"{red_info['description']}\n\n"
        f"Projected to 2D via PCA for visualisation\n"
        f"(clustering performed in {red_info['dim']}-D space)\n\n"
        f"Dataset: {n_samples} days  ·  Base: {n_features} features"
    )
    ax_info.text(
        0.05, 0.5, info_text,
        va="center", ha="left", fontsize=9, color="white",
        wrap=True, transform=ax_info.transAxes,
        bbox=dict(facecolor="#1a1d2e", edgecolor="#4fc3f7", boxstyle="round,pad=0.8"),
    )

    safe_name = red_name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "").replace(",", "").replace("%", "pct").replace("=","").replace(".","")
    fig_path = os.path.join(OUT_DIR, f"clusters_{safe_name}.png")
    plt.suptitle(
        f"Clustering Comparison  ·  Reduction: {red_name}\n"
        f"Mental Health L1 Feature Space",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {fig_path}")

# =============================================================================
# 8.  BEST CONFIGURATIONS SUMMARY
# =============================================================================

print("\n" + "="*90)
print("TOP 10 CONFIGURATIONS BY SILHOUETTE SCORE")
print("="*90)
top10 = (
    metrics_df
    .dropna(subset=["Silhouette ↑"])
    .sort_values("Silhouette ↑", ascending=False)
    .head(10)
    [["Reduction", "Dim", "Clusterer", "Clusters", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]]
)
print(top10.to_string(index=False))

# ── Best by Davies-Bouldin (lower is better) ──────────────────────────────────
print("\n" + "="*90)
print("TOP 5 CONFIGURATIONS BY DAVIES-BOULDIN SCORE (lower ↓ better)")
print("="*90)
top5_db = (
    metrics_df
    .dropna(subset=["Davies-Bouldin ↓"])
    .sort_values("Davies-Bouldin ↓", ascending=True)
    .head(5)
    [["Reduction", "Dim", "Clusterer", "Clusters", "Silhouette ↑", "Davies-Bouldin ↓"]]
)
print(top5_db.to_string(index=False))

csv_path = os.path.join(OUT_DIR, "dim_reduction_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"\nFull metrics saved → {csv_path}")
print("\n✓ All done!")
