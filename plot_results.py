import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Global Plot Style
plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': True,
    'figure.figsize': (10, 6)
})

# Utility
RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

# Change into float/int
def to_numeric_clean(s: pd.Series) -> pd.Series:
    if s.dtype != "object":
        return pd.to_numeric(s, errors="coerce")
    s2 = (
        s.astype(str)
        .str.strip()
        .str.replace(r"[sS]$", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce")

# Load all results into one dataframe
def load_all_results(results_dir: Path) -> pd.DataFrame:
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {results_dir.resolve()}")
    dfs = []
    for f in csvs:
        df = pd.read_csv(f)
        if "cores" not in df.columns:
            m = re.search(r"core_(\d+)", f.name)
            df["cores"] = int(m.group(1)) if m else None
        df["source_file"] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Standardize column name
def standardize(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "n_splits": "splits",
        "total_time": "total",
        "fit_time": "fit",
        "ext_time": "ext"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    numeric_cols = [
        "N", "d", "centers", "std", "noise_frac", "seed",
        "splits", "splitA_pos", "splitA_size", "chunksize", "knn_eff",
        "fit", "ext", "total", "n_obs", "ari", "ami", "cores"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = to_numeric_clean(df[c])

    for c in ["phase", "method", "dataset", "backend"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "total" in df.columns:
        df = df.dropna(subset=["total"])
    return df

# Speedup and Efficiency according to the increase in the number of cores
def plot_parallel_performance(df, outpath, method_name="parallel_sdoclust"):
    sub = df[df["method"] == method_name].copy()
    group_cols = [c for c in ["dataset", "N", "d", "centers", "splits"] if c in sub.columns]
    
    rows = []
    for _, g in sub.groupby(group_cols):
        g = g.sort_values("cores")
        base_time = g.iloc[0]["total"]
        if base_time > 0:
            g["speedup"] = base_time / g["total"]
            g["efficiency"] = g["speedup"] / g["cores"]
            rows.append(g)
    
    if not rows: return
    perf_df = pd.concat(rows, ignore_index=True)
    agg = perf_df.groupby(["splits", "cores"], as_index=False).agg({"speedup":"mean", "efficiency":"mean"})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for s in sorted(agg["splits"].unique()):
        data = agg[agg["splits"] == s]
        ax1.plot(data["cores"], data["speedup"], marker='o', label=f"splits={int(s)}")
        ax2.plot(data["cores"], data["efficiency"], marker='s', label=f"splits={int(s)}")
    
    ax1.set_title("Speedup vs Number of Cores"), ax1.set_xlabel("Cores"), ax1.set_ylabel("Speedup"), ax1.legend(title="Splits")
    ax2.set_title("Efficiency vs Number of Cores"), ax2.set_xlabel("Cores"), ax2.set_ylabel("Efficiency"), ax2.set_ylim(0, 1.1), ax2.legend(title="Splits")
    fig.suptitle(f"Parallel Performance Analysis: {method_name}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

# How execution time and accuracy (AMI) change when the number of cluster centers increases
def plot_centers_analysis(df, outpath):
    sub = df[df["method"] == "parallel_sdoclust"].copy()
    if "centers" not in sub.columns: return

    agg = sub.groupby(["cores", "centers"], as_index=False).agg({
        "total": "mean", 
        "ami": "mean"
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    center_vals = sorted(agg["centers"].unique())

    for c in center_vals:
        data = agg[agg["centers"] == c]
        ax1.plot(data["cores"], data["total"], marker='o', label=f"centers={int(c)}")
        ax2.plot(data["cores"], data["ami"], marker='s', label=f"centers={int(c)}")

    ax1.set_title("Execution Time by Core Count"), ax1.set_xlabel("Number of Cores"), ax1.set_ylabel("Total Time (sec)"), ax1.legend(title="Centers")
    ax2.set_title("Clustering Accuracy (AMI) by Core Count"), ax2.set_xlabel("Number of Cores"), ax2.set_ylabel("AMI Score"), ax2.set_ylim(0, 1.1), ax2.legend(title="Centers")

    fig.suptitle("Impact of Cluster Centers: Speed vs Accuracy", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

# Compares the drop in accuracy between algorithms as noise_frac increases
def plot_full_robustness_comparison(df, outpath):
    methods_map = {"parallel_sdoclust": "SDO (Parallel)", "sdoclust_full": "SDO (Full)", "minibatch_joblib": "MB-KMeans", "kmeans_seq": "KMeans"}
    sub = df[df["dataset"] == "noisy_blobs"].copy()
    sub = sub[sub["method"].isin(methods_map.keys())]
    sub["method_label"] = sub["method"].map(methods_map)
    agg = sub.groupby(["method_label", "noise_frac"], as_index=False)["ami"].mean()

    plt.figure(figsize=(10, 6))
    for label, data in agg.groupby("method_label"):
        plt.plot(data["noise_frac"], data["ami"], label=label, marker='o')
    plt.title("Clustering Robustness: AMI Score by Noise Fraction")
    plt.xlabel("Noise Fraction"), plt.ylabel("AMI Score"), plt.ylim(0.4, 1.05), plt.legend(title="Algorithms")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

# Compares the absolute execution times of all algorithms
def plot_algorithm_time_series(df, outpath, x_axis="cores"):
    methods_map = {"sdoclust_full": "SDO (Full)", "parallel_sdoclust": "SDO (Parallel)", "kmeans_seq": "KMeans", "minibatch_seq": "MiniBatch KM"}
    sub = df[df["method"].isin(methods_map.keys())].copy()
    sub["method_label"] = sub["method"].map(methods_map)
    
    plt.figure(figsize=(10, 6))
    for label, data in sub.groupby("method_label"):
        agg = data.groupby(x_axis)["total"].mean().sort_index()
        plt.plot(agg.index, agg.values, marker='o', label=label, linewidth=2)
    plt.title(f"Execution Time Comparison by {x_axis.capitalize()}")
    plt.xlabel(x_axis.capitalize()), plt.ylabel("Total Time (sec)"), plt.legend(title="Algorithms")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_results(RESULTS_DIR)
    df = standardize(df)

    # Speedup/Efficiency
    plot_parallel_performance(df, FIG_DIR / "parallel_scalability.png")
    # Noise Robustness (AMI)
    plot_full_robustness_comparison(df, FIG_DIR / "algorithm_robustness.png")
    # Speed comparison between algorithms
    plot_algorithm_time_series(df, FIG_DIR / "algorithm_speed_comparison.png")
    # Centers complexity analysis
    plot_centers_analysis(df, FIG_DIR / "centers_complexity_analysis.png")

    print(f"Visualization complete. Check figures in: {FIG_DIR}")

if __name__ == "__main__":
    main()