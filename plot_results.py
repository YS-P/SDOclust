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
    methods_map = {
        "parallel_sdoclust": "SDO (Parallel)", 
        "sdoclust_full": "SDO (Full)", 
        "minibatch_joblib": "MB-KMeans", 
        "kmeans_seq": "KMeans"
    }
    
    sub = df[df["dataset"] == "noisy_blobs"].copy()
    sub = sub[sub["method"].isin(methods_map.keys())]
    sub["method_label"] = sub["method"].map(methods_map)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    agg_ami = sub.groupby(["method_label", "noise_frac"], as_index=False)["ami"].mean()
    for label, data in agg_ami.groupby("method_label"):
        ax1.plot(data["noise_frac"], data["ami"], label=label, marker='o', linewidth=2)
    
    ax1.set_title("Robustness: AMI Score by Noise Fraction")
    ax1.set_xlabel("Noise Fraction")
    ax1.set_ylabel("AMI Score")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(title="Algorithms")
    ax1.grid(True, alpha=0.3)

    if "ari" in sub.columns:
        agg_ari = sub.groupby(["method_label", "noise_frac"], as_index=False)["ari"].mean()
        for label, data in agg_ari.groupby("method_label"):
            ax2.plot(data["noise_frac"], data["ari"], label=label, marker='s', linewidth=2)
        
        ax2.set_title("Robustness: ARI Score by Noise Fraction")
        ax2.set_xlabel("Noise Fraction")
        ax2.set_ylabel("ARI Score")
        ax2.set_ylim(0.4, 1.05)
        ax2.legend(title="Algorithms")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "ARI Data Not Available", ha='center', va='center')

    fig.suptitle("Clustering Robustness Analysis: AMI vs ARI", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()

# Compares accuracy between Full and Parallel models across different datasets
def plot_accuracy_consistency_comparison(df, outpath):
    target_datasets = ["blobs", "noisy_blobs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    axes = {"blobs": ax1, "noisy_blobs": ax2}

    for ds_name in target_datasets:
        ax = axes[ds_name]
        ds_sub = df[df["dataset"] == ds_name].copy()
        
        full_val = ds_sub[ds_sub["method"] == "sdoclust_full"]["ami"].mean()
        para_sub = ds_sub[ds_sub["method"] == "parallel_sdoclust"].copy()
        para_agg = para_sub.groupby("splits")["ami"].mean().sort_index()
        
        labels = ["Full"] + [str(int(s)) for s in para_agg.index]
        values = [full_val] + para_agg.values.tolist()
        colors = ['#1f77b4'] + ['#ff7f0e'] * len(para_agg)
        
        x_pos = range(len(labels))
        ax.bar(x_pos, values, color=colors, width=0.8, edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        
        ax.set_title(f"Accuracy Consistency: {ds_name.capitalize()}")
        ax.set_xlabel("Number of Splits")
        ax.set_ylabel("AMI Score")
        ax.set_ylim(0, 1.2)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle("Clustering Accuracy Comparison: Full vs Parallel (by Splits)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    
# Compares the absolute execution times of all algorithms
def plot_algorithm_time_series(df, outpath, x_axis="cores"):
    methods_map = {
        "sdoclust_full": "SDO (Full)", 
        "parallel_sdoclust": "SDO (Parallel)", 
        "kmeans_seq": "KMeans", 
        "minibatch_seq": "MiniBatch KM"
    }
    sub = df[df["method"].isin(methods_map.keys())].copy()
    sub["method_label"] = sub["method"].map(methods_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for label, data in sub.groupby("method_label"):
        agg = data.groupby(x_axis)["total"].mean().sort_index()
        ax1.plot(agg.index, agg.values, marker='o', label=label, linewidth=2)
    
    ax1.set_title("Execution Time Comparison by Algorithms")
    ax1.set_xlabel(x_axis.capitalize())
    ax1.set_ylabel("Total Time (sec)")
    ax1.legend(title="Algorithms")
    ax1.grid(True)

    if "backend" in df.columns:
        para_sub = df[df["method"] == "parallel_sdoclust"].copy()
        para_sub = para_sub[para_sub["backend"].isin(["seq", "joblib", "dask"])]
        
        for backend, data in para_sub.groupby("backend"):
            agg = data.groupby(x_axis)["total"].mean().sort_index()
            ax2.plot(agg.index, agg.values, marker='s', label=f"Backend: {backend}", linewidth=2)
        
        ax2.set_title("Scalability: Parallel SDOclust Backend Comparison (seq vs Parallel)")
        ax2.set_xlabel(x_axis.capitalize())
        ax2.set_ylabel("Total Time (sec)")
        ax2.legend(title="Backends")
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "Backend column not found", ha='center', va='center')

    fig.suptitle("Performance and Backend Scalability Analysis", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
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
    # Centers complexity analysis
    plot_centers_analysis(df, FIG_DIR / "centers_complexity_analysis.png") 
    # Accuracy consistency between splits
    plot_accuracy_consistency_comparison(df, FIG_DIR / "accuracy_consistency_analysis.png")
    # Speed comparison between algorithms
    plot_algorithm_time_series(df, FIG_DIR / "algorithm_speed_comparison.png")
    
    print(f"Visualization complete. Check figures in: {FIG_DIR}")

if __name__ == "__main__":
    main()