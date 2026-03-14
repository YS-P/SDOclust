import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Plot Style
plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': True,
    'figure.figsize': (10, 6)
})

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

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
        "n_jobs",
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


def amdahl_speedup(n, p):
    return 1.0 / ((1.0 - p) + p / np.asarray(n, dtype=float))

def estimate_amdahl_p(speedups, core_counts):
    estimates = []
    for S, n in zip(speedups, core_counts):
        if n > 1 and S > 0:
            p = (1.0 / S - 1.0) / (1.0 / n - 1.0)
            estimates.append(float(np.clip(p, 0.0, 1.0)))
    return float(np.mean(estimates)) if estimates else float("nan")


# Plot 1: Speedup & Efficiency
def plot_parallel_performance(df, outpath, method_name="parallel_sdoclust"):
    sub = df[df["method"] == method_name].copy()
    sub = sub[sub["centers"] <= 10]
    group_cols = [c for c in ["dataset", "N", "d", "centers", "splits"] if c in sub.columns]

    rows = []
    for _, g in sub.groupby(group_cols):
        g = g.sort_values("cores")
        base_time = g.iloc[0]["total"]
        if base_time > 0:
            g = g.copy()
            g["speedup"] = base_time / g["total"]
            g["efficiency"] = g["speedup"] / g["cores"]
            rows.append(g)

    if not rows:
        return
    perf_df = pd.concat(rows, ignore_index=True)
    agg = perf_df.groupby(["splits", "cores"], as_index=False).agg(
        {"speedup": "mean", "efficiency": "mean"}
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for s in sorted(agg["splits"].unique()):
        data = agg[agg["splits"] == s].sort_values("cores")
        ax1.plot(data["cores"], data["speedup"], marker='o', label=f"splits={int(s)}")
        ax2.plot(data["cores"], data["efficiency"], marker='s', label=f"splits={int(s)}")

    ax1.set_title("Speedup vs Number of Cores")
    ax1.set_xlabel("Cores")
    ax1.set_ylabel("Speedup")
    ax1.legend(title="Splits")

    ax2.set_title("Efficiency vs Number of Cores")
    ax2.set_xlabel("Cores")
    ax2.set_ylabel("Efficiency")
    ax2.set_ylim(0, 1.2)
    ax2.legend(title="Splits")

    fig.suptitle(f"Parallel Performance Analysis: {method_name}", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# Plot 2: Amdahl's Law Fit
def plot_amdahl_analysis(df, outpath, method_name="parallel_sdoclust"):
    sub = df[(df["method"] == method_name) & (df["backend"] != "seq")].copy()
    seq = df[(df["method"] == method_name) & (df["backend"] == "seq")].copy()
    sub = sub[sub["N"] != 1000000]
    seq = seq[seq["N"] != 1000000]
    sub = sub[sub["centers"] <= 10]

    if sub.empty or seq.empty:
        print("  [amdahl] Insufficient data – skipping.")
        return

    backends = sorted(sub["backend"].unique())
    splits_vals = sorted(sub["splits"].dropna().unique())

    fig, axes = plt.subplots(
        len(backends), len(splits_vals),
        figsize=(4.5 * len(splits_vals), 4 * len(backends)),
        squeeze=False
    )

    for bi, backend in enumerate(backends):
        for si, n_splits in enumerate(splits_vals):
            ax = axes[bi][si]

            par_rows = sub[(sub["backend"] == backend) & (sub["splits"] == n_splits)]
            seq_rows = seq[seq["splits"] == n_splits]

            if par_rows.empty or seq_rows.empty:
                ax.set_visible(False)
                continue

            core_vals = sorted(par_rows["cores"].dropna().unique())
            t_seq_ref = seq_rows["total"].mean()

            speedups = []
            for n in core_vals:
                t_par = par_rows[par_rows["cores"] == n]["total"].mean()
                speedups.append(t_seq_ref / t_par if t_par > 0 else np.nan)

            valid = [(S, n) for S, n in zip(speedups, core_vals) if not np.isnan(S)]
            if not valid:
                continue

            S_vals, n_vals = zip(*valid)
            p = estimate_amdahl_p(list(S_vals), list(n_vals))

            ax.scatter(n_vals, S_vals, color="steelblue", zorder=5, label="Empirical")

            n_fine = np.linspace(1, max(n_vals) * 1.1, 200)
            if not np.isnan(p):
                ax.plot(n_fine, amdahl_speedup(n_fine, p), color="tomato",
                        linewidth=1.8, label=f"Amdahl fit\np={p:.3f}")

            ax.plot(n_fine, n_fine, linestyle="--", color="gray", alpha=0.5, label="Ideal")

            ax.set_title(f"{backend} | splits={int(n_splits)}", fontsize=9)
            ax.set_xlabel("Cores")
            ax.set_ylabel("Speedup")

            if not np.isnan(p):
                max_su = 1.0 / (1.0 - p) if p < 1 else float("inf")
                note = (f"1-p={1-p:.3f}\n"
                        f"Max S∞={max_su:.1f}x" if max_su < 1000 else f"1-p={1-p:.3f}\nMax S∞→∞")
                ax.text(0.97, 0.05, note, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

            ax.legend(fontsize=8)

    fig.suptitle("Amdahl's Law Analysis — Empirical vs Theoretical Speedup",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# Plot 3: Efficiency Drop heat-map
def plot_efficiency_drop(df, outpath, method_name="parallel_sdoclust"):
    sub = df[(df["method"] == method_name) & (df["backend"] != "seq")].copy()
    seq = df[(df["method"] == method_name) & (df["backend"] == "seq")].copy()
    sub = sub[sub["centers"] <= 10]

    if sub.empty or seq.empty:
        print("  [efficiency_drop] Insufficient data – skipping.")
        return

    backends = sorted(sub["backend"].unique())
    fig, axes = plt.subplots(1, len(backends), figsize=(6 * len(backends), 5), squeeze=False)

    for bi, backend in enumerate(backends):
        ax = axes[0][bi]
        par_rows = sub[sub["backend"] == backend]

        core_vals = sorted(par_rows["cores"].dropna().unique())
        split_vals = sorted(par_rows["splits"].dropna().unique())

        mat = np.full((len(split_vals), len(core_vals)), np.nan)

        for si, n_splits in enumerate(split_vals):
            seq_t = seq[seq["splits"] == n_splits]["total"].mean()
            for ci, n_cores in enumerate(core_vals):
                par_t = par_rows[
                    (par_rows["splits"] == n_splits) & (par_rows["cores"] == n_cores)
                ]["total"].mean()
                if not np.isnan(par_t) and par_t > 0 and not np.isnan(seq_t) and seq_t > 0:
                    S = seq_t / par_t
                    mat[si, ci] = S / n_cores

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")
        ax.set_xticks(range(len(core_vals)))
        ax.set_xticklabels([str(int(c)) for c in core_vals])
        ax.set_yticks(range(len(split_vals)))
        ax.set_yticklabels([str(int(s)) for s in split_vals])
        ax.set_xlabel("Cores")
        ax.set_ylabel("Splits")
        ax.set_title(f"Efficiency — backend={backend}")

        for si in range(len(split_vals)):
            for ci in range(len(core_vals)):
                v = mat[si, ci]
                if not np.isnan(v):
                    ax.text(ci, si, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="black")

        plt.colorbar(im, ax=ax, label="Efficiency (S/n)")

    fig.suptitle("Parallel Efficiency Heat-map", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# Plot 4: Centers analysis
def plot_centers_analysis(df, outpath):
    sub = df[df["method"] == "parallel_sdoclust"].copy()
    sub = sub[sub["N"] != 1000000]
    if "centers" not in sub.columns:
        return

    agg = sub.groupby(["cores", "centers"], as_index=False).agg(
        {"total": "mean", "ami": "mean"}
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for c in sorted(agg["centers"].unique()):
        data = agg[agg["centers"] == c].sort_values("cores")
        ax1.plot(data["cores"], data["total"], marker='o', label=f"centers={int(c)}")
        ax2.plot(data["cores"], data["ami"], marker='s', label=f"centers={int(c)}")

    ax1.set_title("Execution Time by Core Count (N≤200k)")
    ax1.set_xlabel("Number of Cores")
    ax1.set_ylabel("Total Time (sec)")
    ax1.legend(title="Centers")

    ax2.set_title("Clustering Accuracy (AMI) by Core Count (N≤200k)")
    ax2.set_xlabel("Number of Cores")
    ax2.set_ylabel("AMI Score")
    ax2.set_ylim(0, 1.1)
    ax2.legend(title="Centers")

    fig.suptitle("Impact of Cluster Centers: Speed vs Accuracy", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

# Plot 5: AMI/ARI vs noise_frac, split by centers=50 vs centers<=10, for SDO Full and SDO Parallel
def plot_centers_accuracy_comparison(df, outpath):
    sub = df[df["dataset"] == "noisy_blobs"].copy()
    sub = sub[sub["method"].isin(["parallel_sdoclust", "sdoclust_full"])]
    methods_map = {
        "parallel_sdoclust": "SDO (Parallel)",
        "sdoclust_full":     "SDO (Full)",
    }
    sub["method_label"] = sub["method"].map(methods_map)

    style = {
        ("SDO (Full)",     "low"):  ("-",  "o"),
        ("SDO (Full)",     "high"): ("-", "o"),
        ("SDO (Parallel)", "low"):  ("--",  "s"),
        ("SDO (Parallel)", "high"): ("--", "s"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip([ax1, ax2], ["ami", "ari"]):
        for label in ["SDO (Full)", "SDO (Parallel)"]:
            grp = sub[sub["method_label"] == label]
            low = grp[grp["centers"] <= 10].groupby("noise_frac")[metric].mean()
            high = grp[grp["centers"] == 50].groupby("noise_frac")[metric].mean()
            ls_low, mk_low = style[(label, "low")]
            ls_high, mk_high = style[(label, "high")]
            if not low.empty:
                ax.plot(low.index, low.values, linestyle=ls_low, marker=mk_low,
                        linewidth=2.5, markersize=8, label=f"{label} (c≤10)")
            if not high.empty:
                ax.plot(high.index, high.values, linestyle=ls_high, marker=mk_high,
                        linewidth=2.5, markersize=8, label=f"{label} (c=50)")
        ax.set_xlabel("Noise Fraction")
        ax.set_ylabel(metric.upper() + " Score")
        ax.set_title(f"{metric.upper()}: centers≤10 vs centers=50")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
    fig.suptitle("SDOclust Accuracy Degradation: Impact of High Cluster Count",
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

# Plot 6: Noise robustness comparison across all algorithms
def plot_full_robustness_comparison(df, outpath):
    methods_map = {
        "parallel_sdoclust": "SDO (Parallel)",
        "sdoclust_full":     "SDO (Full)",
        "minibatch_seq":     "MB-KMeans (Seq)",
        "kmeans_seq":        "KMeans (Seq)",
        "daskml_kmeans":     "KMeans (DaskML)",
    }

    style_map = {
        "SDO (Full)":         dict(linestyle="-",  marker="o", linewidth=3, markersize=10, zorder=5),
        "SDO (Parallel)":     dict(linestyle="--", marker="D", linewidth=3, markersize=10, zorder=5),
        "MB-KMeans": dict(linestyle="-",  marker="s", linewidth=2, markersize=7),
        "KMeans (Seq)":       dict(linestyle="-",  marker="^", linewidth=2, markersize=7),
        "KMeans (DaskML)":    dict(linestyle="-",  marker="v", linewidth=2, markersize=7),
    }

    sub = df[df["dataset"] == "noisy_blobs"].copy()
    sub = sub[sub["centers"] <= 10]
    sub = sub[sub["method"].isin(methods_map.keys())]
    sub["method_label"] = sub["method"].map(methods_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    agg_ami = sub.groupby(["method_label", "noise_frac"], as_index=False)["ami"].mean()
    for label, data in agg_ami.groupby("method_label"):
        kw = style_map.get(label, dict(linestyle="-", marker="o", linewidth=2))
        ax1.plot(data["noise_frac"], data["ami"], label=label, **kw)

    ax1.set_title("Robustness: AMI Score by Noise Fraction")
    ax1.set_xlabel("Noise Fraction")
    ax1.set_ylabel("AMI Score")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(title="Algorithms")

    if "ari" in sub.columns:
        agg_ari = sub.groupby(["method_label", "noise_frac"], as_index=False)["ari"].mean()
        for label, data in agg_ari.groupby("method_label"):
            kw = style_map.get(label, dict(linestyle="-", marker="s", linewidth=2))
            ax2.plot(data["noise_frac"], data["ari"], label=label, **kw)

        ax2.set_title("Robustness: ARI Score by Noise Fraction")
        ax2.set_xlabel("Noise Fraction")
        ax2.set_ylabel("ARI Score")
        ax2.set_ylim(0.4, 1.05)
        ax2.legend(title="Algorithms")
    else:
        ax2.text(0.5, 0.5, "ARI Data Not Available", ha='center', va='center')

    fig.suptitle("Clustering Robustness Analysis: AMI vs ARI", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# Plot 7: Accuracy consistency
def plot_accuracy_consistency_comparison(df, outpath):
    target_datasets = ["blobs", "noisy_blobs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    axes = {"blobs": ax1, "noisy_blobs": ax2}

    for ds_name in target_datasets:
        ax = axes[ds_name]
        ds_sub = df[(df["dataset"] == ds_name) & (df["centers"] <= 10)].copy()

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
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle("Clustering Accuracy Comparison: Full vs Parallel (by Splits)",
                 fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# Plot 8: Algorithm time series (N<=200k)
def plot_algorithm_time_series(df, outpath, x_axis="cores"):
    methods_map = {
        "sdoclust_full":     "SDO (Full)",
        "parallel_sdoclust": "SDO (Parallel)",
        "kmeans_seq":        "KMeans (Seq)",
        "minibatch_seq":     "MiniBatch KM (Seq)",
        "daskml_kmeans":     "KMeans (DaskML)",
    }

    backend_style_map = {
        "dask":   dict(linestyle="-", marker="o", linewidth=2.5, markersize=8),
        "joblib": dict(linestyle="--",  marker="s", linewidth=2.5, markersize=8),
        "seq":    dict(linestyle="-",  marker="^", linewidth=2.5, markersize=8),
    }

    sub = df[df["method"].isin(methods_map.keys())].copy()
    if "N" in sub.columns:
        sub = sub[sub["N"] != 1000000]
    sub = sub[sub["centers"] <= 10]
    sub["method_label"] = sub["method"].map(methods_map)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for label, data in sub.groupby("method_label"):
        agg = data.groupby(x_axis)["total"].mean().sort_index()
        ax1.plot(agg.index, agg.values, marker='o', label=label, linewidth=2)

    ax1.set_title("Execution Time Comparison by Algorithm (N≤200k)")
    ax1.set_xlabel(x_axis.capitalize())
    ax1.set_ylabel("Total Time (sec)")
    ax1.legend(title="Algorithms")

    if "backend" in df.columns:
        para_sub = df[df["method"] == "parallel_sdoclust"].copy()
        if "N" in para_sub.columns:
            para_sub = para_sub[para_sub["N"] != 1000000]
        para_sub = para_sub[para_sub["centers"] <= 10]
        para_sub = para_sub[para_sub["backend"].isin(["seq", "joblib", "dask"])]

        for backend, data in para_sub.groupby("backend"):
            agg = data.groupby(x_axis)["total"].mean().sort_index()
            kw = backend_style_map.get(backend, dict(linestyle="-", marker="s", linewidth=2))
            ax2.plot(agg.index, agg.values, label=f"Backend: {backend}", **kw)

        ax2.set_title("Scalability: Parallel SDOclust Backend Comparison (N≤200k)")
        ax2.set_xlabel(x_axis.capitalize())
        ax2.set_ylabel("Total Time (sec)")
        ax2.legend(title="Backends")
    else:
        ax2.text(0.5, 0.5, "Backend column not found", ha='center', va='center')

    fig.suptitle("Performance and Backend Scalability Analysis", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")


# Plot 9: Large-scale (N=1M) analysis, CORES=1,2,4 only
def plot_large_scale_analysis(df, outpath):
    sub = df[df["N"] == 1000000].copy()
    sub = sub[sub["centers"] <= 10]
    if sub.empty:
        print("  [large_scale] No N=1M data – skipping.")
        return

    methods_map = {
        "sdoclust_full":     "SDO (Full)",
        "parallel_sdoclust": "SDO (Parallel)",
        "kmeans_seq":        "KMeans (Seq)",
        "minibatch_seq":     "MiniBatch KM (Seq)",
        "daskml_kmeans":     "KMeans (DaskML)",
    }

    backend_style_map = {
        "dask":   dict(linestyle="--", marker="o", linewidth=2.5, markersize=8),
        "joblib": dict(linestyle="-",  marker="s", linewidth=2.5, markersize=8),
        "seq":    dict(linestyle="-",  marker="^", linewidth=2.5, markersize=8),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: algorithm comparison
    algo_sub = sub[sub["method"].isin(methods_map.keys())].copy()
    algo_sub["method_label"] = algo_sub["method"].map(methods_map)
    for label, data in algo_sub.groupby("method_label"):
        agg = data.groupby("cores")["total"].mean().sort_index()
        ax1.plot(agg.index, agg.values, marker='o', label=label, linewidth=2)

    ax1.set_title("Execution Time by Algorithm (N=1M)")
    ax1.set_xlabel("Cores")
    ax1.set_ylabel("Total Time (sec)")
    ax1.set_xticks([1, 2, 4])
    ax1.legend(title="Algorithms")

    # Right: backend comparison for parallel_sdoclust
    para_sub = sub[sub["method"] == "parallel_sdoclust"].copy()
    para_sub = para_sub[para_sub["backend"].isin(["seq", "joblib", "dask"])]
    for backend, data in para_sub.groupby("backend"):
        agg = data.groupby("cores")["total"].mean().sort_index()
        kw = backend_style_map.get(backend, dict(linestyle="-", marker="s", linewidth=2))
        ax2.plot(agg.index, agg.values, label=f"Backend: {backend}", **kw)

    ax2.set_title("Backend Comparison: Parallel SDOclust (N=1M)")
    ax2.set_xlabel("Cores")
    ax2.set_ylabel("Total Time (sec)")
    ax2.set_xticks([1, 2, 4])
    ax2.legend(title="Backends")

    fig.suptitle("Large-Scale Performance Analysis: N=1,000,000", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")

    
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all_results(RESULTS_DIR)
    df = standardize(df)

    print("Generating figures...")

    plot_parallel_performance(df, FIG_DIR / "parallel_scalability.png")
    plot_amdahl_analysis(df, FIG_DIR / "amdahl_analysis.png")
    plot_efficiency_drop(df, FIG_DIR / "efficiency_drop_heatmap.png")
    plot_full_robustness_comparison(df, FIG_DIR / "algorithm_robustness.png")
    plot_centers_analysis(df, FIG_DIR / "centers_complexity_analysis.png")
    plot_centers_accuracy_comparison(df, FIG_DIR / "centers_accuracy_comparison.png")
    plot_accuracy_consistency_comparison(df, FIG_DIR / "accuracy_consistency_analysis.png")
    plot_algorithm_time_series(df, FIG_DIR / "algorithm_speed_comparison.png")
    plot_large_scale_analysis(df, FIG_DIR / "large_scale_analysis.png")

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()