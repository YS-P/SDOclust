import time
import numpy as np

from joblib import Parallel, delayed
from dask import delayed as ddelayed, compute as dcompute

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from scipy.spatial.distance import cdist

import sdoclust as sdo
import argparse

import os
import csv
import socket

from parallel_kmeans import (
    kmeans_seq,
    minibatch_kmeans_seq,
    minibatch_kmeans_joblib,
    minibatch_kmeans_dask,
    evaluate_model,
)

# -----------------------------
# CSV schema (single source of truth)
# -----------------------------
FIELDNAMES = [
    "phase", "method",
    "dataset", "N", "d", "centers", "std", "noise_frac", "seed",
    "backend", "n_splits", "splitA_pos", "splitA_size",
    "chunksize", "knn_eff",
    "fit_time", "ext_time", "total_time",
    "n_obs", "ari", "ami",
    # HPC / SLURM metadata (optional)
    "slurm_job_id", "slurm_cpus_per_task", "slurm_node_list", "hostname",
]

# -----------------------------
# Utils
# -----------------------------
def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def get_hpc_meta():
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_node_list": os.environ.get("SLURM_NODELIST", ""),
        "hostname": socket.gethostname(),
    }

def append_result_csv(output_path, row_dict):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_header = (not os.path.exists(output_path)) or (os.path.getsize(output_path) == 0)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        # Ensure all required keys exist
        full_row = {k: row_dict.get(k, "") for k in FIELDNAMES}
        writer.writerow(full_row)

# Split into roughly equal parts
def make_splits(indices, n_splits, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    idx = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(idx)
    return np.array_split(idx, n_splits)

# Extract observers + labels
def extract_observers_and_labels(model, d):
    attrs = vars(model)

    # Observers: any 2D array with shape[1]==d, take the largest by rows
    O_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == d:
            O_candidates.append((name, val))
    if not O_candidates:
        raise ValueError("Could not find observer matrix in model attributes.")
    O_name, O = max(O_candidates, key=lambda t: t[1].shape[0])

    n_obs = O.shape[0]

    # Labels: any 1D int array with length n_obs
    l_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == n_obs:
            if np.issubdtype(val.dtype, np.integer):
                l_candidates.append((name, val))
    if not l_candidates:
        raise ValueError("Could not find observer label array in model attributes.")

    def score_label_name(nm):
        nm2 = nm.lower()
        score = 0
        if "label" in nm2:
            score += 3
        if "cluster" in nm2:
            score += 2
        if "assign" in nm2:
            score += 1
        return score

    l_name, l = max(l_candidates, key=lambda t: score_label_name(t[0]))
    return (O_name, O), (l_name, l)

# Predict labels by querying k nearest observers and doing majority vote over their labels
def extend_labels_chunk(X_chunk, O2, labs, l_idx, knn):
    k = min(int(knn), O2.shape[0])
    if k <= 0:
        return -np.ones(X_chunk.shape[0], dtype=int)

    # (n_chunk, n_obs)
    D = cdist(X_chunk, O2, metric="euclidean")

    # top-k (unordered)
    closest = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    lknn = l_idx[closest]  # (n_chunk, k) in [0..K-1]

    K = len(labs)
    C = np.zeros((X_chunk.shape[0], K), dtype=np.int32)
    rows = np.arange(X_chunk.shape[0])[:, None]
    np.add.at(C, (rows, lknn), 1)

    y_idx = np.argmax(C, axis=1)
    return labs[y_idx]

# Extend cluster labels against the learned observers
def extend_on_split(X, split_idx, O, l, knn, chunksize=2000):
    Xi = X[split_idx]
    out = np.empty(len(Xi), dtype=int)

    valid = (l >= 0)
    O2 = O[valid]
    l2 = l[valid]
    if O2.shape[0] == 0:
        return split_idx, -np.ones(len(Xi), dtype=int)

    labs = np.unique(l2)
    lab_to_idx = {lab: i for i, lab in enumerate(labs)}
    l_idx = np.fromiter((lab_to_idx[v] for v in l2), dtype=int, count=len(l2))

    for start in range(0, len(Xi), chunksize):
        out[start:start + chunksize] = extend_labels_chunk(
            Xi[start:start + chunksize],
            O2,
            labs,
            l_idx,
            knn=knn,
        )

    return split_idx, out

# Run label extension across splits
def run_extend_backend(
    X, splits, O, l,
    backend="seq",
    knn=None, chunksize=2000,
    n_jobs=-1,
):
    if knn is None:
        raise ValueError("knn must be provided (use model.xc or model.x)")

    if backend == "seq":
        return [extend_on_split(X, sp, O, l, knn=knn, chunksize=chunksize) for sp in splits]

    if backend == "joblib":
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(extend_on_split)(X, sp, O, l, knn=knn, chunksize=chunksize)
            for sp in splits
        )

    if backend == "dask":
        tasks = [
            ddelayed(extend_on_split)(X, sp, O, l, knn=knn, chunksize=chunksize)
            for sp in splits
        ]
        results = dcompute(*tasks)
        return list(results)

    raise ValueError(f"Unknown backend: {backend}")

# Merge results
def merge_results(y_pred, results):
    for idx_sp, y_sp in results:
        y_pred[idx_sp] = y_sp
    return y_pred

# Experiment with n_splits
def experiment_fixed_nsplits(
    X, y_true,
    n_splits_list=(2, 4, 8, 16),
    splitA_pos=0,
    knn=10,
    chunksize=2000,
    seed=0,
    backends=("seq", "joblib", "dask"),
    n_jobs=-1,
    output_path=None,
    dataset_name=None,
    N=None, d=None, centers=None, std=None, noise_frac=None,
):
    N_X, d_X = X.shape

    for backend in backends:
        print("\n" + "=" * 70)
        print(f"[backend={backend}] chunksize={chunksize}")
        print(f"{'backend':>7s} {'splitA':>6s} {'|A|':>6s} {'fit':>7s} {'ext':>7s} "
              f"{'total':>7s} {'n_obs':>6s} {'ARI':>6s} {'AMI':>6s}")

        for n_splits in n_splits_list:
            all_idx = np.arange(N_X)
            splits = make_splits(all_idx, n_splits=n_splits, seed=seed, shuffle=True)

            splitA_pos_eff = int(splitA_pos) % n_splits
            splitA_idx = splits[splitA_pos_eff]
            rest_splits = [sp for i, sp in enumerate(splits) if i != splitA_pos_eff]

            # Fit SDOclust on split-A
            t0 = time.time()
            model = sdo.SDOclust().fit(X[splitA_idx])
            fit_time = time.time() - t0

            (_, O), (_, l) = extract_observers_and_labels(model, d=d_X)
            n_obs = len(O)

            # Use SDOclust's own x/xc for label extension if present
            if hasattr(model, "xc") and model.xc is not None:
                knn_eff = int(model.xc)
            elif hasattr(model, "x") and model.x is not None:
                knn_eff = int(model.x)
            else:
                knn_eff = int(knn)

            # Extend all splits
            t1 = time.time()
            y_pred = -np.ones(N_X, dtype=int)

            idxA, yA = extend_on_split(X, splitA_idx, O, l, knn=knn_eff, chunksize=chunksize)
            y_pred[idxA] = yA

            if len(rest_splits) > 0:
                results = run_extend_backend(
                    X, rest_splits, O, l,
                    backend=backend, knn=knn_eff, chunksize=chunksize,
                    n_jobs=n_jobs,
                )
                y_pred = merge_results(y_pred, results)

            ext_time = time.time() - t1
            total_time = fit_time + ext_time

            ari = adjusted_rand_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)

            print(f"{backend:>7s} {n_splits:6d} {len(splitA_idx):6d} "
                  f"{fit_time:7.3f} {ext_time:7.3f} {total_time:7.3f} "
                  f"{n_obs:6d} {ari:6.3f} {ami:6.3f}")

            if output_path is not None:
                row = {
                    "phase": "experiment",
                    "method": "wrapper_sdoclust",
                    "dataset": dataset_name,
                    "N": int(N if N is not None else N_X),
                    "d": int(d if d is not None else d_X),
                    "centers": int(centers) if centers is not None else "",
                    "std": float(std) if std is not None else "",
                    "noise_frac": float(noise_frac) if noise_frac is not None else "",
                    "seed": int(seed),

                    "backend": backend,
                    "n_splits": int(n_splits),
                    "splitA_pos": int(splitA_pos),
                    "splitA_size": int(len(splitA_idx)),
                    "chunksize": int(chunksize),
                    "knn_eff": int(knn_eff),

                    "fit_time": float(fit_time),
                    "ext_time": float(ext_time),
                    "total_time": float(total_time),

                    "n_obs": int(n_obs),
                    "ari": float(ari),
                    "ami": float(ami),
                }
                row.update(get_hpc_meta())
                append_result_csv(output_path, row)

# Baseline: full SDOclust
def baseline_sdoclust(X, y_true):
    t0 = time.time()
    y_pred = sdo.SDOclust().fit_predict(X)
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print("\n" + "-" * 70)
    print(f"Baseline SDOclust Time={te:.3f}s  ARI={ari:.3f}  AMI={ami:.3f}")
    return te, ari, ami

# Baseline: KMeans 
def get_kmeans_pred(model, X):
    if hasattr(model, "labels_"):
        return model.labels_
    if hasattr(model, "predict"):
        return model.predict(X)
    raise ValueError("Model has neither labels_ nor predict().")

def baseline_kmean_parallel(X, y_true, n_clusters):
    results = []
    print("Baseline parallel kmeans")

    t0 = time.time()
    model_seq = kmeans_seq(X, n_clusters=n_clusters)
    y_pred = get_kmeans_pred(model_seq, X)
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    evaluate_model(model_seq, X, y_true, "KMeans Sequential", t0)
    results.append(("kmeans_seq", te, ari, ami))

    t0 = time.time()
    model_mb_seq = minibatch_kmeans_seq(X, n_clusters=n_clusters, batch_size=2000)
    y_pred = get_kmeans_pred(model_mb_seq, X)
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    evaluate_model(model_mb_seq, X, y_true, "MiniBatchKMeans Seq", t0)
    results.append(("minibatch_seq", te, ari, ami))

    t0 = time.time()
    model_joblib = minibatch_kmeans_joblib(X, n_clusters=n_clusters, batch_size=2000, n_jobs=-1)
    y_pred = get_kmeans_pred(model_joblib, X)
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    evaluate_model(model_joblib, X, y_true, "MiniBatchKMeans Joblib", t0)
    results.append(("minibatch_joblib", te, ari, ami))

    t0 = time.time()
    model_dask = minibatch_kmeans_dask(X, n_clusters=n_clusters, batch_size=2000)
    y_pred = get_kmeans_pred(model_dask, X)
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    evaluate_model(model_dask, X, y_true, "MiniBatchKMeans Dask", t0)
    results.append(("minibatch_dask", te, ari, ami))

    return results

# Build Dataset
def build_dataset(name, N, d, centers, std, seed, noise_frac=0.0):
    if name in ("blobs", "noisy_blobs"):
        X, y = make_blobs(
            n_samples=N,
            centers=centers,
            n_features=d,
            cluster_std=std,
            random_state=seed,
        )

        # noise/outliers injection
        # uniform box outliers labeled as -1
        if name == "noisy_blobs":
            nf = max(0.0, float(noise_frac))
            rng = np.random.default_rng(seed)
            n_noise = int(N * nf)
            if n_noise > 0:
                mins = X.min(axis=0)
                maxs = X.max(axis=0)
                Xn = rng.uniform(mins, maxs, size=(n_noise, d))
                yn = -np.ones(n_noise, dtype=int)
                X = np.vstack([X, Xn])
                y = np.hstack([y, yn])

        X = StandardScaler().fit_transform(X)
        return X, y

    raise ValueError(f"Unknown dataset name: {name}")

# Main suite
def run_suite(
    datasets,
    N_list,
    seeds,
    backends,
    n_splits_list,
    *,
    d_list=(10, 50),
    centers_list=(5,),
    std_list=(1.0, 2.0),
    noise_list=(0.05,),
    splitA_pos=0,
    knn=10,
    chunksize=2000,
    n_jobs=-1,
    output_path=None,
):
    for ds in datasets:
        for centers in centers_list:
            for N in N_list:
                for d in d_list:
                    for std in std_list:
                        for noise_frac in noise_list:
                            noise_used = float(noise_frac) if ds == "noisy_blobs" else 0.0

                            print("\n" + "#" * 70)
                            print(f"DATASET={ds}  N={N}  d={d}  centers={centers}  std={std}  noise_frac={noise_used}")
                            print("#" * 70)

                            for seed in seeds:
                                print(f"[SEED={seed}]")

                                X, y_true = build_dataset(
                                    ds,
                                    N=N,
                                    d=d,
                                    centers=centers,
                                    std=std,
                                    seed=seed,
                                    noise_frac=noise_used,
                                )

                                # -------- baseline: full SDOclust --------
                                te, ari, ami = baseline_sdoclust(X, y_true)
                                if output_path is not None:
                                    row = {
                                        "phase": "baseline",
                                        "method": "sdoclust_full",
                                        "dataset": ds,
                                        "N": int(N),
                                        "d": int(d),
                                        "centers": int(centers),
                                        "std": float(std),
                                        "noise_frac": float(noise_used),
                                        "seed": int(seed),

                                        "backend": "na",
                                        "n_splits": -1,
                                        "splitA_pos": -1,
                                        "splitA_size": -1,
                                        "chunksize": int(chunksize),
                                        "knn_eff": -1,

                                        "fit_time": float(te),
                                        "ext_time": 0.0,
                                        "total_time": float(te),

                                        "n_obs": -1,
                                        "ari": float(ari),
                                        "ami": float(ami),
                                    }
                                    row.update(get_hpc_meta())
                                    append_result_csv(output_path, row)

                                # baseline
                                kmeans_results = baseline_kmean_parallel(X, y_true, n_clusters=centers)
                                if output_path is not None:
                                    for (mname, te, ari, ami) in kmeans_results:
                                        row = {
                                            "phase": "baseline",
                                            "method": mname,
                                            "dataset": ds,
                                            "N": int(N),
                                            "d": int(d),
                                            "centers": int(centers),
                                            "std": float(std),
                                            "noise_frac": float(noise_used),
                                            "seed": int(seed),

                                            "backend": "na",
                                            "n_splits": -1,
                                            "splitA_pos": -1,
                                            "splitA_size": -1,
                                            "chunksize": int(chunksize),
                                            "knn_eff": -1,

                                            "fit_time": float(te),
                                            "ext_time": 0.0,
                                            "total_time": float(te),

                                            "n_obs": -1,
                                            "ari": float(ari),
                                            "ami": float(ami),
                                        }
                                        row.update(get_hpc_meta())
                                        append_result_csv(output_path, row)

                                # Wrapper SDOclust
                                experiment_fixed_nsplits(
                                    X, y_true,
                                    n_splits_list=tuple(n_splits_list),
                                    splitA_pos=splitA_pos,
                                    knn=knn,
                                    chunksize=chunksize,
                                    seed=seed,
                                    backends=tuple(backends),
                                    n_jobs=n_jobs,
                                    output_path=output_path,
                                    dataset_name=ds,
                                    N=N, d=d, centers=centers, std=std,
                                    noise_frac=noise_used,
                                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str, default="blobs,noisy_blobs",
                        help="Dataset names")
    parser.add_argument("--size", type=str, default="200000",
                        help="Sample sizes (e.g. 50000,200000)")
    parser.add_argument("--backends", type=str, default="seq,joblib,dask",
                        help="Execution backends (seq,joblib,dask)")
    parser.add_argument("--splits", type=str, default="2,4,8,16",
                        help="Number of splits")
    parser.add_argument("--noise", type=str, default="0.05",
                        help="Noise fractions(e.g. 0,0.05,0.1)")
    parser.add_argument("--centers", type=str, default="5",
                        help="Ground-truth cluster counts (e.g. 3,5,10)")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Random seeds")
    parser.add_argument("--output", type=str, default="results/results.csv",
                        help="Path to CSV file (appends results)")

    parser.add_argument("--chunksize", type=int, default=2000, help="Chunk size for cdist in label extension")
    parser.add_argument("--knn", type=int, default=10, help="Fallback k for label extension if model has no x/xc")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Joblib n_jobs for backend=joblib")

    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip() != ""]
    N_list = parse_int_list(args.size)
    backends = [x.strip() for x in args.backends.split(",") if x.strip() != ""]
    n_splits_list = parse_int_list(args.splits)
    noise_list = parse_float_list(args.noise)
    centers_list = parse_int_list(args.centers)
    seeds = parse_int_list(args.seeds)

    run_suite(
        datasets=datasets,
        N_list=N_list,
        seeds=seeds,
        backends=backends,
        n_splits_list=n_splits_list,
        d_list=(10, 50),
        centers_list=centers_list,
        std_list=(1.0, 2.0),
        noise_list=noise_list,
        splitA_pos=0,
        knn=args.knn,
        chunksize=args.chunksize,
        n_jobs=args.n_jobs,
        output_path=args.output,
    )
