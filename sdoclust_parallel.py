import time
import numpy as np
import scipy.spatial.distance as distance

from joblib import Parallel, delayed
import dask
from dask import delayed as ddelayed, compute as dcompute
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import sdoclust as sdo
import argparse


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

    # Observers
    O_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == d:
            O_candidates.append((name, val))
    O_name, O = max(O_candidates, key=lambda t: t[1].shape[0])

    n_obs = O.shape[0]

    # Labels
    l_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == n_obs:
            if np.issubdtype(val.dtype, np.integer):
                l_candidates.append((name, val))

    # Automatically identify the cluster label
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


# Predict labels for one chunk via kNN voting against observers
def extend_labels_chunk(X_chunk, O, l, knn=10):
    valid = (l >= 0)
    O2 = O[valid]
    l2 = l[valid]
    if O2.shape[0] == 0:
        return -np.ones(X_chunk.shape[0], dtype=int)

    labs = np.unique(l2)
    lab_to_idx = {lab: i for i, lab in enumerate(labs)}
    l_idx = np.fromiter((lab_to_idx[v] for v in l2), dtype=int, count=len(l2))

    dist = distance.cdist(X_chunk, O2, metric="euclidean")
    k = min(knn, O2.shape[0])
    closest = np.argpartition(dist, k - 1, axis=1)[:, :k]
    lknn = l_idx[closest]

    cl = len(labs)
    C = np.zeros((X_chunk.shape[0], cl), dtype=int)
    for j in range(cl):
        C[:, j] = np.sum(lknn == j, axis=1)

    y_idx = np.argmax(C, axis=1)
    return labs[y_idx]

# Extend cluster labels against the learned observers
def extend_on_split(X, split_idx, O, l, knn=10, chunksize=2000):
    Xi = X[split_idx]
    out = np.empty(len(Xi), dtype=int)

    for start in range(0, len(Xi), chunksize):
        out[start:start + chunksize] = extend_labels_chunk(
            Xi[start:start + chunksize], O, l, knn=knn
        )
    return split_idx, out

# Run label extension across splits
def run_extend_backend(
    X, splits, O, l,
    backend="seq",
    knn=10, chunksize=2000,
    n_jobs=-1,
):
    if backend == "seq":
        return [extend_on_split(X, sp, O, l, knn=knn, chunksize=chunksize) for sp in splits]

    if backend == "joblib":
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(extend_on_split)(X, sp, O, l, knn=knn, chunksize=chunksize) for sp in splits
        )

    if backend == "dask":
        tasks = [ddelayed(extend_on_split)(X, sp, O, l, knn=knn, chunksize=chunksize) for sp in splits]
        results = dcompute(*tasks)
        return list(results)

# Merge results
def merge_results(y_pred, results):
    for idx_sp, y_sp in results:
        y_pred[idx_sp] = y_sp
    return y_pred


# Experiment with n_splits
def experiment_fixed_nsplits(X, y_true, n_splits_list=(2, 4, 8, 16),  splitA_pos=0, knn=10, chunksize=2000, seed=0, backends=("seq","joblib","dask"), n_jobs=-1,):
    N, d = X.shape

    for backend in backends:
        print("\n" + "=" * 70)
        print(f"[backend={backend}] knn={knn} chunksize={chunksize}")
        print(f"{'backend':>7s} {'splitA':>6s} {'|A|':>6s} {'fit':>7s} {'ext':>7s} "
              f"{'total':>7s} {'n_obs':>6s} {'ARI':>6s} {'AMI':>6s}")

        # printed_attr = False
        for n_splits in n_splits_list:
            # Simlar size splits
            all_idx = np.arange(N)
            splits = make_splits(all_idx, n_splits=n_splits, seed=seed, shuffle=True)

            # Select split-A
            splitA_pos_eff = int(splitA_pos) % n_splits
            splitA_idx = splits[splitA_pos_eff]
            rest_splits = [sp for i, sp in enumerate(splits) if i != splitA_pos_eff]

            # SDOclust fit on split-A
            t0 = time.time()
            model = sdo.SDOclust().fit(X[splitA_idx])
            fit_time = time.time() - t0

            (O_name, O), (l_name, l) = extract_observers_and_labels(model, d=d)
            n_obs = len(O)
            
            # DEBUG
            # if not printed_attr:
            #     print(f"Using observers attr='{O_name}', labels attr='{l_name}'")
            #     printed_attr = True

            # Extend rest splits (parallel)
            t1 = time.time()
            y_pred = -np.ones(N, dtype=int)

            # Extend split-A (for evaluation)
            idxA, yA = extend_on_split(X, splitA_idx, O, l, knn=knn, chunksize=chunksize)
            y_pred[idxA] = yA

            # rest splits extend (backend)
            if len(rest_splits) > 0:
                results = run_extend_backend(
                    X, rest_splits, O, l,
                    backend=backend, knn=knn, chunksize=chunksize,
                    n_jobs=n_jobs,
                )

                y_pred = merge_results(y_pred, results)

            ext_time = time.time() - t1
            total_time = fit_time + ext_time

            mask = (y_true != -1)
            ari = adjusted_rand_score(y_true[mask], y_pred[mask])
            ami = adjusted_mutual_info_score(y_true[mask], y_pred[mask])

            print(f"{backend:>7s} {n_splits:6d} {len(splitA_idx):6d} {fit_time:7.3f} {ext_time:7.3f} {total_time:7.3f} {n_obs:6d} {ari:6.3f} {ami:6.3f}")


# Run baseline sdo clust
def baseline_sdoclust(X, y_true):
    t0 = time.time()
    y_pred = sdo.SDOclust().fit_predict(X)
    te = time.time() - t0
    mask = (y_true != -1)
    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    ami = adjusted_mutual_info_score(y_true[mask], y_pred[mask])
    print("\n" + "-" * 70)
    print(f"Baseline SDOclust Time={te:.3f}s  ARI={ari:.3f}  AMI={ami:.3f}")


# Build Dataset
def build_dataset(name, N, d, centers, std, seed, noise_frac=0.0):
    if name in ("blobs", "noisy_blobs"):
        # blobs base
        X, y = make_blobs(
            n_samples=N,
            centers=centers,
            n_features=d,
            cluster_std=std,
            random_state=seed,
        )

        # noise injection
        if name == "noisy_blobs":
            nf = noise_frac if noise_frac > 0 else 0.05
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

# Loop over datasets/sizes/std/seeds and run baseline
def run_suite(
    datasets,
    N_list,
    seeds,
    backends,
    n_splits_list,
    *,
    d_list=(10, 50),
    centers=5,
    std_list=(1.0, 2.0),
    noise_frac=0.05,
    splitA_pos=0,
    knn=10,
    chunksize=2000,
    n_jobs=-1,
):
    for ds in datasets:
        for N in N_list:
            for d in d_list:
                for std in std_list:
                    print("\n" + "#" * 70)
                    noise_str = noise_frac if ds == "noisy_blobs" else 0.0
                    print(f"DATASET={ds}  N={N}  d={d}  centers={centers}  std={std}  noise_frac={noise_str}")
                    print("#" * 70)

                    for seed in seeds:
                        print(f"[SEED={seed}]")

                        X, y_true = build_dataset(
                            ds, N=N, d=d, centers=centers, std=std, seed=seed, noise_frac=noise_frac
                        )

                        # baseline: full SDOclust on whole X
                        baseline_sdoclust(X, y_true)

                        # fixed n_splits experiment
                        experiment_fixed_nsplits(
                            X, y_true,
                            n_splits_list=tuple(n_splits_list),
                            splitA_pos=splitA_pos,
                            knn=knn,
                            chunksize=chunksize,
                            seed=seed,
                            backends=tuple(backends),
                            n_jobs=n_jobs,
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Select dataset
    parser.add_argument("--datasets", type=str, default="blobs,noisy_blobs", help="Dataset names")

    # Data size
    parser.add_argument("--size", type=str, default="200000", help="Sample sizes")
    
    # Backends
    parser.add_argument("--backends", type=str, default="seq,joblib,dask", help="Execution backends")
    
    # Number of Splits
    parser.add_argument("--splits", type=str, default="2,4,8,16", help="Number of splits")

    # Replicability
    parser.add_argument("--seeds", type=str, default="42", help="Random seeds")

    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",")]
    N_list = [int(x) for x in args.size.split(",")]
    backends = [x.strip() for x in args.backends.split(",")]
    n_splits_list = [int(x) for x in args.splits.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    run_suite(
        datasets=datasets,
        N_list=N_list,
        seeds=seeds,
        backends=backends,
        n_splits_list=n_splits_list,
        d_list=(10, 50),
        centers=5,
        std_list=(1.0, 2.0),
        noise_frac=0.05,
        splitA_pos=0,
        knn=10,
        chunksize=2000,
        n_jobs=-1,
    )
