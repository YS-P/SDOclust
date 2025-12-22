import time
import numpy as np
import scipy.spatial.distance as distance

from joblib import Parallel, delayed
import dask
from dask import delayed as ddelayed, compute as dcompute

try:
    from dask.distributed import Client, LocalCluster
    HAS_DASK_DISTRIBUTED = True
except Exception:
    HAS_DASK_DISTRIBUTED = False
dask.config.set({"distributed.worker.daemon": False})


from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import sdoclust as sdo


def make_splits(indices, n_splits, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    idx = np.array(indices, copy=True)
    if shuffle:
        rng.shuffle(idx)
    return np.array_split(idx, n_splits)


# Extract observers + labels
def extract_observers_and_labels(model, d):
    attrs = vars(model)

    O_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == d:
            O_candidates.append((name, val))
    O_name, O = max(O_candidates, key=lambda t: t[1].shape[0])

    n_obs = O.shape[0]

    l_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == n_obs:
            if np.issubdtype(val.dtype, np.integer):
                l_candidates.append((name, val))

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


# extend label for one chunk
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


def extend_on_split(X, split_idx, O, l, knn=10, chunksize=2000):
    Xi = X[split_idx]
    out = np.empty(len(Xi), dtype=int)

    for start in range(0, len(Xi), chunksize):
        out[start:start + chunksize] = extend_labels_chunk(
            Xi[start:start + chunksize], O, l, knn=knn
        )
    return split_idx, out


def run_extend_backend(
    X, splits, O, l,
    backend="seq",
    knn=10, chunksize=2000,
    n_jobs=-1,
    client=None,
):
    if backend == "seq":
        return [extend_on_split(X, sp, O, l, knn=knn, chunksize=chunksize) for sp in splits]

    if backend == "joblib":
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(extend_on_split)(X, sp, O, l, knn=knn, chunksize=chunksize) for sp in splits
        )

    if backend == "dask-local":
        tasks = [ddelayed(extend_on_split)(X, sp, O, l, knn=knn, chunksize=chunksize)
                 for sp in splits]
        results = dcompute(*tasks)
        return list(results)

    if backend == "dask-dist":
        if client is None:
            raise ValueError("backend='dask-dist' requires a dask.distributed Client.")
        futs = [client.submit(extend_on_split, X, sp, O, l, knn, chunksize)
                for sp in splits]
        return client.gather(futs)



def merge_results(y_pred, results):
    for idx_sp, y_sp in results:
        y_pred[idx_sp] = y_sp
    return y_pred


# Experiment with fixed n_splits
def experiment_fixed_nsplits(X, y_true, n_splits_list=(2, 4, 8, 16),  splitA_pos=0, knn=10, chunksize=2000, seed=0, backends=("seq","joblib","dask-local","dask-dist"), n_jobs=-1, client=None,):
    N, d = X.shape

    for backend in backends:
        print("\n" + "=" * 70)
        print(f"[FIXED n_splits] [backend={backend}] knn={knn} chunksize={chunksize}")
        print(f"{'backend':>7s} {'splitA':>6s} {'|A|':>6s} {'fit':>7s} {'ext':>7s} "
              f"{'total':>7s} {'n_obs':>6s} {'ARI':>6s} {'AMI':>6s}")

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

            (_, O), (_, l) = extract_observers_and_labels(model, d=d)
            n_obs = len(O)

            # Extend rest splits (parallel)
            t1 = time.time()
            y_pred = np.empty(N, dtype=int)

            # Extend split-A (for evaluation)
            idxA, yA = extend_on_split(X, splitA_idx, O, l, knn=knn, chunksize=chunksize)
            y_pred[idxA] = yA

            # rest splits extend (backend)
            if len(rest_splits) > 0:
                results = run_extend_backend(
                    X, rest_splits, O, l,
                    backend=backend, knn=knn, chunksize=chunksize,
                    n_jobs=n_jobs, client=client
                )

                y_pred = merge_results(y_pred, results)

            ext_time = time.time() - t1
            total_time = fit_time + ext_time

            ari = adjusted_rand_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)

            print(f"{backend:>7s} {n_splits:6d} {len(splitA_idx):6d} {fit_time:7.3f} {ext_time:7.3f} {total_time:7.3f} {n_obs:6d} {ari:6.3f} {ami:6.3f}")



# Experiment split-A fraction sweep
def experiment_splitA_rest(
    X, y_true,
    splitA_fracs=(0.02, 0.05, 0.10, 0.20, 0.40, 1.00),
    n_rest_splits=8,
    knn=10,
    chunksize=2000,
    seed=0,
    backends=("seq","joblib","dask-local","dask-dist"), 
    n_jobs=-1,
    client=None,
):
    rng = np.random.default_rng(seed)
    N, d = X.shape

    for backend in backends:
        print("\n" + "=" * 70)
        print(f"[splitA-frac] [backend={backend}] knn={knn} chunksize={chunksize} rest_splits={n_rest_splits}")
        print(f"{'backend':>7s} {'splitA':>6s} {'|A|':>6s} {'fit':>7s} {'ext':>7s} {'total':>7s} {'n_obs':>6s} {'ARI':>6s} {'AMI':>6s}")

        for frac in splitA_fracs:
            idx = rng.permutation(N)

            nA = max(1, int(N * frac))
            splitA_idx = idx[:nA]
            rest_idx = idx[nA:]

            # fit on split-A only
            t0 = time.time()
            model = sdo.SDOclust().fit(X[splitA_idx])
            fit_time = time.time() - t0

            (_, O), (_, l) = extract_observers_and_labels(model, d=d)
            n_obs = len(O)

            # split rest into n parts
            if len(rest_idx) > 0:
                rest_splits = make_splits(rest_idx, n_splits=n_rest_splits, seed=seed, shuffle=True)
            else:
                rest_splits = []

            # extend on rest splits (backend)
            t1 = time.time()
            y_pred = np.empty(N, dtype=int)

            # Labeling split-A (for ARI/AMI computatoin)
            idxA, yA = extend_on_split(X, splitA_idx, O, l, knn=knn, chunksize=chunksize)
            y_pred[idxA] = yA

            # rest extend
            if len(rest_splits) > 0:
                results = run_extend_backend(
                    X, rest_splits, O, l,
                    backend=backend, knn=knn, chunksize=chunksize,
                    n_jobs=n_jobs, client=client
                )
                y_pred = merge_results(y_pred, results)

            ext_time = time.time() - t1

            total_time = fit_time + ext_time
            ari = adjusted_rand_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)

            print(f"{backend:>7s} {frac:6.2f} {nA:6d} {fit_time:7.3f} {ext_time:7.3f} {total_time:7.3f} {n_obs:6d} {ari:6.3f} {ami:6.3f}")


def baseline_sdoclust(X, y_true):
    t0 = time.time()
    y_pred = sdo.SDOclust().fit_predict(X)
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print("\n" + "-" * 70)
    print(f"Baseline SDOclust Time={te:.3f}s  ARI={ari:.3f}  AMI={ami:.3f}")


if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=50000, centers=5, n_features=10, random_state=0)
    X = StandardScaler().fit_transform(X)

    # Show baseline
    baseline_sdoclust(X, y_true)
    
    # Set Client
    client = None
    if HAS_DASK_DISTRIBUTED:
        cluster = LocalCluster(
            n_workers=4,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        )
        client = Client(cluster)


    # Fixed n_splits
    experiment_fixed_nsplits(
        X, y_true,
        n_splits_list=(2, 4, 8, 16),
        splitA_pos=0,
        knn=10,
        chunksize=2000,
        seed=0,
        backends=("seq", "joblib", "dask-local", "dask-dist"),
        n_jobs=-1,
        client=client,
    )

    # split-A fraction sweep + parallel rest_splits
    experiment_splitA_rest(
        X, y_true,
        splitA_fracs=(0.02, 0.05, 0.10, 0.20, 0.40, 1.00),
        n_rest_splits=8,
        knn=10,
        chunksize=2000,
        seed=0,
        backends=("seq", "joblib", "dask-local", "dask-dist"),
        n_jobs=-1,
        client=client,
    )

    if client is not None:
        client.close()
        cluster.close()