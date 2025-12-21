import time
import numpy as np
import scipy.spatial.distance as distance

from joblib import Parallel, delayed
import dask
dask.config.set({"distributed.worker.daemon": False})

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import sdoclust as sdo



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


# Extend-labels chunk (cdist brute)
def extend_labels_chunk(X_chunk, O, l, knn=10):
    valid = (l >= 0)
    O2 = O[valid]
    l2 = l[valid]

    if O2.shape[0] == 0:
        y_chunk = -np.ones(X_chunk.shape[0], dtype=int)
        C_chunk = np.zeros((X_chunk.shape[0], 0))
        return C_chunk, y_chunk

    labs = np.unique(l2)
    lab_to_idx = {lab: i for i, lab in enumerate(labs)}
    l_idx = np.fromiter((lab_to_idx[v] for v in l2), dtype=int, count=len(l2))

    cl = len(labs)
    C = np.zeros((X_chunk.shape[0], cl), dtype=float)

    dist = distance.cdist(X_chunk, O2, metric="euclidean")
    k = min(knn, O2.shape[0])
    closest = np.argpartition(dist, k - 1, axis=1)[:, :k]
    lknn = l_idx[closest]

    for j in range(cl):
        C[:, j] = np.sum(lknn == j, axis=1)

    y_idx = np.argmax(C, axis=1)
    y_pred = labs[y_idx]
    return C / max(1, k), y_pred


def _merge_results(results):
    C_list = [res[0] for res in results if res[0].size > 0]
    C_all = np.vstack(C_list) if C_list else np.empty((0, 0))
    y_all = np.hstack([res[1] for res in results]) if results else np.empty((0,), dtype=int)
    return C_all, y_all


def extend_labels_seq(X, O, l, knn=10, chunksize=2000):
    results = []
    for i in range(0, len(X), chunksize):
        results.append(extend_labels_chunk(X[i:i+chunksize], O, l, knn=knn))
    return _merge_results(results)


def extend_labels_joblib(X, O, l, knn=10, chunksize=2000, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(extend_labels_chunk)(X[i:i+chunksize], O, l, knn)
        for i in range(0, len(X), chunksize)
    )
    return _merge_results(results)


def extend_labels_dask(X, O, l, knn=10, chunksize=2000):
    tasks = [
        dask.delayed(extend_labels_chunk)(X[i:i+chunksize], O, l, knn)
        for i in range(0, len(X), chunksize)
    ]
    results = dask.compute(*tasks)
    return _merge_results(results)




if __name__ == "__main__":
    X, y_true = make_blobs(n_samples=50000, centers=5, n_features=10, random_state=0)
    X = StandardScaler().fit_transform(X)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))  # fix for reproducibility

    splitA_fracs = [0.02, 0.05, 0.10, 0.20, 0.40, 1.00]
    knn = 10
    chunksize = 2000
    for backend in ["seq", "joblib", "dask"]:
        print("\n" + "="*60)
        print(f"[backend={backend}] knn={knn} chunksize={chunksize}")
        
        print(f"{'backend':>7s} {'splitA':>6s} {'|A|':>6s} {'fit':>7s} {'ext':>7s} {'total':>7s} {'n_obs':>6s} {'ARI':>6s} {'AMI':>6s}")

        for frac in splitA_fracs:
            nA = max(1, int(len(X) * frac))
            X_A = X[idx[:nA]]

            # fit on split-A
            t0 = time.time()
            model = sdo.SDOclust().fit(X_A)
            fit_time = time.time() - t0

            (_, O), (_, l) = extract_observers_and_labels(model, d=X.shape[1])
            n_obs = len(O)

            t1 = time.time()
            if backend == "seq":
                _, y_pred = extend_labels_seq(X, O, l, knn=knn, chunksize=chunksize)
            elif backend == "joblib":
                _, y_pred = extend_labels_joblib(X, O, l, knn=knn, chunksize=chunksize, n_jobs=-1)
            elif backend == "dask":
                _, y_pred = extend_labels_dask(X, O, l, knn=knn, chunksize=chunksize)
            ext_time = time.time() - t1

            total_time = fit_time + ext_time
            ari = adjusted_rand_score(y_true, y_pred)
            ami = adjusted_mutual_info_score(y_true, y_pred)

            print(f"{backend:>7s} {frac:6.2f} {nA:6d} {fit_time:7.3f} {ext_time:7.3f} {total_time:7.3f} {n_obs:6d} {ari:6.3f} {ami:6.3f}")

