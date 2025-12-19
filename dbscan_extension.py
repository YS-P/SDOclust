import numpy as np
import time
from joblib import Parallel, delayed
import dask
from dask.distributed import Client

from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

# Utilities
def evaluate(y_true, y_pred, name, t0):
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    noise = np.mean(y_pred == -1)
    print(f"{name:28s} Time={te:.3f}s  ARI={ari:.3f}  AMI={ami:.3f}  noise={noise:.3f}")

# Find observers(O) and observer labels(l) inside a fitted sdoclust model object
def extract_observers_and_labels(model, d):
    attrs = vars(model)

    # Find candidate O
    O_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[1] == d:
            O_candidates.append((name, val))
            
    # Choose the one with largest n_obs
    O_name, O = max(O_candidates, key=lambda t: t[1].shape[0])

    n_obs = O.shape[0]

    # Find candidate l
    l_candidates = []
    for name, val in attrs.items():
        if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == n_obs:
            if np.issubdtype(val.dtype, np.integer):
                l_candidates.append((name, val))

    # Choose label array with "label" or "cluster"
    def score_label_name(nm):
        nm2 = nm.lower()
        score = 0
        if "label" in nm2: score += 3
        if "cluster" in nm2: score += 2
        if "assign" in nm2: score += 1
        return score

    l_name, l = max(l_candidates, key=lambda t: score_label_name(t[0]))

    return (O_name, O), (l_name, l)





# Build observer index
def build_observer_index(O, l, eps, algorithm="ball_tree"):
    valid = (l >= 0)
    O2 = O[valid]
    l2_raw = l[valid]

    if O2.shape[0] == 0:
        return None, None, None

    labs = np.unique(l2_raw)
    lab_to_idx = {lab:i for i,lab in enumerate(labs)}
    l2 = np.array([lab_to_idx[v] for v in l2_raw], dtype=int)

    nn = NearestNeighbors(radius=eps, algorithm=algorithm, metric="euclidean")
    nn.fit(O2)

    return nn, l2, labs

# Randomly sample up to per_cluster observers
def compress_observers_sample(O, l, per_cluster=200, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for lab in np.unique(l):
        if lab < 0:
            continue
        idx = np.flatnonzero(l == lab)
        if len(idx) <= per_cluster:
            keep.append(idx)
        else:
            keep.append(rng.choice(idx, size=per_cluster, replace=False))
    keep = np.concatenate(keep) if keep else np.array([], dtype=int)
    return O[keep], l[keep]





# Radius voting (DBSCAN like)
# Sequencial, joblib, dask
def _predict_chunk_radius(nn, l2, labs, X_chunk):
    y_idx = np.full(len(X_chunk), -1, dtype=int)
    ind_list = nn.radius_neighbors(X_chunk, return_distance=False)

    K = len(labs)
    for i, inds in enumerate(ind_list):
        if len(inds) == 0:
            continue
        votes = np.bincount(l2[inds], minlength=K)
        y_idx[i] = votes.argmax()

    y = np.full(len(X_chunk), -1, dtype=int)
    ok = y_idx >= 0
    y[ok] = labs[y_idx[ok]]
    return y

def extend_labels_radius_seq(X, O, l, eps, chunksize=20000, algorithm="ball_tree"):
    nn, l2, labs = build_observer_index(O, l, eps, algorithm)
    if nn is None:
        return -np.ones(len(X), dtype=int)

    y_all = np.empty(len(X), dtype=int)
    for start in range(0, len(X), chunksize):
        Xc = X[start:start + chunksize]
        y_all[start:start + len(Xc)] = _predict_chunk_radius(nn, l2, labs, Xc)
    return y_all

def extend_labels_radius_threads(X, O, l, eps, chunksize=20000, n_jobs=-1, algorithm="ball_tree"):
    nn, l2, labs = build_observer_index(O, l, eps, algorithm)
    if nn is None:
        return -np.ones(len(X), dtype=int)

    chunks = [X[i:i + chunksize] for i in range(0, len(X), chunksize)]
    ys = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_predict_chunk_radius)(nn, l2, labs, c) for c in chunks
    )
    return np.concatenate(ys)

def extend_labels_radius_dask(X, O, l, eps, chunksize=20000, algorithm="ball_tree", client=None):
    nn, l2, labs = build_observer_index(O, l, eps, algorithm)
    if nn is None:
        return -np.ones(len(X), dtype=int)

    tasks = [
        dask.delayed(_predict_chunk_radius)(nn, l2, labs, X[i:i+chunksize])
        for i in range(0, len(X), chunksize)
    ]

    if client is None:
        ys = dask.compute(*tasks)
    else:
        futures = client.compute(tasks)
        ys = client.gather(futures)

    return np.concatenate(ys)





# FASTEST: 1-NN + eps cutoff (KMeans-competitive)
# Sequential, joblib (threads), dask
def extend_labels_nn1_fast(
    X, O, l, eps, chunksize=50000, algorithm="ball_tree"
):
    valid = (l >= 0)
    O2 = O[valid]
    l2 = l[valid]

    if O2.shape[0] == 0:
        return -np.ones(len(X), dtype=int)

    nn = NearestNeighbors(n_neighbors=1, algorithm=algorithm, metric="euclidean")
    nn.fit(O2)

    y_all = np.full(len(X), -1, dtype=int)
    for start in range(0, len(X), chunksize):
        Xc = X[start:start + chunksize]
        dist, ind = nn.kneighbors(Xc, return_distance=True)
        dist = dist.ravel()
        ind = ind.ravel()
        ok = dist <= eps
        y_all[start:start + len(Xc)][ok] = l2[ind[ok]]

    return y_all

def _predict_chunk_nn1(nn, l2, eps, X_chunk):
    dist, ind = nn.kneighbors(X_chunk, return_distance=True)
    dist = dist.ravel()
    ind = ind.ravel()
    y = np.full(len(X_chunk), -1, dtype=int)
    ok = dist <= eps
    y[ok] = l2[ind[ok]]
    return y

def extend_labels_nn1_threads(X, O, l, eps, chunksize=50000, n_jobs=-1, algorithm="ball_tree"):
    valid = (l >= 0)
    O2 = O[valid]
    l2 = l[valid]
    if O2.shape[0] == 0:
        return -np.ones(len(X), dtype=int)

    nn = NearestNeighbors(n_neighbors=1, algorithm=algorithm, metric="euclidean")
    nn.fit(O2)

    chunks = [X[i:i+chunksize] for i in range(0, len(X), chunksize)]
    ys = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_predict_chunk_nn1)(nn, l2, eps, c) for c in chunks
    )
    return np.concatenate(ys)

def extend_labels_nn1_dask(X, O, l, eps, chunksize=50000, algorithm="ball_tree", client=None):
    valid = (l >= 0)
    O2 = O[valid]
    l2 = l[valid]
    if O2.shape[0] == 0:
        return -np.ones(len(X), dtype=int)

    nn = NearestNeighbors(n_neighbors=1, algorithm=algorithm, metric="euclidean")
    nn.fit(O2)

    tasks = [
        dask.delayed(_predict_chunk_nn1)(nn, l2, eps, X[i:i+chunksize])
        for i in range(0, len(X), chunksize)
    ]

    if client is None:
        ys = dask.compute(*tasks)
    else:
        futures = client.compute(tasks)
        ys = client.gather(futures)

    return np.concatenate(ys)





# main
if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=1, processes=True)
    
    # data
    X, y = make_blobs(
        n_samples=50000,
        centers=5,
        n_features=10,
        random_state=0
    )
    X = StandardScaler().fit_transform(X)

    # split A
    n1 = len(X) // 5
    X1 = X[:n1]

    # DBSCAN
    eps = 0.9
    min_samples = 10
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X1)
    labels1 = db.labels_

    core_idx = db.core_sample_indices_
    O1 = X1[core_idx]
    l1 = labels1[core_idx]

    print(
        f"DBSCAN on X1: clusters="
        f"{len(set(labels1)) - (1 if -1 in labels1 else 0)}, "
        f"core={len(core_idx)}, "
        f"noise={(labels1 == -1).mean():.3f}"
    )

    # Sample a few core points per cluster
    O_obs, l_obs = compress_observers_sample(O1, l1, per_cluster=200, seed=0)

    print(f"Compressed observers: {len(O1)} -> {len(O_obs)}")

    # Radius voting using compressed observers
    t = time.time()
    y_r_seq = extend_labels_radius_seq(X, O_obs, l_obs, eps, chunksize=20000)
    evaluate(y, y_r_seq, "Radius voting (seq)", t)

    t = time.time()
    y_r_thr = extend_labels_radius_threads(X, O_obs, l_obs, eps, chunksize=20000)
    evaluate(y, y_r_thr, "Radius voting (joblib)", t)
    
    t = time.time()
    y_r_dask = extend_labels_radius_dask(X, O_obs, l_obs, eps, chunksize=20000, client=client)
    evaluate(y, y_r_dask, "Radius voting (dask)", t)

    # FAST: 1-NN + eps cutoff using compressed observers
    t = time.time()
    y_nn1 = extend_labels_nn1_fast(X, O_obs, l_obs, eps)
    evaluate(y, y_nn1, "1-NN + eps (FAST)", t)
    
    t = time.time()
    y_nn1_thr = extend_labels_nn1_threads(X, O_obs, l_obs, eps, chunksize=50000)
    evaluate(y, y_nn1_thr, "1-NN + eps (joblib)", t)

    t = time.time()
    y_nn1_dask = extend_labels_nn1_dask(X, O_obs, l_obs, eps, chunksize=50000, client=client)
    evaluate(y, y_nn1_dask, "1-NN + eps (dask)", t)

    client.close()
