import time
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from joblib import Parallel, delayed
from dask.distributed import Client

import sdoclust as sdo
from sklearn.datasets import make_blobs

def evaluate(y_true, y_pred, name, t0):
    te = time.time() - t0
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    noise = np.mean(y_pred == -1)
    print(f"{name:32s} Time={te:.3f}s  ARI={ari:.3f}  AMI={ami:.3f}  noise={noise:.3f}")


# Returns a list of index arrays
# [0..n-1]
def make_splits(n, n_splits, seed=0, shuffle=True):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
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


# Fit SDOclust on one split
def fit_sdoclust_on_split(Xi):
    model = sdo.SDOclust().fit(Xi)
    return model


# Aggregate observers from all splits
def aggregate_observers(models, d):
    Os, ls = [], []
    offset = 0
    for m in models:
        (_, Oi), (_, li) = extract_observers_and_labels(m, d=d)

        li2 = li.copy()
        mask = li2 >= 0
        li2[mask] = li2[mask] + offset

        # update offset by number of clusters in this split
        if np.any(mask):
            offset = li2[mask].max() + 1

        Os.append(Oi)
        ls.append(li2)

    O_all = np.vstack(Os) if Os else np.empty((0, d))
    l_all = np.concatenate(ls) if ls else np.empty((0,), dtype=int)
    return O_all, l_all


# compress observers (communication/memory)
def compress_observers_sample(O, l, per_cluster=500, seed=0):
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


# knn voting 
def knn_vote_label_extension(Xi, O_all, l_all, x=10, algorithm="ball_tree"):
    if O_all.shape[0] == 0:
        return -np.ones(len(Xi), dtype=int)

    valid = (l_all >= 0)
    O = O_all[valid]
    l = l_all[valid]
    if O.shape[0] == 0:
        return -np.ones(len(Xi), dtype=int)

    x_eff = min(x, O.shape[0])

    nn = NearestNeighbors(n_neighbors=x_eff, algorithm=algorithm, metric="euclidean")
    nn.fit(O)

    dist, ind = nn.kneighbors(Xi, return_distance=True)
    lab = l[ind]

    y = np.full(len(Xi), -1, dtype=int)
    for i in range(len(Xi)):
        labs_i = lab[i]
        dist_i = dist[i]

        uniq, counts = np.unique(labs_i, return_counts=True)
        maxc = counts.max()
        tied = uniq[counts == maxc]

        if len(tied) == 1:
            y[i] = tied[0]
        else:
            best_lab = None
            best_sum = None
            for t in tied:
                s = dist_i[labs_i == t].sum()
                if best_sum is None or s < best_sum:
                    best_sum = s
                    best_lab = t
            y[i] = best_lab

    return y


# sequential
def run_pipeline_sequential(X, splits, x_vote=10, compress_per_cluster=None, seed=0):
    d = X.shape[1]

    # Fit SDOclust on each split
    models = []
    t_fit = time.time()
    for idx in splits:
        Xi = X[idx]
        models.append(fit_sdoclust_on_split(Xi))
    t_fit = time.time() - t_fit

    # Aggregate observers
    O_all, l_all = aggregate_observers(models, d=d)

    # Label extension on each split (sequential)
    y_pred = np.empty(len(X), dtype=int)
    t_ext = time.time()
    for idx in splits:
        Xi = X[idx]
        y_pred[idx] = knn_vote_label_extension(Xi, O_all, l_all, x=x_vote)
    t_ext = time.time() - t_ext

    return y_pred, {"fit_time": t_fit, "ext_time": t_ext, "n_obs": len(O_all)}


# joblib
def run_pipeline_joblib(X, splits, x_vote=10, compress_per_cluster=None, seed=0, n_jobs=-1):
    d = X.shape[1]

    # Fit SDOclust on each split
    t_fit = time.time()
    models = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(fit_sdoclust_on_split)(X[idx]) for idx in splits
    )
    t_fit = time.time() - t_fit

    # Aggregate observers
    O_all, l_all = aggregate_observers(models, d=d)

    if compress_per_cluster is not None:
        O_all, l_all = compress_observers_sample(O_all, l_all, per_cluster=compress_per_cluster, seed=seed)

    # Label extension on each split (sequential)
    t_ext = time.time()
    ys = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(knn_vote_label_extension)(X[idx], O_all, l_all, x_vote) for idx in splits
    )
    t_ext = time.time() - t_ext

    y_pred = np.empty(len(X), dtype=int)
    for idx, yi in zip(splits, ys):
        y_pred[idx] = yi

    return y_pred, {"fit_time": t_fit, "ext_time": t_ext, "n_obs": len(O_all)}


# dask
def run_pipeline_dask(X, splits, client: Client, x_vote=10, compress_per_cluster=None, seed=0):
    d = X.shape[1]

    # Fit SDOclust on each split (distributed)
    t_fit = time.time()
    fit_futs = [client.submit(fit_sdoclust_on_split, X[idx]) for idx in splits]
    models = client.gather(fit_futs)
    t_fit = time.time() - t_fit

    # Aggregate observers (central)
    O_all, l_all = aggregate_observers(models, d=d)

    if compress_per_cluster is not None:
        O_all, l_all = compress_observers_sample(O_all, l_all, per_cluster=compress_per_cluster, seed=seed)

    # Extension (distributed)
    t_ext = time.time()
    ext_futs = [client.submit(knn_vote_label_extension, X[idx], O_all, l_all, x_vote) for idx in splits]
    ys = client.gather(ext_futs)
    t_ext = time.time() - t_ext

    y_pred = np.empty(len(X), dtype=int)
    for idx, yi in zip(splits, ys):
        y_pred[idx] = yi

    return y_pred, {"fit_time": t_fit, "ext_time": t_ext, "n_obs": len(O_all)}


if __name__ == "__main__":
    X, y = make_blobs(n_samples=50000, centers=5, n_features=10, random_state=0)
    X = StandardScaler().fit_transform(X)

    x_vote = 10
    compress_per_cluster = 500
    
    import dask
    dask.config.set({"distributed.worker.daemon": False})

    from dask.distributed import Client
    client = Client(n_workers=4, threads_per_worker=1, processes=True)

    # Test splits
    for n_splits in [1, 2, 4, 8, 16]:
        splits = make_splits(len(X), n_splits=n_splits, seed=0, shuffle=True)

        t0 = time.time()
        y_seq, info_seq = run_pipeline_sequential(X, splits, x_vote=x_vote,
                                                  compress_per_cluster=compress_per_cluster, seed=0)
        evaluate(y, y_seq, f"SEQ (splits={n_splits})", t0)
        print("  ", info_seq)

        t0 = time.time()
        y_jb, info_jb = run_pipeline_joblib(X, splits, x_vote=x_vote,
                                            compress_per_cluster=compress_per_cluster, seed=0, n_jobs=-1)
        evaluate(y, y_jb, f"JOBLIB (splits={n_splits})", t0)
        print("  ", info_jb)

        t0 = time.time()
        y_dk, info_dk = run_pipeline_dask(X, splits, client=client, x_vote=x_vote,
                                          compress_per_cluster=compress_per_cluster, seed=0)
        evaluate(y, y_dk, f"DASK (splits={n_splits})", t0)
        print("  ", info_dk)

    client.close()
