# run_parallel_sdoclust.py

import time
import numpy as np
from dask.distributed import Client

import sdoclust as sdo
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from dbscan_extension import (
    evaluate,
    extract_observers_and_labels,
    compress_observers_sample,
    extend_labels_radius_seq,
    extend_labels_radius_threads,
    extend_labels_radius_dask,
    extend_labels_nn1_fast,
    extend_labels_nn1_threads,
    extend_labels_nn1_dask,
)

if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=1, processes=True)

    # data
    X, y = make_blobs(n_samples=50000, centers=5, n_features=10, random_state=0)
    X = StandardScaler().fit_transform(X)

    # split-A
    n1 = len(X) // 5
    X1 = X[:n1]

    t = time.time()
    model = sdo.SDOclust().fit(X1)
    print(f"SDOclust fit done in {time.time()-t:.3f}s")

    (O_name, O1), (l_name, l1) = extract_observers_and_labels(model, d=X.shape[1])
    print(f"Observers: {O_name} {O1.shape}, labels: {l_name} {l1.shape}")

    O_obs, l_obs = compress_observers_sample(O1, l1, per_cluster=200, seed=0)

    # extension parameter (parameter ε)
    eps = 0.9

    # radius voting
    t = time.time()
    y_r_seq = extend_labels_radius_seq(X, O_obs, l_obs, eps)
    evaluate(y, y_r_seq, "Radius voting (seq)", t)

    t = time.time()
    y_r_thr = extend_labels_radius_threads(X, O_obs, l_obs, eps)
    evaluate(y, y_r_thr, "Radius voting (joblib)", t)

    t = time.time()
    y_r_dsk = extend_labels_radius_dask(X, O_obs, l_obs, eps, client=client)
    evaluate(y, y_r_dsk, "Radius voting (dask)", t)

    # 1-NN + eps
    t = time.time()
    y_nn1 = extend_labels_nn1_fast(X, O_obs, l_obs, eps)
    evaluate(y, y_nn1, "1-NN + eps (FAST)", t)

    t = time.time()
    y_nn1_thr = extend_labels_nn1_threads(X, O_obs, l_obs, eps)
    evaluate(y, y_nn1_thr, "1-NN + eps (joblib)", t)

    t = time.time()
    y_nn1_dsk = extend_labels_nn1_dask(X, O_obs, l_obs, eps, client=client)
    evaluate(y, y_nn1_dsk, "1-NN + eps (dask)", t)

    client.close()

    x2 = X[:, :2]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    plots = [
        ("Radius voting (seq)", y_r_seq),
        ("Radius voting (joblib)", y_r_thr),
        ("Radius voting (dask)", y_r_dsk),
        ("1-NN + eps (FAST)", y_nn1),
        ("1-NN + eps (joblib)", y_nn1_thr),
        ("1-NN + eps (dask)", y_nn1_dsk),
    ]

    for ax, (title, labels) in zip(axes, plots):
        sc = ax.scatter(
            x2[:, 0],
            x2[:, 1],
            c=labels,
            s=8,
            cmap="coolwarm",
            alpha=0.9
        )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Parallel SDOclust – Label Extension Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

