import numpy as np
import time
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import joblib
import dask
from dask import delayed, compute


def kmeans_seq(X, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    model.fit(X)
    return model

def minibatch_kmeans_seq(X, n_clusters=5, batch_size=1000):
    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0,  n_init="auto")
    model.fit(X)
    return model

def minibatch_kmeans_joblib(X, n_clusters=5, batch_size=1000, n_jobs=4):
    chunks = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]

    def partial_fit_chunk(chunk):
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0,  n_init="auto")
        model.partial_fit(chunk)
        return model.cluster_centers_

    centers = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(partial_fit_chunk)(chunk) for chunk in chunks
    )
    centers = np.mean(centers, axis=0)

    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0, init=centers, n_init=1)
    model.fit(X)
    return model

def minibatch_kmeans_dask(X, n_clusters=5, batch_size=1000):
    chunks = [X[i:i+batch_size] for i in range(0, len(X), batch_size)]

    @delayed
    def partial_fit_chunk(chunk):
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0, n_init="auto")
        model.partial_fit(chunk)
        return model.cluster_centers_

    results = [delayed(partial_fit_chunk)(chunk) for chunk in chunks]
    centers = compute(*results)
    centers = np.mean(centers, axis=0)

    model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=0, init=centers, n_init=1)
    model.fit(X)
    return model

def evaluate_model(model, X, y_true, name, t_start):
    y_pred = model.predict(X)
    t_elapsed = time.time() - t_start
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    print(f"{name:25s} Time={t_elapsed:.3f}s  ARI={ari:.3f}  AMI={ami:.3f}")

if __name__ == "__main__":
    X, y = make_blobs(n_samples=50000, centers=5, n_features=10, random_state=0)

    t = time.time()
    model_seq = kmeans_seq(X, n_clusters=5)
    evaluate_model(model_seq, X, y, "KMeans Sequential", t)

    t = time.time()
    model_mb_seq = minibatch_kmeans_seq(X, n_clusters=5, batch_size=2000)
    evaluate_model(model_mb_seq, X, y, "MiniBatchKMeans Seq", t)

    t = time.time()
    model_joblib = minibatch_kmeans_joblib(X, n_clusters=5, batch_size=2000, n_jobs=-1)
    evaluate_model(model_joblib, X, y, "MiniBatchKMeans Joblib", t)

    t = time.time()
    model_dask = minibatch_kmeans_dask(X, n_clusters=5, batch_size=2000)
    evaluate_model(model_dask, X, y, "MiniBatchKMeans Dask", t)

