import tslearn
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesResampler
import matplotlib.pyplot as plt
import numpy as np
from visualize import visualize_clusters, visualize_kernel_kmeans

def K_Means(X_train, n_clusters, dist, n_init=1, seed=None):
    if dist == "cosine":
        cluster_alg = KernelKMeans(n_clusters=n_clusters, kernel=dist, n_init=n_init, random_state=seed)
        y_pred = cluster_alg.fit_predict(X_train)
        visualize_kernel_kmeans(X_train, y_pred, cluster_alg.n_clusters)
    if dist == "euclidean" or dist == "dtw":
        cluster_alg = TimeSeriesKMeans(n_clusters=n_clusters, metric=dist, random_state=seed, n_init=n_init)
        y_pred = cluster_alg.fit_predict(X_train)
        visualize_clusters(X_train, y_pred,  cluster_alg.n_clusters,cluster_alg.cluster_centers_)


def hoora(X_train, n_clusters, n_init=1, seed=None, title_suffix=""):
    X_train_eucl = np.nan_to_num(X_train, 0)
    km, y_pred = K_Means(X_train_eucl, n_clusters, "euclidean", n_init, seed)
    visualize_clusters(km, X_train_eucl, y_pred, title="Euclidean KMeans, not equalized" + title_suffix)

    km, y_pred = K_Means(X_train, n_clusters, "dtw", n_init, seed)
    visualize_clusters(km, X_train, y_pred, title="DTW KMeans, not equalized" + title_suffix)

    km, y_pred = K_Means(X_train, n_clusters, "cosine", n_init, seed)
    visualize_kernel_kmeans(km, X_train, y_pred, title="Kernel KMeans, not equalized" + title_suffix)