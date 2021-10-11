import math

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


def show_one_session(df, id, log=False):
    s39 = df[df["session_id"] == id]
    if log:
        s39["price"] = s39.apply(lambda x: math.log(x["price"]), axis=1)
    fig = px.line(s39, x='full_date', y="price")
    fig.show()


def visualize_kernel_kmeans(X_train, y_pred, n, title=None):
    plt.figure(figsize=(20, 20))
    n_subplots = int(np.ceil(np.sqrt(n)))
    for yi in range(n):
        plt.subplot(n_subplots, n_subplots, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=0.2)
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)
    plt.show()


def visualize_clusters(X_train, y_pred, n, clusters, title=None):
    plt.figure(figsize=(20, 20))
    n_subplots = int(np.ceil(np.sqrt(n)))

    for yi in range(n):
        plt.subplot(n_subplots, n_subplots, 1 + yi)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=0.2)
        plt.plot(clusters[yi].ravel(), "r-")
        plt.text(0.55, 0.85, "Cluster %d" % (yi + 1), transform=plt.gca().transAxes)
        if yi == 1:
            plt.title(title)

    plt.show()
