import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sample_size = 500


def plot_pca(data: np.ndarray, labels: np.ndarray, save_fname: str):
    """
    Plot PCA Diagramm for the data. Labels specify the class for the given data array.
    So they must be ordered in the same.
    :param data: data to plot
    :param labels: labels for the given data (e.g. synthetic, real, ..)
    :param save_fname: Save name for file
    :return:
    """
    n_components = 2
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )

    principalDf["target"] = labels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)
    targets = np.unique(labels)
    colors = (
        ["red", "green"]
        if len(targets) == 2
        else ["black", "blue", "red", "cyan", "pink", "yellow", "green", "orange"]
    )
    for target_i, color in zip(targets, colors):
        indicesToKeep = principalDf["target"] == target_i
        ax.scatter(
            principalDf.loc[indicesToKeep, "principal component 1"],
            principalDf.loc[indicesToKeep, "principal component 2"],
            c=color,
            s=50,
        )

    ax.legend(targets)
    ax.grid()
    plt.savefig(save_fname)


def plot_tsne(data: np.ndarray, labels: np.ndarray, save_fname: str):
    """
    Plot t-SNE Diagramm for the data. Labels specify the class for the given data array.
    So they must be ordered in the same.
    :param data: data to plot
    :param labels: labels for the given data (e.g. synthetic, real, ..)
    :param save_fname: Save name for file
    :return:
    """
    n_components = 2
    tsne = TSNE(n_components=n_components, n_iter=300)
    tsne_results = pd.DataFrame(tsne.fit_transform(data))
    tsne_results["target"] = labels
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)
    targets = np.unique(labels)
    colors = (
        ["red", "green"]
        if len(targets) == 2
        else ["black", "blue", "red", "cyan", "pink", "yellow", "green", "orange"]
    )
    for target_i, color in zip(targets, colors):
        indicesToKeep = tsne_results["target"] == target_i
        ax.scatter(
            tsne_results.loc[indicesToKeep, 0],
            tsne_results.loc[indicesToKeep, 1],
            c=color,
            s=50,
        )
    ax.legend(targets)
    ax.grid()
    plt.savefig(save_fname)
