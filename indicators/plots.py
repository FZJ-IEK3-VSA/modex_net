# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import pandas as pd
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

figsize = (8,6)
linewidth = 2.5
fontsize = 16

def _plot_dendrogram(model, color_threshold=0.7, **kwargs):
    # from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=max(linkage_matrix[:, 2]) * color_threshold, **kwargs)


def plot_dendrogram(data, color_threshold=0.7, title=None, figsize=figsize, linewidth=linewidth, fontsize=fontsize,
                    xticks_rotation=45, savefig=None, dpi=300, **kwargs):
    """
    Plot a dendrogram from the data. Rows correspond to samples and columns to features.
    """

    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(data)

    matplotlib.rcParams['lines.linewidth'] = linewidth

    fig, ax = plt.subplots(figsize=figsize)

    if title is None: title = 'Hierarchical Clustering Dendrogram'
    plt.title(title, fontsize=fontsize)

    _plot_dendrogram(model, color_threshold=color_threshold, leaf_rotation=xticks_rotation, leaf_font_size=fontsize,
                     ax=ax, **kwargs)

    # set labels
    if isinstance(data, pd.DataFrame):
        labels = pd.Series(data.index)
        ax.set_xticklabels([labels[int(i.get_text())] for i in ax.get_xticklabels()])
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Distance', fontsize=fontsize + 2)

    plt.tight_layout()

    if savefig: plt.savefig(savefig, dpi=dpi)

    return ax


def plot_load_duration_df(df, title=None, ylabel=None, legend=True, figsize=figsize, linewidth=linewidth,
                          fontsize=fontsize, savefig=None, dpi=300, **kwargs):

    df = df.copy()

    # re-write datetimes in integer format
    df.index = list(range(len(df.index)))

    df = pd.DataFrame({key: sorted(values, reverse=True) for key, values in df.transpose().iterrows()})
    df.index.name = 'Time'

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax = df.plot(ax=ax, legend=legend, linewidth=linewidth, fontsize=fontsize, **kwargs)
    ax.set_title(title, fontsize=fontsize+4)
    ax.set_ylabel(ylabel, fontsize=fontsize+2)
    ax.set_xlabel('Time', fontsize=fontsize)
    if legend: ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=fontsize)

    plt.tight_layout()
    if savefig: plt.savefig(savefig, dpi=dpi)

    return ax