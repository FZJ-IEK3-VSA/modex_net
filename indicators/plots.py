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


linewidth = 2.5
fontsize = 16

colors_energy_mix = {'Wind': 'mediumblue',
                     'Wind Onshore': 'blue',
                     'Wind Offshore': 'mediumblue',
                     'Solar': 'yellow',
                     'Hydro': 'dodgerblue',
                     'Run of River': 'deepskyblue',
                     'Reservoir': 'dodgerblue',
                     'Pumped Hydro Storage': 'lightskyblue',
                     'Bioenergy': 'green',
                     'Nuclear': 'purple',
                     'Lignite': 'saddlebrown',
                     'Hard Coal': 'black',
                     'Natural Gas': 'orange',
                     'Other': 'darkgray',
                     'Other Conventional': 'darkgray',
                     'Other Renewable': 'lime',
                     'Oil': 'red',
                     'Geothermal': 'gold',
                     'Waste': 'lightgray',
                     'Batteries': 'pink',
                     'Hydrogen': 'hotpink',
                     'Storage': 'hotpink'}

aggregate_carriers = {'Wind Onshore': 'Wind',
                      'Wind Offshore': 'Wind',
                      'Solar': 'Solar',
                      'Run of River': 'Hydro',
                      'Reservoir': 'Hydro',
                      'Pumped Hydro Storage': 'Hydro',
                      'Bioenergy': 'Bioenergy',
                      'Nuclear': 'Nuclear',
                      'Lignite': 'Lignite',
                      'Hard Coal': 'Hard Coal',
                      'Natural Gas': 'Natural Gas',
                      'Other Conventional': 'Other',
                      'Other Renewable': 'Other',
                      'Oil': 'Oil',
                      'Geothermal': 'Other',
                      'Waste': 'Other'}


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


def plot_dendrogram(data, color_threshold=0.7, title=None, figsize=(8,6), linewidth=linewidth, fontsize=fontsize,
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


def plot_load_duration_df(df, title=None, ylabel=None, legend=True, figsize=(8,6), linewidth=linewidth,
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


# inspired by https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
def plot_clustered_stacked(dfall, labels=None, figsize=(16,9), title=None, ylabel=None, ylim=None, fontsize=fontsize,
                           **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    H = "/"

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    fig, axe = plt.subplots(1, 1, figsize=figsize)

    for df in dfall:  # for each data frame
        axe = df.plot(ax=axe, kind="bar", stacked=True, linewidth=0, legend=False,
                      color=df.columns.map(colors_energy_mix),**kwargs)  # make bar plots

    patch_width = 1 / float(n_df + 1)

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(patch_width)

    axe.set_xticks((np.arange(0, n_ind, 1) + patch_width * 2))
    axe.set_xticklabels(df.index, rotation=0, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    axe.set_xlim(-0.4, axe.get_xticks()[-1] + 0.55)
    axe.set_ylim(0., ylim)
    axe.yaxis.grid(True, linestyle='dotted')

    axe.set_title(title, fontsize=fontsize + 4)
    axe.set_ylabel(ylabel, fontsize=fontsize + 2)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], fontsize=fontsize, bbox_to_anchor=(1, 1))
    if labels is not None:
        l2 = plt.legend(n, labels, fontsize=fontsize, bbox_to_anchor=(1, 0.35))
    axe.add_artist(l1)

    plt.tight_layout()
    plt.show()

    return axe


def plot_heatmap(df, quantity, metric=None, title=None, figsize=(12,8), fontsize=fontsize, savefig=None, dpi=300, **kwargs):

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    valfmt = "{x:.1f}"
    cmap = "RdYlGn_r"
    if quantity == 'energy_mix':
        df = df.multiply(100)
        percent = True
        vmin = 0.
        vmax = 100.
        cbarlabel = "Percentage"
    else:
        if metric == 'correlation':
            cmap = "RdYlGn"
        percent = False
        vmin = df.min().min()
        vmax = df.max().max()
        cbarlabel = "Distance"
        if vmax < 1.:
            valfmt = "{x:.2f}"

    threshold = 0.2 * vmax
    textcolors = ["white", "black"]
    if metric == 'correlation':
        threshold = 0.8 * vmax
        textcolors = ["black", "white"]

    im, _ = _heatmap(df.values, df.index.values, df.columns.values, ax=ax, cmap=cmap, percent=percent, vmin=vmin,
                     vmax=vmax, valfmt=valfmt, cbarlabel=cbarlabel, fontsize=fontsize, **kwargs)
    _annotate_heatmap(im, valfmt=valfmt, size=fontsize, threshold=threshold, textcolors=textcolors)
    plt.xticks(rotation=-90)
    ax.set_title(title, fontsize=fontsize + 4)
    fig.tight_layout()

    if savefig: plt.savefig(savefig, dpi=dpi)

    return ax


# heatmap fucnctions from https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def _heatmap(data, row_labels, col_labels, percent=True, vmin=0, vmax=100, valfmt="{x:.2f}", ax=None,
            cbar_kw={}, cbarlabel="", fontsize=fontsize, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    cbar = ax.figure.colorbar(im, cax=cax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize+2)
    if percent:
        cbar.ax.set_yticklabels([str(i)+'%' for i in range(0, 120, 20)], fontsize=fontsize)
    else:
        cbar.ax.set_yticklabels([valfmt.replace('x', '').format(i) for i in np.linspace(vmin, vmax, len(cbar.ax.get_yticks()))],
                                fontsize=fontsize)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels, fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-60, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
#%%
def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
