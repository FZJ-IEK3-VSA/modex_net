# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import os
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde


def wasserstein(a, b):
    x_min = min([a.min(), b.min()])
    x_max = max([a.max(), b.max()])
    x = np.linspace(x_min, x_max, 1000)

    pdf_a = gaussian_kde(a)
    pdf_b = gaussian_kde(b)
    return wasserstein_distance(pdf_a(x), pdf_b(x))


def bhattacharyya(a, b):
    """ Bhattacharyya distance between distributions (lists of floats).
    see https://gist.github.com/miku/1671b2014b003ee7b9054c0618c805f7
    """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    x_min = min([min(a), min(b)])
    x_max = max([max(a), max(b)])
    x = np.linspace(x_min, x_max, 1000)

    pdf_a = gaussian_kde(a)
    pdf_b = gaussian_kde(b)
    if sum(pdf_a(x)) and sum(pdf_b(x)):
        return -np.log(sum((np.sqrt(u * w) for u, w in zip(pdf_a(x)/sum(pdf_a(x)), pdf_b(x)/sum(pdf_b(x))))))
    else:
        return -0.01


def bhattacharyya_norm(a, b):
    return bhattacharyya(a - a.mean(), b - b.mean())


def max_diff(a, b):
    return (a - b).abs().max()


def mape(a, b):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(a, b)


def euclidean(a, b):
    return np.linalg.norm(a - b)


def frechet(a, b):
    import frechetdist
    return frechetdist.frdist([range(len(a)), a], [range(len(b)), b])


def dtw(a, b):

    # import module from given path: https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    import importlib.util
    dtw_mod_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dtw", "dtw", "dtw.py"))
    spec = importlib.util.spec_from_file_location("dtw", dtw_mod_path)
    dtw_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dtw_mod)

    return dtw_mod.accelerated_dtw(a.values, b.values, 'minkowski', p=1)[0]


def corr(a, b):
    return a.corr(b)


metrics_dict = {'wasserstein': wasserstein,
                'bhattacharyya': bhattacharyya,
                'bhattacharyya_norm': bhattacharyya_norm,
                'max_diff': max_diff,
                'mape': mape,
                'euclidean': euclidean,
                'frechet': frechet,
                'dtw': dtw,
                'correlation': corr}