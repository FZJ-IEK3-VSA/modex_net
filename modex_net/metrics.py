# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import os
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde
import similaritymeasures


def _range(a, b):
    x_min = min([a.min(), b.min()])
    x_max = max([a.max(), b.max()])
    return np.linspace(x_min, x_max, 1000)

def _points(x):
    return np.array(list(zip(range(len(x)), x)))

def wasserstein(a, b):
    x = _range(a, b)
    pdf_a = gaussian_kde(a)
    pdf_b = gaussian_kde(b)
    return wasserstein_distance(pdf_a(x), pdf_b(x))

def wasserstein_norm(a, b):
    return wasserstein(a - a.mean(), b - b.mean())

def bhattacharyya(a, b):
    """ Bhattacharyya distance between distributions (lists of floats).
    see https://gist.github.com/miku/1671b2014b003ee7b9054c0618c805f7
    """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    x = _range(a, b)

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

def max_diff_norm(a, b):
    return max_diff(a - a.mean(), b - b.mean())

def mape(a, b):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(a, b)

def mape_norm(a, b):
    return mape(a - a.mean(), b - b.mean())

def euclidean(a, b):
    return np.linalg.norm(a - b)

def euclidean_norm(a, b):
    return euclidean(a - a.mean(), b - b.mean())

def frechet(a, b):
    return similaritymeasures.frechet_dist(_points(a), _points(b))

def frechet_norm(a, b):
    return frechet(a - a.mean(), b - b.mean())

def dtw(a, b):
    d, dd = similaritymeasures.dtw(_points(a), _points(b))
    return d

def dtw_norm(a, b):
    return dtw(a - a.mean(), b - b.mean())

def corr(a, b):
    return a.corr(b)


metrics_dict = {'wasserstein': wasserstein,
                'wasserstein_norm': wasserstein_norm,
                'bhattacharyya': bhattacharyya,
                'bhattacharyya_norm': bhattacharyya_norm,
                'max_diff': max_diff,
                'max_diff_norm': max_diff_norm,
                'mape': mape,
                'mape_norm': mape_norm,
                'euclidean': euclidean,
                'euclidean_norm': euclidean_norm,
                'frechet': frechet,
                'frechet_norm': frechet_norm,
                'dtw': dtw,
                'dtw_norm': dtw_norm,
                'correlation': corr}