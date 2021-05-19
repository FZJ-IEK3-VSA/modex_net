# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""


def df_sum(df):
    return df.sum()


def df_mean(df):
    return df.mean()


def df_std(df):
    return df.std()


def df_normalized_std(df):
    return df.std()/df.mean()


def df_percentile(df, percent):
    return df.quantile(percent).rename(None)


def df_percentile_converter(df, percent):
    return df.clip(0, df_percentile(df, percent), axis=1).sum()


operators_dict = {'sum': df_sum,
                  'mean': df_mean,
                  'std': df_std,
                  'normalized_std': df_normalized_std,
                  'percentile': df_percentile,
                  'percentile_converter': df_percentile_converter}
