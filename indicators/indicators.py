# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, gaussian_kde

from . import plots

import logging
logger = logging.getLogger(__name__)

de_nuts1 = ["DE1", "DE2", "DE3", "DE4", "DE5", "DE6", "DE7", "DE8", "DE9", "DEA", "DEB", "DEC", "DED", "DEE", "DEF", "DEG"]
de_nuts1_names = pd.Series(["Baden-Württemberg", "Bayern", "Berlin", "Brandenburg", "Bremen", "Hamburg", "Hessen",
                            "Mecklenburg-Vorpommern", "Niedersachsen", "Nordrhein-Westfalen", "Rheinland-Pfalz",
                            "Saarland", "Sachsen", "Sachsen-Anhalt", "Schleswig-Holstein", "Thüringen"],
                           index=de_nuts1)

eu_neighs_ISO2 = ["AT", "BE", "CH", "CZ", "DK", "FR", "LU", "NL", "NO", "PL", "SE"]
eu_neighs_names = pd.Series(["Austria", "Belgium", "Switzerland", "Czechia", "Denmark", "France", "Luxemburg",
                             "Netherlands", "Norway", "Poland", "Sweden"], index=eu_neighs_ISO2)

model_names = ["europower", "perseus", "mars", "powerflex", "ego", "miles", "isaar", "elmod"]
model_names_plot = pd.Series(["Europower", "Perseus/Tango", "MarS/ZKNOT", "Powerflex-Grid-EU",
                              "eGo/eTraGo", "MILES", "ISAaR", "ELMOD"], index=model_names)

quantities_time = [
                   "vres_curtailments",    # Dataframes, time/regions
                   "redispatch_vol",       # Dataframes, time/regions
                   "emissions",            # Dataframes, time/regions
                   "load_curtailments",    # Dataframes, time/regions

                   "storage_discharging",  # Dataframes, time/regions
                   "storage_charging",     # Dataframes, time/regions

                   "import_export",        # Dataframes, time/interconnections
                   "line_loadings",        # Dataframes, time/lines

                   "electricity_prices",   # Dataframes, time/regions
                   "redispatch_cost"       # Dataframes, time/regions
                  ]
quantities_categorical = ["energy_mix"]    # Dataframes, carrier/regions
quantities = quantities_time + quantities_categorical


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
    return -np.log(sum((np.sqrt(u * w) for u, w in zip(pdf_a(x)/sum(pdf_a(x)), pdf_b(x)/sum(pdf_b(x))))))


def frechet(a, b):
    return (a - b).abs().max()


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


metrics = {'wasserstein': wasserstein,
           'bhattacharyya': bhattacharyya,
           'frechet': frechet,
           'dtw': dtw,
           'correlation': corr}


class Calculator:

    def _quantity_get_set(quantity):

        @property
        def quantity_prop(self):

            for model in model_names:

                if getattr(self, '_'+quantity)[model].empty:
                    logger.info(str(quantity)+" data for "+str(model)+" is empty. Loading from "+str(self.data_source))

                    if self.data_source == "csv":

                        data_path = os.path.join(os.path.dirname(__file__), "..", "data", str(self.scenario), model,
                                                 quantity+".csv")

                        if quantity in quantities_time:
                            getattr(self, '_'+quantity).update({model: pd.read_csv(data_path,
                                                                                   index_col='snapshots',
                                                                                   parse_dates=True)})
                        if quantity in quantities_categorical:
                            getattr(self, '_'+quantity).update({model: pd.read_csv(data_path,
                                                                               index_col='carrier')})

                    elif self.data_source == "oep":
                        pass

                    else:
                        raise NotImplementedError

            return getattr(self, '_'+quantity)

        @quantity_prop.setter
        def quantity_prop(self, value):
            setattr(self, '_'+quantity, value)

        return quantity_prop

    # quantities
    vres_curtailments = _quantity_get_set("vres_curtailments")
    redispatch_vol = _quantity_get_set("redispatch_vol")
    emissions = _quantity_get_set("emissions")
    load_curtailments = _quantity_get_set("load_curtailments")

    storage_discharging = _quantity_get_set("storage_discharging")
    storage_charging = _quantity_get_set("storage_charging")

    exports = _quantity_get_set("exports")
    imports = _quantity_get_set("imports")
    line_loadings = _quantity_get_set("line_loadings")

    electricity_prices = _quantity_get_set("electricity_prices")
    redispatch_cost = _quantity_get_set("redispatch_cost")

    energy_mix = _quantity_get_set("energy_mix")

    def __init__(self, scenario, data_source="csv"):

        logging.basicConfig(level=logging.INFO)
        logger.info("All methods assume hourly profiles.")

        self.scenario = scenario
        self.data_source = data_source

        for quantity in quantities:
            setattr(self, quantity, dict(zip(model_names, [pd.DataFrame()]*len(model_names))))

    def sum(self, quantity):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"

        return (pd.concat([getattr(self, quantity)[model].sum() for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def normalized_std(self, quantity):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"

        return (pd.concat([getattr(self, quantity)[model].std()/getattr(self, quantity)[model].mean()
                           for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def percentile(self, quantity, percent=0.75):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert 0. <= percent <= 1., "Percent should be in the [0, 1] domain."

        return (pd.concat([getattr(self, quantity)[model].quantile(percent).rename(None)
                           for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def percentile_converter(self, quantity, percent=0.75):

        converter_capacities = self.percentile(quantity, percent)

        return (pd.concat([getattr(self, quantity)[model].clip(0, converter_capacities[model], axis=1).sum()
                           for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def plot_dendrogram(self, quantity, func, percent=None, **kwargs):

        one_arg_funcs = ['sum', 'normalized_std']
        two_arg_funcs = ["percentile", "percentile"]
        all_funcs = one_arg_funcs + two_arg_funcs

        assert func in all_funcs, "Valid functions can only be one of [" + ", ".join(all_funcs) + "]"

        if percent:
            if func in two_arg_funcs:
                plots.plot_dendrogram(getattr(self, func)(quantity, percent).transpose(), **kwargs)
                return
            else:
                logger.warning("This function does accept a second argument. It is ignored.")
        plots.plot_dendrogram(getattr(self, func)(quantity).transpose(), **kwargs)

    def pair_distance(self, quantity, metric):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert metric in metrics.keys(), "Valid metrics can only be one of [" + ", ".join(metrics.keys()) + "]"

        return {model: (pd.concat([(getattr(self, quantity)[model]
                                    .combine(getattr(self, quantity)[mod], metrics[metric])
                                    .iloc[0]
                                    .rename(None))
                                   for mod in model_names], axis=1)
                        .rename(columns=dict(enumerate(model_names))))
                for model in model_names}

    def line_overload_incidents(self, threshold=0.7):
        """
         Returns the percentage of all line overloads over all time steps. A line overload is considered when it's load
         exceeds the corresponding capacity times the given threshold.
        """

        def overload_incidents(df):
            df = df > threshold
            return df.sum().sum() / (df.shape[0] * df.shape[1])

        return pd.Series([overload_incidents(df) for df in self.line_loadings.values()],
                         index=self.line_loadings.keys())

