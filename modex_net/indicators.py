# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import os
import pandas as pd
import numpy as np
import calendar
from scipy.stats import wasserstein_distance, gaussian_kde

from . import plots, config

import logging
logger = logging.getLogger(__name__)

model_names = config.model_names['model_names'].to_list()

quantities_time = [
                   "vres_curtailments",    # Dataframes, time/regions
                   "redispatch_vol",       # Dataframes, time/regions
                   "emissions",            # Dataframes, time/regions
                   "load_curtailments",    # Dataframes, time/regions

                   "storage_p",            # Dataframes, time/regions
                   "storage_soc",          # Dataframes, time/regions

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


def _zeros_df_t(year, index_name, level="market", columns=None, quantity=None):
    # return zeros dataframe
    if index_name == "snapshots":
        drop_hours = 1
        if calendar.isleap(year) and not config.leap_years:
            drop_hours = 25
        index = pd.date_range(str(year) + "-01-01", str(year + 1) + "-01-01", freq='h')[:-drop_hours]
    elif index_name == "carrier":
        index = config.carriers_all
    else:
        raise ValueError("index_name can only be either snapshots or carrier")
    if columns is None:
        if quantity == "import_export":
            columns = config.eu_neighs_conns
        else:
            columns = config.eu_neighs_ISO2['eu_neighs_ISO2']
            if level == "grid" and quantity != "energy_mix":
                columns = config.de_nuts1['de_nuts1_full_name']
    df = pd.DataFrame(np.zeros((len(index), len(columns))), index=index, columns=columns)
    df.index.name = index_name
    return df


class Calculator(object):

    """
    Toolset for a given scenario.

    Args:
        year (int): Year of the scenario.
        level (string): Level of the scenario. Can only be either "market" or "grid".
        scenario (string): Name of the scenario.
        data_source (string, default "csv"):
            Type of data source. Can only be either "csv" or "oep" for .csv and Open Energy Platform respectively.
        csv_path (string): Root path of the csv file structure.

    Example:
        >>> base = Calculator(2016, "market", "base")
    """

    def __repr__(self):
        return "%s %s %s %s" % ("Scenario", self.year, self.level, self.scenario)

    def _quantity_get_set(quantity):

        @property
        def quantity_prop(self):

            for model in model_names:

                if getattr(self, '_'+quantity)[model].empty:
                    logger.info(quantity + " data for " + model+" is empty. Loading from " + self.data_source)

                    if self.data_source == "csv":

                        data_path = os.path.join(self.data_path, model, quantity + ".csv")

                        if quantity in quantities_time:
                            index_name = "snapshots"
                        elif quantity in quantities_categorical:
                            index_name = "carrier"
                        else:
                            raise ValueError(quantity + " not in [" + ", ".join(quantities) + "]")

                        df_0 = _zeros_df_t(self.year, index_name, self.level, quantity=quantity)
                        try:
                            df = pd.read_csv(data_path, index_col=index_name, parse_dates=True).fillna(0.)
                            self.warning_flags.at[model, quantity] = "Ok."
                        except FileNotFoundError:
                            logger.error(quantity + " for model " + model + " was not found. Returning zeros.")
                            self.warning_flags.at[model, quantity] = "Missing file. Zeros"
                            df = _zeros_df_t(self.year, index_name, self.level, quantity=quantity)
                        except:
                            logger.error("Reading" + quantity + " for model " + model + " failed. Returning zeros.")
                            self.warning_flags.at[model, quantity] = "Missing file. Zeros"
                            df = _zeros_df_t(self.year, index_name, self.level, quantity=quantity)

                        # fit dataframe to desired format
                        if not df_0.index.difference(df.index).empty:
                            logger.warning("Missing " + index_name + " for " + quantity + " in model " + model +
                                           ". Filling with zeros.")
                            self.warning_flags.at[model, quantity] = "Partly broken."
                        if not df_0.columns.difference(df.columns).empty:
                            logger.warning("Missing columns for " + quantity + " in model " + model +
                                           ". Filling with zeros.")
                            self.warning_flags.at[model, quantity] = "Partly broken."
                        missing_cols = df_0.columns.difference(df.columns)
                        existing_cols = df_0.columns.intersection(df.columns)
                        df = pd.concat([df[existing_cols].reindex(df_0.index),
                                        _zeros_df_t(self.year, index_name, columns=missing_cols)],
                                       axis=1)

                        getattr(self, '_' + quantity).update({model: df.fillna(0)})

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

    storage_p = _quantity_get_set("storage_p")
    storage_soc = _quantity_get_set("storage_soc")

    import_export = _quantity_get_set("import_export")
    line_loadings = _quantity_get_set("line_loadings")

    electricity_prices = _quantity_get_set("electricity_prices")
    redispatch_cost = _quantity_get_set("redispatch_cost")

    energy_mix = _quantity_get_set("energy_mix")

    def __init__(self, year, level, scenario, data_source="csv", csv_path=None):

        logging.basicConfig(level=logging.INFO)
        logger.info("All methods assume hourly profiles.")

        self.year = year
        if level not in ["market", "grid"]:
            raise ValueError("level can only be either market or grid")
        self.level = level
        self.scenario = scenario
        if data_source not in ["csv", "oep"]:
            raise ValueError("data_source can only be either csv or oep")
        self.data_source = data_source

        if not csv_path:
            csv_path = os.path.join(os.path.dirname(__file__), "..", "data")
        if not os.path.isdir(csv_path):
            raise FileNotFoundError("directory "+csv_path+" was not found")

        self.data_path = ""
        if self.data_source == "csv":
            self.data_path = os.path.join(csv_path, str(self.year), self.level, str(self.scenario))
            if not os.path.isdir(self.data_path):
                raise FileNotFoundError("scenario folder "+self.scenario+" was not found")

        for quantity in quantities:
            setattr(self, quantity, dict(zip(model_names, [pd.DataFrame()]*len(model_names))))

        self.warning_flags = pd.DataFrame(index=model_names, columns=quantities).fillna("Unknown.")

        self.entsoe_mix = pd.read_csv(os.path.join(csv_path, "entso-e-energy-mix-modex.csv"),
                                      index_col='carrier').reindex(config.carriers_all)
        self.entsoe_factsheets_net_balance = pd.read_csv(os.path.join(csv_path,
                                                                      "entsoe_factsheets-net-balance-2016.csv"),
                                                         index_col='name')['imp-exp']
        self.entsoe_factsheets_net_exchanges = pd.read_csv(os.path.join(csv_path,
                                                                        "entsoe_factsheets-net-exchanges-2016.csv"),
                                                           index_col='name')['exchange']
        self.entsoe_day_ahead_prices = pd.read_csv(os.path.join(csv_path, "entsoe_day_ahead_prices_2016.csv"),
                                                   index_col='snapshots')
        # add zeros to missing countries
        for missing_col in ['NO', 'DK', 'SE']:
            self.entsoe_day_ahead_prices[missing_col] = 0.

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
                return plots.plot_dendrogram(getattr(self, func)(quantity, percent).transpose(), **kwargs)
            else:
                logger.warning("This function does not  accept a second argument. It is ignored.")
        return plots.plot_dendrogram(getattr(self, func)(quantity).transpose(), **kwargs)

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

    def energy_mix_indicator(self):

        def em_indicator(df1, df2):
            reg_indicator = (df1 - df2).abs().sum().divide(df1.sum())
            return (reg_indicator * df1.sum()).sum() / df1.sum().sum()

        return pd.concat([pd.Series([em_indicator(self.energy_mix[m1], self.energy_mix[m2]) for m2 in model_names],
                                    index=model_names)
                          for m1 in model_names], axis=1).rename(columns=dict(enumerate(model_names)))

    def plot_energy_mix(self, relative=False, aggregate=False, entsoe=True, title=None, ylabel="TWh", ylim=None, **kwargs):

        labels = [m for m in self.energy_mix.keys() if self.energy_mix[m].sum().sum()]
        dfs = [self.energy_mix[m].T.replace(',', '.', regex=True).astype(float) for m in labels]

        if entsoe and self.year == 2016:
            dfs.append(self.entsoe_mix.T)
            labels.append("ENTSO-E")

        if aggregate:
            agg_dict = config.aggregate_carriers.copy()
            dfs = [df.groupby(agg_dict, axis=1).sum() for df in dfs]
            dfs = [df[sorted(df.columns, key=lambda s:  [i for i, x in enumerate(config.carriers_reduced) if x == s])]
                   for df in dfs]  # sort carriers

        if relative:
            for i in range(len(dfs)):
                dfs[i] = dfs[i].T.divide(dfs[i].sum(axis=1)).T
            if ylabel:
                logger.warning("Removing ylabel for relative values.")
                ylabel = None
            ylim = 1.

        dfs = [df.reindex(dfs[0].index.sort_values()) for df in dfs]  # align indices by sorting them

        return plots.plot_clustered_stacked(dfs, labels=labels, title=title, ylabel=ylabel, ylim=ylim,
                                            **kwargs)

    def plot_heatmap(self, quantity, metric=None, model=None, **kwargs):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"

        if quantity == 'energy_mix':
            if metric: logger.warning("Metric argument is not used for energy mix. Ignoring it.")
            if model: logger.warning("Model argument is not used for energy mix. Ignoring it.")
            df = self.energy_mix_indicator()
            title = None
        else:
            assert metric in metrics.keys(), "Valid metrics can only be one of [" + ", ".join(metrics.keys()) + "]"
            assert model in model_names, "Valid model names can only be one of [" + ", ".join(model_names) + "]"
            df = self.pair_distance(quantity, metric)[model]
            title = model + " model"

        return plots.plot_heatmap(df, quantity=quantity, metric=metric, title=title, **kwargs)

    def price_convergence(self):

        from itertools import combinations
        import functools
        import operator

        region_countries = pd.Series([['DE', 'NL', 'BE', 'FR'],
                                      ['DE', 'PL', 'CZ', 'AT'],
                                      ['DK', 'NO', 'SE'],
                                      config.eu_neighs_ISO2['eu_neighs_ISO2'].to_list()],
                                     index=['CWE', '"CEE"', 'Nordic', 'All'])
        regional_convergence = pd.DataFrame(index=model_names + ['ENTSOE'], columns=region_countries.index)
        for model_name in model_names:
            for region in regional_convergence.columns:

                df = getattr(self, 'electricity_prices')[model_name]
                pairs = [((df[pair[0]] - df[pair[1]]).abs() < 0.01) for pair in
                         combinations(region_countries[region], 2)]
                regional_convergence.loc[model_name, region] = functools.reduce(operator.and_, pairs).sum() / 87.6

                df = self.entsoe_day_ahead_prices
                pairs = [((df[pair[0]] - df[pair[1]]).abs() < 0.01) for pair in
                         combinations(region_countries[region], 2)]
                regional_convergence.loc['ENTSOE', region] = functools.reduce(operator.and_, pairs).sum() / 87.6

        interconn_convergence = pd.DataFrame(index=config.eu_neighs_conns, columns=model_names)
        for model_name in interconn_convergence.columns:
            df = getattr(self, 'electricity_prices')[model_name]
            for conn in interconn_convergence.index:
                interconn_convergence.loc[conn, model_name] = (df[conn[:2]] - df[conn[-2:]]).abs().mean()
            df = self.entsoe_day_ahead_prices
            for conn in interconn_convergence.index:
                interconn_convergence.loc[conn, 'ENTSOE'] = (df[conn[:2]] - df[conn[-2:]]).abs().mean()

        return {'regional convergence': regional_convergence,
                'interconnection convergence': interconn_convergence}




