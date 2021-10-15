# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import os
import pandas as pd
import numpy as np
import calendar

from . import plots, config, metrics, operators
from progressbar import ProgressBar
from modex_net_oep import ModexClient

import logging
logger = logging.getLogger(__name__)

model_names = config.model_names['model_names'].to_list()

quantities_time = [
                   "vres_curtailments",    # Dataframes, time/regions
                   "emissions",            # Dataframes, time/regions
                   "load_curtailments",    # Dataframes, time/regions

                   "storage_p",            # Dataframes, time/regions
                   "storage_soc",          # Dataframes, time/regions

                   "import_export",        # Dataframes, time/interconnections

                   "electricity_prices"   # Dataframes, time/regions
                  ]
quantities_categorical = ["energy_mix"]    # Dataframes, carrier/regions
quantities = quantities_time + quantities_categorical


metrics_dict = metrics.metrics_dict

operators_dict = operators.operators_dict


def _zeros_df_t(year, index_name, level="market", columns=None, quantity=None, data_source="csv", **kwargs):
    """Returns the appropriate dataframe template filled with zeros."""

    if data_source == "oep":
        cli = ModexClient(**kwargs)
    if index_name == "snapshots":
        drop_hours = 1
        if calendar.isleap(year) and not config.leap_years:
            drop_hours = 25
        index = pd.date_range(str(year) + "-01-01", str(year + 1) + "-01-01", freq='h')[:-drop_hours]
    elif index_name == "carrier":
        if data_source == "csv":
            index = config.carriers_all
        elif data_source == "oep":
            index = pd.DataFrame.from_records(cli.select_table("mn_dimension_carrier"))['carrier'].tolist()
    else:
        raise ValueError("index_name can only be either snapshots or carrier")
    if columns is None:
        if quantity == "import_export":
            columns = config.eu_neighs_conns[year]
        else:
            if data_source == "csv":
                columns = config.eu_neighs_ISO2['eu_neighs_ISO2']
            elif data_source == "oep":
                columns = pd.DataFrame.from_records(cli.select_table("mn_dimension_country"))['country_code'].tolist()
            if level == "grid" and quantity != "energy_mix":
                if data_source == "csv":
                    columns = config.de_nuts1['de_nuts1_full_name']
                elif data_source == "oep":
                    columns = pd.DataFrame.from_records(cli.select_table("mn_dimension_region"))['region_code'].tolist()
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

            pbar = ProgressBar()
            for model in pbar(model_names):

                if getattr(self, '_'+quantity)[model].empty:
                    logger.info(quantity + " data for " + model+" is empty. Loading from " + self.data_source)

                    if quantity in quantities_time:
                        index_name = "snapshots"
                    elif quantity in quantities_categorical:
                        index_name = "carrier"
                    else:
                        raise ValueError(quantity + " not in [" + ", ".join(quantities) + "]")

                    df_0 = _zeros_df_t(self.year, index_name, self.level, quantity=quantity)

                    if self.data_source == "csv":

                        data_path = os.path.join(self.csv_path, model, quantity + ".csv")

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

                    elif self.data_source == "oep":
                        try:
                            data = self.cli.download_parameter(self.year, self.level, self.scenario, quantity, model)
                            df = pd.DataFrame.from_records(data)
                            if quantity == "energy_mix":
                                df = df.pivot_table(index='carrier', columns='country_code', values='value')
                            elif quantity == "import_export":
                                df['from_to'] = df['country_code_from'] + '_' + df['country_code_to']
                                from_to_sorted = df[['country_code_from', 'country_code_to']].apply(np.sort, axis=1)
                                df['from_to_sorted'] = from_to_sorted.apply(
                                    lambda x: x[0]) + '_' + from_to_sorted.apply(lambda x: x[1])
                                inverse_flows = df.from_to_sorted != df.from_to
                                df.loc[inverse_flows, 'value'] *= -1
                                df = df.pivot_table(index='hour', columns='from_to_sorted', values='value')
                                df.index = df.index.map(self._time_map)
                            else:
                                df = df.pivot_table(index='hour', columns='country_code', values='value')
                                df.index = df.index.map(self._time_map)
                            df.index.name = index_name
                            self.warning_flags.at[model, quantity] = "Ok."
                        except:
                            logger.error("Table for " + model + " could not be retrieved. Returning zeros.")
                            self.warning_flags.at[model, quantity] = "Missing table. Zeros"
                            df = _zeros_df_t(self.year, index_name, self.level, quantity=quantity)

                    else:
                        raise NotImplementedError("Available data sources: csv, oep")

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

    def __init__(self, year, level, scenario, data_source="csv", data_path=None,
                 oep_token="", oep_modex_user="c.syranidou@fz-juelich.de"):

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

        if level == "market":
            self.regions = config.eu_neighs_ISO2['eu_neighs_ISO2'].to_list()
        else:
            self.regions = config.de_nuts1['de_nuts1_full_name'].to_list()

        if self.data_source == "csv":
            if not data_path:
                data_path = os.path.join(os.path.dirname(__file__), "..", "data")
            if not os.path.isdir(data_path):
                raise FileNotFoundError("directory "+data_path+" was not found")

            self.csv_path = os.path.join(data_path, str(self.year), self.level, str(self.scenario))
            if not os.path.isdir(self.csv_path):
                raise FileNotFoundError("scenario folder "+self.scenario+" was not found")

        if data_source == "oep":
            self.oep_token = oep_token
            self.oep_user = oep_modex_user
            self.cli = ModexClient(token=self.oep_token, user=self.oep_user)
            self._time_map = pd.to_datetime(
                pd.DataFrame.from_records(self.cli.select_table("mn_dimension_time"))
                .set_index('hour')[f'timestamp_{self.year}'])

        for quantity in quantities:
            setattr(self, quantity, dict(zip(model_names, [pd.DataFrame()]*len(model_names))))

        self.warning_flags = pd.DataFrame(index=model_names, columns=quantities).fillna("Unknown.")

        data_path = os.path.join(os.path.dirname(__file__), "..", "data")
        self.entsoe_mix = pd.read_csv(os.path.join(data_path, "entso-e-energy-mix-modex.csv"),
                                      index_col='carrier').reindex(config.carriers_all)
        self.entsoe_factsheets_net_balance = pd.read_csv(os.path.join(data_path,
                                                                      "entsoe_factsheets-net-balance-2016.csv"),
                                                         index_col='name')['imp-exp']
        self.entsoe_factsheets_net_exchanges = pd.read_csv(os.path.join(data_path,
                                                                        "entsoe_factsheets-net-exchanges-2016.csv"),
                                                           index_col='name')['exchange']
        self.entsoe_day_ahead_prices = self._read_entsoe_prices(data_path)
        self.eea_emissions = self._read_eea_emissions(data_path)
        # add zeros to missing countries
        for missing_col in ['NO', 'DK', 'SE']:
            self.entsoe_day_ahead_prices[missing_col] = 0.

    @staticmethod
    def _read_entsoe_prices(data_path):
        df = pd.read_csv(os.path.join(data_path, "entsoe_day_ahead_prices_2016.csv"),
                         index_col='snapshots', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.index = df.index + pd.DateOffset(hours=1)
        return df

    def _read_eea_emissions(self, data_path):
        to_iso2 = config.eu_neighs_ISO2.set_index('eu_neighs_full_name')['eu_neighs_ISO2']
        df = pd.read_csv(os.path.join(data_path, 'co2-emission-intensity-from-electricity-generation-2.csv'))
        df = df[df['date:number'] == 2016]
        df['ugeo:text'] = df['ugeo:text'].replace(to_iso2)
        df = df.set_index('ugeo:text').reindex(to_iso2)
        return (df['obsValue:number'].multiply(1e-3) * self.entsoe_mix.sum()).fillna(0)

    def reduction(self, quantity, operator, percent=0.75, entsoe=True):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert operator in operators_dict, "Valid quantities to measure can only be one of [" + ", ".join(operators_dict.keys()) + "]"
        if operator.startswith('percentile'):
            assert 0. <= percent <= 1., "Percent should be in the [0, 1] domain."

        dfs = getattr(self, quantity)
        models = model_names.copy()
        if self.year == 2016 and quantity == 'electricity_prices' and entsoe:
            dfs['ENTSOE'] = self.entsoe_day_ahead_prices
            models = models + ['ENTSOE']
        return (pd.concat([operators_dict[operator](dfs[model], percent) if operator.startswith('percentile') else
                           operators_dict[operator](dfs[model])
                           for model in models], axis=1)
                .rename(columns=dict(enumerate(models))))

    def plot_dendrogram(self, quantity, func, percent=None, entsoe=False, **kwargs):

        return plots.plot_dendrogram(self.reduction(quantity, func, percent, entsoe).transpose(), **kwargs)

    def pair_distance_single(self, quantity, metric, model):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert metric in metrics_dict.keys(), "Valid metrics can only be one of [" + ", ".join(metrics_dict.keys()) + "]"

        dfs = getattr(self, quantity)
        return (pd.concat([(dfs[model]
                            .combine(dfs[mod], metrics_dict[metric])
                            .iloc[0]
                            .rename(None))
                           for mod in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def pair_distance(self, quantity, metric):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert metric in metrics_dict.keys(), "Valid metrics can only be one of [" + ", ".join(metrics_dict.keys()) + "]"

        return {model: self.pair_distance_single(quantity, metric, model) for model in model_names}

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

        energy_mix = self.energy_mix
        def em_indicator(df1, df2):
            reg_indicator = (df1 - df2).abs().sum().divide(df1.sum())
            return (reg_indicator * df1.sum()).sum() / df1.sum().sum()

        return pd.concat([pd.Series([em_indicator(energy_mix[m1], energy_mix[m2]) for m2 in model_names],
                                    index=model_names)
                          for m1 in model_names], axis=1).rename(columns=dict(enumerate(model_names)))

    def plot_energy_mix(self, relative=False, aggregate=False, entsoe=True, title=None, ylabel="TWh", ylim=None,
                        savefig=None, dpi=300, **kwargs):

        energy_mix = self.energy_mix
        labels = [m for m in energy_mix.keys() if energy_mix[m].sum().sum()]
        dfs = [energy_mix[m].T.replace(',', '.', regex=True).astype(float) for m in labels]

        if entsoe and self.year == 2016:
            dfs.append(self.entsoe_mix.T)
            labels.append("ENTSO-E")

        carriers = config.carriers_all
        if aggregate:
            agg_dict = config.aggregate_carriers.copy()
            dfs = [df.groupby(agg_dict, axis=1).sum() for df in dfs]
            carriers = config.carriers_reduced
        # sort carriers
        dfs = [df[sorted(df.columns, key=lambda s: [i for i, x in enumerate(carriers) if x == s])] for df in dfs]

        if relative:
            for i in range(len(dfs)):
                dfs[i] = dfs[i].T.divide(dfs[i].sum(axis=1)).T
            if ylabel:
                logger.warning("Removing ylabel for relative values.")
                ylabel = None
            ylim = 1.

        dfs = [df.reindex(dfs[0].index.sort_values()) for df in dfs]  # align indices by sorting them

        return plots.plot_clustered_stacked(dfs, labels=labels, title=title, ylabel=ylabel, ylim=ylim,
                                            savefig=savefig, dpi=dpi, **kwargs)

    def plot_heatmap(self, quantity, metric=None, model=None, **kwargs):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"

        if quantity == 'energy_mix':
            if metric: logger.warning("Metric argument is not used for energy mix. Ignoring it.")
            if model: logger.warning("Model argument is not used for energy mix. Ignoring it.")
            df = self.energy_mix_indicator()
            title = None
        else:
            assert metric in metrics_dict.keys(), "Valid metrics can only be one of [" + ", ".join(metrics_dict.keys()) + "]"
            assert model in model_names, "Valid model names can only be one of [" + ", ".join(model_names) + "]"
            df = self.pair_distance_single(quantity, metric, model)
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

        def pairs(d, reg):
            return [((d[pair[0]] - d[pair[1]]).abs() < 0.01) for pair in combinations(region_countries[reg], 2)]

        prices = self.electricity_prices
        regional_convergence = pd.DataFrame(index=model_names + ['ENTSOE'], columns=region_countries.index)
        for model_name in model_names:
            for region in regional_convergence.columns:

                df = prices[model_name]
                regional_convergence.loc[model_name, region] = functools.reduce(operator.and_,
                                                                                pairs(df, region)).sum() / 87.6

                if self.year == 2016:
                    df = self.entsoe_day_ahead_prices
                    regional_convergence.loc['ENTSOE', region] = functools.reduce(operator.and_,
                                                                                  pairs(df, region)).sum() / 87.6
                    if region == 'Nordic':
                        regional_convergence.loc['ENTSOE', region] = 0.

        interconn_convergence = pd.DataFrame(index=config.eu_neighs_conns[self.year], columns=model_names)
        for model_name in interconn_convergence.columns:
            df = prices[model_name]
            for conn in interconn_convergence.index:
                interconn_convergence.loc[conn, model_name] = (df[conn[:2]] - df[conn[-2:]]).abs().mean()
            if self.year == 2016:
                df = self.entsoe_day_ahead_prices
                for conn in interconn_convergence.index:
                    interconn_convergence.loc[conn, 'ENTSOE'] = (df[conn[:2]] - df[conn[-2:]]).abs().mean()
        if 'ENTSOE' in interconn_convergence.columns:
            nordic_conns = ['DE_DK', 'DE_NO', 'DE_SE', 'DK_NL', 'DK_NO', 'DK_SE', 'NL_NO', 'NO_SE', 'PL_SE']
            interconn_convergence.loc[interconn_convergence.index.intersection(nordic_conns), 'ENTSOE'] = 0

        return {'regional convergence': regional_convergence,
                'interconnection convergence': interconn_convergence}

    def net_balances(self):
        imex = self.import_export
        net_balances = pd.DataFrame(columns=model_names, index=self.regions)
        for model in net_balances.columns:
            df = imex[model]
            for country in net_balances.index:
                conns_export = [i for i in df.columns if i.split('_')[0] == country]
                conns_import = [i for i in df.columns if i.split('_')[1] == country]
                net_balances.at[country, model] = df[conns_import].sum().sum() - df[conns_export].sum().sum()
        if self.year == 2016:
            net_balances['ENTSOE'] = self.entsoe_factsheets_net_balance
        net_balances.index.name = ''
        return net_balances

    def plot_taylor_diagram(self, quantity, col, reference, **kwargs):

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        if quantity == "import_export":
            conns = config.eu_neighs_conns[self.year]
            assert col in conns, "Valid interconnections can only be one of [" + ", ".join(conns) + "]"
        else:
            assert col in self.regions, "Valid regions can only be one of [" + ", ".join(self.regions) + "]"
        if self.year == 2016 and quantity == 'electricity_prices':
            assert reference in model_names+['ENTSOE'], "Valid references can only be one of [" + ", ".join(model_names+['ENTSOE']) + "]"
        else:
            assert reference in model_names, "Valid references can only be one of [" + ", ".join(model_names) + "]"

        names = model_names.copy()
        if reference == 'ENTSOE':
            assert col not in ['NO', 'SE', 'DK'], "No ENTSOE prices for ['NO', 'SE', 'DK'])"
            x0 = self.entsoe_day_ahead_prices[col].iloc[:8760]
        else:
            x0 = getattr(self, quantity)[reference][col]
            names.remove(reference)

        dfs = getattr(self, quantity)
        predictions = [dfs[model][col] for model in names]

        return plots.taylor_diagram(predictions, x0, names, **kwargs)




