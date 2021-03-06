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
from oep_client import OepClient

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

                   "electricity_prices"    # Dataframes, time/regions
                  ]
quantities_categorical = ["energy_mix"]    # Dataframes, carrier/regions
quantities = quantities_time + quantities_categorical


metrics_dict = metrics.metrics_dict

operators_dict = operators.operators_dict


def _zeros_df_t(year, index_name, level="market", columns=None, quantity=None, data_source="csv", token=""):
    """Returns the appropriate dataframe template filled with zeros."""

    if data_source == "oep":
        cli = OepClient(token=token)
    if index_name == "snapshots":
        drop_hours = 1
        if calendar.isleap(year) and not config.leap_years:
            drop_hours = 25
        index = pd.date_range(str(year) + "-01-01", str(year + 1) + "-01-01", freq='h')[:-drop_hours]
    elif index_name == "carrier":
        if data_source == "csv":
            index = config.carriers_all
        elif data_source == "oep":
            index = config.carriers_all
            #index = pd.DataFrame.from_records(cli.select_from_table("modexnet_dimension_carrier"))['carrier'].tolist()
    else:
        raise ValueError("index_name can only be either snapshots or carrier")
    if columns is None:
        if quantity == "import_export":
            columns = config.eu_neighs_conns[year]
        else:
            if data_source == "csv":
                columns = config.eu_neighs_ISO2['eu_neighs_ISO2']
            elif data_source == "oep":
                columns = config.eu_neighs_ISO2['eu_neighs_ISO2']
                #columns = pd.DataFrame.from_records(cli.select_from_table("modexnet_dimension_country"))['country_code'].tolist()
            if level == "grid" and quantity != "energy_mix":
                if data_source == "csv":
                    columns = config.de_nuts1['de_nuts1_full_name']
                elif data_source == "oep":
                    columns = pd.DataFrame.from_records(cli.select_from_table("modexnet_dimension_region"))['region_code'].tolist()
    df = pd.DataFrame(np.zeros((len(index), len(columns))), index=index, columns=columns)
    df.index.name = index_name
    return df


class Calculator(object):

    """
    Toolset for generating indicators and visualizations for comparing the market results of power system models.

    Args:
        year (int): Year of the scenario.
        level (string): Level of the scenario. Can only be either "market" or "grid". Defaults to "market".
        scenario (string): Name of the scenario, optional.
        data_source (string, default "csv"):
            Type of data source. Can only be either "csv" or "oep" for .csv and Open Energy Platform respectively.
            Defaults to "csv".
        data_path (string): Root path of the csv file structure, optional.
        oep_token (string): The API token for the Open Energy Platform, optional.
        entsoe (boolean): Whether to include ENTSO-E data in the comparison. It requires the necessary files. Defaults to False.
        entsoe_path (string): The path for the ENTSO-E files, optional.

    Example:
        >>> base = Calculator(2016, "market", "base")
    """

    def __repr__(self):
        return "%s %s %s %s" % ("Scenario", self.year, self.level, self.scenario)

    def _quantity_get_set(quantity):

        @property
        def quantity_prop(self):

            logger.info("Retrieving " + quantity)

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

                    df_0 = _zeros_df_t(self.year, index_name, self.level, quantity=quantity,
                                       data_source=self.data_source, token=self.oep_token)

                    if self.data_source == "csv":

                        data_path = os.path.join(self.csv_path, model, quantity + ".csv")

                        try:
                            df = pd.read_csv(data_path, index_col=index_name, parse_dates=True).fillna(0.)
                            self.warning_flags.at[model, quantity] = "Ok."
                        except FileNotFoundError:
                            logger.error(quantity + " for model " + model + " was not found. Returning zeros.")
                            self.warning_flags.at[model, quantity] = "Missing file. Zeros"
                            df = _zeros_df_t(self.year, index_name, self.level, quantity=quantity,
                                             data_source=self.data_source, token=self.oep_token)
                        except:
                            logger.error("Reading" + quantity + " for model " + model + " failed. Returning zeros.")
                            self.warning_flags.at[model, quantity] = "Missing table. Zeros"
                            df = _zeros_df_t(self.year, index_name, self.level, quantity=quantity,
                                             data_source=self.data_source, token=self.oep_token)

                    elif self.data_source == "oep":
                        table_name = f'modexnet_{model}_{self.year}_{self.level}_{quantity}'
                        try:
                            data = self.cli.select_from_table(table_name)
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
                            df = _zeros_df_t(self.year, index_name, self.level, quantity=quantity,
                                             data_source=self.data_source, token=self.oep_token)

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
                                    _zeros_df_t(self.year, index_name, columns=missing_cols,
                                                data_source=self.data_source, token=self.oep_token)],
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

    def __init__(self, year, level="market", scenario=None, data_source="csv", data_path=None, oep_token="", entsoe=False,
                 entsoe_path=""):

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
            self.cli = OepClient(token=self.oep_token)
            self._time_map = pd.to_datetime(
                pd.DataFrame.from_records(self.cli.select_from_table("modexnet_dimension_time"))
                .set_index('hour')[f'timestamp_{self.year}'])

        for quantity in quantities:
            setattr(self, quantity, dict(zip(model_names, [pd.DataFrame()]*len(model_names))))

        self.warning_flags = pd.DataFrame(index=model_names, columns=quantities).fillna("Unknown.")

        if entsoe_path:
            self.entsoe_path = entsoe_path
        else:
            self.entsoe_path = os.path.join(os.path.dirname(__file__), "..", "data")

        if entsoe:
            self.entsoe = True
        else:
            self.entsoe = False

    def _entsoe_mix(self):
        return pd.read_csv(os.path.join(self.entsoe_path, "entso-e-energy-mix-modex.csv"),
                           index_col='carrier').reindex(config.carriers_all)

    def _entsoe_factsheets_net_balance(self):
        return pd.read_csv(os.path.join(self.entsoe_path, "entsoe_factsheets-net-balance-2016.csv"),
                           index_col='name')['imp-exp']

    def _entsoe_factsheets_net_exchanges(self):
        return pd.read_csv(os.path.join(self.entsoe_path, "entsoe_factsheets-net-exchanges-2016.csv"),
                           index_col='name')['exchange']

    def _entsoe_day_ahead_prices(self):
        df = pd.read_csv(os.path.join(self.entsoe_path, "entsoe_day_ahead_prices_2016.csv"),
                         index_col='snapshots', parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.index = df.index + pd.DateOffset(hours=1)

        for missing_col in ['NO', 'DK', 'SE']:
            df[missing_col] = 0.
        return df

    def _eea_emissions(self):
        to_iso2 = config.eu_neighs_ISO2.set_index('eu_neighs_full_name')['eu_neighs_ISO2']
        df = pd.read_csv(os.path.join(elf.entsoe_path, 'co2-emission-intensity-from-electricity-generation-2.csv'))
        df = df[df['date:number'] == 2016]
        df['ugeo:text'] = df['ugeo:text'].replace(to_iso2)
        df = df.set_index('ugeo:text').reindex(to_iso2)
        return (df['obsValue:number'].multiply(1e-3) * self._entsoe_mix().sum()).fillna(0)

    def reduction(self, quantity, operator, percent=0.75):
        """
        Returns dataframe with the values of the selected quantity reduced by the given operator for all models and
        regions.

        Args:
            quantity (string): Selected quantity. Must be one of the available quanities in 'indicators.quantities'.
            operator (string): Time series reduction method. Must be one of the available operators in
            'indicators.operators_dict.keys()
            percent (float): Percentile value for the application of the 'percentile' and 'percentile_converter'
            operators. Must be between [0, 1]. Optional

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> df = base.reduction("electricity_prices", "mean")
        """

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert operator in operators_dict, "Valid quantities to measure can only be one of [" + ", ".join(operators_dict.keys()) + "]"
        if operator.startswith('percentile'):
            assert 0. <= percent <= 1., "Percent should be in the [0, 1] domain."

        dfs = getattr(self, quantity)
        models = model_names.copy()
        if self.year == 2016 and quantity == 'electricity_prices' and self.entsoe:
            dfs['ENTSOE'] = self._entsoe_day_ahead_prices()
            models = models + ['ENTSOE']
        return (pd.concat([operators_dict[operator](dfs[model], percent) if operator.startswith('percentile') else
                           operators_dict[operator](dfs[model])
                           for model in models], axis=1)
                .rename(columns=dict(enumerate(models))))

    def plot_dendrogram(self, quantity, operator, percent=None, **kwargs):
        """
        Returns dendrogram plot for the selected quantity reduced by the given operator for all models and
        regions.

        Args:
            quantity (string): Selected quantity. Must be one of the available quanities in 'indicators.quantities'.
            operator (string): Time series reduction method. Must be one of the available operators in
            'indicators.operators_dict.keys()
            percent (float): Percentile value for the application of the 'percentile' and 'percentile_converter'
            operators. Must be between [0, 1]. Optional

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> ax = base.plot_dendrogram("electricity_prices", "mean")
        """

        return plots.plot_dendrogram(self.reduction(quantity, operator, percent).transpose(), **kwargs)

    def pair_distance_single(self, quantity, metric, model):
        """
        Returns dataframe with the distances of the selected model to all other models for the given quantity measured
        by the given metric.

        Args:
            quantity (string): Selected quantity. Must be one of the available quanities in 'indicators.quantities'.
            metric (string): Distance metric. Must be one of the available metrics in 'indicators.metrics_dict.keys().
            model (string): Model name. Must be one of the indicators.model_names.

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> df = base.pair_distance_single("electricity_prices", "correlation", "europower")
        """

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
        """
        Returns a dicitionary of dataframes with the distances of all models to all other models for the given quantity
        measured by the given metric.

        Args:
            quantity (string): Selected quantity. Must be one of the available quanities in 'indicators.quantities'.
            metric (string): Distance metric. Must be one of the available metrics in 'indicators.metrics_dict.keys().

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> dfs = base.pair_distance("electricity_prices", "correlation")
        """

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        assert metric in metrics_dict.keys(), "Valid metrics can only be one of [" + ", ".join(metrics_dict.keys()) + "]"

        return {model: self.pair_distance_single(quantity, metric, model) for model in model_names}

    def energy_mix_indicator(self):
        """
        Returns dataframe with the error-like indicator for the energy mix for all pairs of models. It is computed as
        the average of the differences of the energy mix weighted over the fuel types and regions.

        Args:

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> df = base.energy_mix_indicator()
        """

        energy_mix = self.energy_mix
        def em_indicator(df1, df2):
            reg_indicator = (df1 - df2).abs().sum().divide(df1.sum())
            return (reg_indicator * df1.sum()).sum() / df1.sum().sum()

        return pd.concat([pd.Series([em_indicator(energy_mix[m1], energy_mix[m2]) for m2 in model_names],
                                    index=model_names)
                          for m1 in model_names], axis=1).rename(columns=dict(enumerate(model_names)))

    def plot_energy_mix(self, relative=False, aggregate=False, title=None, ylabel="TWh", ylim=None,
                        savefig=None, dpi=300, **kwargs):
        """
        Returns stacked bar chart of the energy mix for all models and regions.

        Args:
            relative (boolean): Plot in relative (to the total production) values.
            aggregate (boolean): Aggregate fuel types based on the 'config.aggregate_carriers' dictionary.
            title (string): Plot title.
            ylabel (string): Y axis label.
            ylim (tuple): Min and max values of the y axis.
            savefig (string): File name for the saved figure.
            dpi (float): Resolution of the saved figure.

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> ax = base.plot_energy_mix()
        """

        energy_mix = self.energy_mix
        labels = [m for m in energy_mix.keys() if energy_mix[m].sum().sum()]
        dfs = [energy_mix[m].T.replace(',', '.', regex=True).astype(float) for m in labels]

        if self.entsoe and self.year == 2016:
            dfs.append(self._entsoe_mix().T)
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
        """
        Returns heatmap plot with the distances of the selected model to all other models for the given quantity
        measured by the given metric. If quantity == "energy_mix", the heatmap represents the 'energy_mix_indicator' for
        all pairs of models.

        Args:
            quantity (string): Selected quantity. Must be one of the available quanities in 'indicators.quantities'.
            metric (string): Distance metric. Must be one of the available metrics in 'indicators.metrics_dict.keys().
            model (string): Model name. Must be one of the indicators.model_names.

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> ax = base.plot_heatmap("electricity_prices", "correlation", "europower")
        """

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
        """
        Returns dataframes with the 'regional convergence' and 'interconnection convergence' values for all models.

        The 'regional convergence' represents the percentage of full price convergence over all countries inside a
        region. The regions include the 'CWE', 'CEE', 'Nordic' and 'All' regions as defined by ENTSO-E, where the
        respective countries are also part of the electrical neighbors of Germany, as defined in 'config.eu_neighs_ISO2'

        The 'interconnection convergence' represents the mean difference of electricity prices over all interconnections
        as defined in 'config.eu_neighs_conns'.

        Args:

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> reg_conv, inter_conv = base.price_convergence()
        """

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

                if self.year == 2016 and self.entsoe:
                    df = self._entsoe_day_ahead_prices()
                    regional_convergence.loc['ENTSOE', region] = functools.reduce(operator.and_,
                                                                                  pairs(df, region)).sum() / 87.6
                    if region == 'Nordic':
                        regional_convergence.loc['ENTSOE', region] = 0.

        interconn_convergence = pd.DataFrame(index=config.eu_neighs_conns[self.year], columns=model_names)
        for model_name in interconn_convergence.columns:
            df = prices[model_name]
            for conn in interconn_convergence.index:
                interconn_convergence.loc[conn, model_name] = (df[conn[:2]] - df[conn[-2:]]).abs().mean()
            if self.year == 2016 and self.entsoe:
                df = self._entsoe_day_ahead_prices()
                for conn in interconn_convergence.index:
                    interconn_convergence.loc[conn, 'ENTSOE'] = (df[conn[:2]] - df[conn[-2:]]).abs().mean()
        if 'ENTSOE' in interconn_convergence.columns:
            nordic_conns = ['DE_DK', 'DE_NO', 'DE_SE', 'DK_NL', 'DK_NO', 'DK_SE', 'NL_NO', 'NO_SE', 'PL_SE']
            interconn_convergence.loc[interconn_convergence.index.intersection(nordic_conns), 'ENTSOE'] = 0

        return {'regional convergence': regional_convergence,
                'interconnection convergence': interconn_convergence}

    def net_balances(self):
        """
        Returns dataframe with the net balances of all models and regions. Positive values correspond to net imports.

        Args:

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> net_balances = base.net_balances()
        """

        imex = self.import_export
        net_balances = pd.DataFrame(columns=model_names, index=self.regions)
        for model in net_balances.columns:
            df = imex[model]
            for country in net_balances.index:
                conns_export = [i for i in df.columns if i.split('_')[0] == country]
                conns_import = [i for i in df.columns if i.split('_')[1] == country]
                net_balances.at[country, model] = df[conns_import].sum().sum() - df[conns_export].sum().sum()
        if self.year == 2016 and self.entsoe:
            net_balances['ENTSOE'] = self._entsoe_factsheets_net_balance()
        net_balances.index.name = ''
        return net_balances

    def plot_taylor_diagram(self, quantity, region, reference_model, **kwargs):
        """
        Returns taylor diagram plot for the given quantity (must be a time series), selected region or interconnector
        and given model as reference.

        Args:
            quantity (string): Selected quantity. Must be one of the available quanities in 'indicators.quantities'.
            region (string): Either region name from 'config.eu_neighs_ISO2' or interconnector name from
            'config.eu_neighs_conns' for the 'import_export' quantity.
            reference_model (string): Model name. Must be one of the indicators.model_names.

        Example:
            >>> base = Calculator(2016, data_source="oep", oep_token="")
            >>> ax = base.plot_taylor_diagram("electricity_prices", "DE", "europower")
        """

        assert quantity in quantities, "Valid quantities to measure can only be one of [" + ", ".join(quantities) + "]"
        if quantity == "import_export":
            conns = config.eu_neighs_conns[self.year]
            assert region in conns, "Valid interconnections can only be one of [" + ", ".join(conns) + "]"
        else:
            assert region in self.regions, "Valid regions can only be one of [" + ", ".join(self.regions) + "]"
        if self.year == 2016 and quantity == 'electricity_prices' and self.entsoe:
            assert reference_model in model_names+['ENTSOE'], "Valid references can only be one of [" + ", ".join(model_names+['ENTSOE']) + "]"
        else:
            assert reference_model in model_names, "Valid references can only be one of [" + ", ".join(model_names) + "]"

        names = model_names.copy()
        if reference_model == 'ENTSOE':
            assert region not in ['NO', 'SE', 'DK'], "No ENTSOE prices for ['NO', 'SE', 'DK'])"
            x0 = self._entsoe_day_ahead_prices()[col].iloc[:8760]
        else:
            x0 = getattr(self, quantity)[reference_model][region]
            names.remove(reference_model)

        dfs = getattr(self, quantity)
        predictions = [dfs[model][region] for model in names]

        return plots.taylor_diagram(predictions, x0, names, **kwargs)




