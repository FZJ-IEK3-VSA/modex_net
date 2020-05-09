# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import pandas as pd

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
                   "vres_curtailments",    # Dataframes, regions/time
                   "redispatch_vol",       # Dataframes, regions/time
                   "emissions",            # Dataframes, regions/time
                   "load_curtailments",    # Dataframes, regions/time

                   "storage_discharging",  # Dataframes, regions/time
                   "storage_charging",     # Dataframes, regions/time

                   "exports",              # Dataframes, interconnections/time
                   "imports",              # Dataframes, interconnections/time
                   "line_loadings",        # Dataframes, lines/time

                   "electricity_prices",   # Dataframes, regions/time
                   "redispatch_cost"       # Dataframes, regions/time
                  ]
quantities_categorical = ["energy_mix"]    # Dataframes, regions/carrier
quantities = quantities_time + quantities_categorical

class Calculator:

    def quantity_get_set(quantity):

        @property
        def quantity_prop(self):

            for model in model_names:

                if getattr(self, '_'+quantity)[model].empty:
                    logger.info(str(quantity)+" data for "+str(model)+" is empty. Loading from "+str(self.data_source))

                    if self.data_source == "csv":

                        if quantity in quantities_time:
                            getattr(self, '_'+quantity).update({model: pd.read_csv("data/" + str(self.scenario) + "/"
                                                                                   + quantity + "/" + model + ".csv",
                                                                                   index_col='snapshots',
                                                                                   parse_dates=True)})
                        if quantity in quantities_categorical:
                            getattr(self, quantity).update({model: pd.read_csv("data/" + str(self.scenario) + "/"
                                                                               + quantity + "/" + model + ".csv",
                                                                               index_col='index')})

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
    vres_curtailments = quantity_get_set("vres_curtailments")
    redispatch_vol = quantity_get_set("redispatch_vol")
    emissions = quantity_get_set("emissions")
    load_curtailments = quantity_get_set("load_curtailments")

    storage_discharging = quantity_get_set("storage_discharging")
    storage_charging = quantity_get_set("storage_charging")

    exports = quantity_get_set("exports")
    imports = quantity_get_set("imports")
    line_loadings = quantity_get_set("line_loadings")

    electricity_prices = quantity_get_set("electricity_prices")
    redispatch_cost = quantity_get_set("redispatch_cost")

    def __init__(self, scenario, data_source="csv"):

        logging.basicConfig(level=logging.INFO)
        logger.info("All methods assume hourly profiles.")

        self.scenario = scenario
        self.data_source = data_source

        for quantity in quantities:
            setattr(self, quantity, dict(zip(model_names, [pd.DataFrame()]*len(model_names))))

    def sum(self, quantity):

        assert quantity in quantities, "\'"+str(quantity)+"\'"+" is not a valid quantity to measure."

        return (pd.concat([getattr(self, quantity)[model].sum() for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def normalized_std(self, quantity):

        assert quantity in quantities, "\'"+str(quantity)+"\'"+" is not a valid quantity to measure."

        return (pd.concat([getattr(self, quantity)[model].std()/getattr(self, quantity)[model].mean()
                           for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def percentile(self, quantity, percent=0.75):

        assert quantity in quantities, "\'"+str(quantity)+"\'"+" is not a valid quantity to measure."

        return (pd.concat([getattr(self, quantity)[model].quantile(percent).rename(None)
                           for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))

    def percentile_converter(self, quantity, percent=0.75):

        assert quantity in quantities, "\'"+str(quantity)+"\'"+" is not a valid quantity to measure."

        converter_capacities = self.percentile(quantity, percent)

        return (pd.concat([getattr(self, quantity)[model].clip(0, converter_capacities[model], axis=1).sum()
                           for model in model_names], axis=1)
                .rename(columns=dict(enumerate(model_names))))


