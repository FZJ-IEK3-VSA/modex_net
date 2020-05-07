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


class Calculator:

    def __init__(self, scenario):

        logging.basicConfig(level=logging.INFO)

        self.scenario = scenario

        self.model_names = ["europower", "perseus", "mars", "powerflex", "ego", "miles", "isaar", "elmod"]
        self.model_names_plot = pd.Series(["Europower", "Perseus/Tango", "MarS/ZKNOT", "Powerflex-Grid-EU",
                                           "eGo/eTraGo", "MILES", "ISAaR", "ELMOD"], index=self.model_names)

        self.quantities_time = [
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
        self.quantities_categorical = ["energy_mix"]    # Dataframes, regions/carrier
        self.quantities = self.quantities_time+self.quantities_categorical

        for quantity in self.quantities:
            setattr(self, quantity, dict(zip(self.model_names, [pd.DataFrame()]*len(self.model_names))))

    def load_data(self, quantity="all", source="csv"):

        if quantity not in self.quantities+["all"]:
            logger.warning("Could not load any data. "+str(quantity).title()+" does not exist.")

        if source == "csv":
            if quantity == "all":
                for q in self.quantities:
                    self.load_data(source, q)
            else:
                if quantity in self.quantities_time:
                    for model in self.model_names:
                        getattr(self, quantity).update({model: pd.read_csv("data/"+quantity+"/"+model+".csv",
                                                                           index_col='snapshots',
                                                                           parse_dates=True)})
                elif quantity in self.quantities_categorical:
                    for model in self.model_names:
                        getattr(self, quantity).update({model: pd.read_csv("data/"+quantity+"/"+model+".csv",
                                                                           index_col='index')})

        elif source == "oep":
            pass

        else:
            raise NotImplementedError
