# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

from . import indicators, plots

de_nuts1 = indicators.de_nuts1
de_nuts1_names = indicators.de_nuts1_names

eu_neighs_ISO2 = indicators.eu_neighs_ISO2
eu_neighs_names = indicators.eu_neighs_names

model_names = indicators.model_names
model_names_plot = indicators.model_names_plot

quantities_time = indicators.quantities_time
quantities_categorical = indicators.quantities_categorical
quantities = indicators.quantities

carriers = list(plots.aggregate_carriers.keys())