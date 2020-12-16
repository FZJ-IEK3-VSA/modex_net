# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import os
import pandas as pd

leap_years = False  # if false, the last day of leap years is dropped

dir_name = os.path.dirname(__file__)
de_nuts1 = pd.read_csv(os.path.join(dir_name, "config_files", "de_nuts1.csv"), encoding='ISO-8859-1')
eu_neighs_ISO2 = pd.read_csv(os.path.join(dir_name, "config_files", "eu_neighs.csv"))
eu_neighs_conns = ["AT-CH", "AT-CZ", "AT-DE", "BE-DE", "BE-FR", "BE-LU", "BE-NL", "CH-DE", "CH-FR", "CZ-DE", "CZ-PL",
                   "DE-DK", "DE-FR", "DE-LU", "DE-NL", "DE-NO", "DE-PL", "DE-SE", "DK-NL", "DK-NO", "DK-SE", "FR-LU",
                   "NL-NO", "NO-SE", "PL-SE"]

model_names = pd.read_csv(os.path.join(dir_name, "config_files", "model_names.csv"))

carriers_reduced = ['Nuclear', 'Lignite', 'Hard Coal', 'Natural Gas', 'Hydro', 'Wind', 'Solar', 'Bioenergy', 'Oil',
                    'Other']
carriers_all = ['Nuclear', 'Lignite', 'Hard Coal', 'Natural Gas', 'Run of River', 'Reservoir', 'Pumped Hydro Storage',
                'Wind Onshore', 'Wind Offshore', 'Solar', 'Bioenergy', 'Oil', 'Geothermal', 'Waste',
                'Other Conventional', 'Other Renewable']

linewidth = 2.5
fontsize = 16

colors_energy_mix = {'Nuclear': 'purple',
                     'Lignite': 'saddlebrown',
                     'Hard Coal': 'black',
                     'Natural Gas': 'orange',
                     'Hydro': 'dodgerblue',
                     'Run of River': 'deepskyblue',
                     'Reservoir': 'dodgerblue',
                     'Pumped Hydro Storage': 'lightskyblue',
                     'Wind': 'mediumblue',
                     'Wind Onshore': 'royalblue',
                     'Wind Offshore': 'mediumblue',
                     'Solar': 'yellow',
                     'Bioenergy': 'green',
                     'Oil': 'red',
                     'Other': 'darkgray',
                     'Other Conventional': 'darkgray',
                     'Other Renewable': 'lime',
                     'Geothermal': 'gold',
                     'Waste': 'lightgray',
                     'Batteries': 'pink',
                     'Hydrogen': 'hotpink',
                     'Storage': 'hotpink'}

aggregate_carriers = {'Nuclear': 'Nuclear',
                      'Lignite': 'Lignite',
                      'Hard Coal': 'Hard Coal',
                      'Natural Gas': 'Natural Gas',
                      'Hydro': 'Hydro',
                      'Reservoir': 'Hydro',
                      'Run of River': 'Hydro',
                      'Pumped Hydro Storage': 'Hydro',
                      'Wind': 'Wind',
                      'Wind Offshore': 'Wind',
                      'Wind Onshore': 'Wind',
                      'Solar': 'Solar',
                      'Bioenergy': 'Bioenergy',
                      'Oil': 'Oil',
                      'Other': 'Other',
                      'Geothermal': 'Other',
                      'Waste': 'Other',
                      'Other Conventional': 'Other',
                      'Other Renewable': 'Other'}

