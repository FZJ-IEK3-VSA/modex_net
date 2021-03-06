from . import config

empty = {"columns": [], "constraints": []}

market_time = {"columns": [{"name": "id", "data_type": "bigint", "is_nullable": "NO"},
                           {"name": "snapshots", "data_type": "timestamp", "is_nullable": "NO"},
                           ] + [{"name": col, "data_type": "real", "is_nullable": "YES"}
                                for col in config.eu_neighs_ISO2['eu_neighs_ISO2']],
               "constraints": [{"constraint_type": "PRIMARY KEY", "constraint_parameter": "id"}]
               }

grid_time = {"columns": [{"name": "id", "data_type": "bigint", "is_nullable": "NO"},
                         {"name": "snapshots", "data_type": "timestamp", "is_nullable": "NO"},
                         ] + [{"name": col, "data_type": "real", "is_nullable": "YES"}
                              for col in config.de_nuts1['de_nuts1']],
             "constraints": [{"constraint_type": "PRIMARY KEY", "constraint_parameter": "id"}]
             }

interconn_time = {"columns": [{"name": "id", "data_type": "bigint", "is_nullable": "NO"},
                              {"name": "snapshots", "data_type": "timestamp", "is_nullable": "NO"},
                              ] + [{"name": col, "data_type": "real", "is_nullable": "YES"}
                                   for col in config.eu_neighs_conns],
                  "constraints": [{"constraint_type": "PRIMARY KEY", "constraint_parameter": "id"}]
                  }

mix = {"columns": [{"name": "id", "data_type": "bigint", "is_nullable": "NO"},
                   {"name": "carrier", "data_type": "varchar(128)", "is_nullable": "NO"},
                   ] + [{"name": col, "data_type": "real", "is_nullable": "YES"}
                        for col in config.eu_neighs_ISO2['eu_neighs_ISO2']],
       "constraints": [{"constraint_type": "PRIMARY KEY", "constraint_parameter": "id"}]
       }

vres_curtailments = {'market': market_time, 'grid': grid_time}
emissions = {'market': market_time, 'grid': grid_time}
load_curtailments = {'market': market_time, 'grid': grid_time}

storage_p = {'market': market_time, 'grid': grid_time}
storage_soc = {'market': market_time, 'grid': grid_time}

import_export = {'market': interconn_time, 'grid': interconn_time}

electricity_prices = {'market': market_time, 'grid': empty}

energy_mix = {'market': mix, 'grid': mix}
