# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

from oep_client import OepClient

def test_oep(token):
    cli = OepClient(token=token)
    data = cli.select_from_table("modexnet_europower_2016_market_energy_mix")
    assert not {'id', 'carrier', 'country_code', 'value'}.symmetric_difference(data[0].keys())


