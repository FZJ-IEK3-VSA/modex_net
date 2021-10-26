# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import numpy as np
import modex_net

def test_oep_pair_distance(token):
    calc = modex_net.indicators.Calculator(2016, "market", "base", "oep", oep_token=token)
    np.testing.assert_almost_equal(calc.pair_distance_single('energy_mix', 'max_diff', 'europower').loc['DE', 'perseus'], 159.56, decimal=2)


