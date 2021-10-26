# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import numpy as np
import modex_net

def test_oep_reduction(token):
    calc = modex_net.indicators.Calculator(2016, "market", "base", "oep", oep_token=token)
    np.testing.assert_almost_equal(calc.reduction('energy_mix', 'sum').loc['DE', 'europower'], 515.56, decimal=2)


