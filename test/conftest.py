# -*- coding: utf-8 -*-
"""
@author: Chloi Syranidou
"""

import pytest

def pytest_addoption(parser):
    parser.addoption("--token", action="store")

@pytest.fixture(scope='session')
def token(request):
    return request.config.getoption("--token")



