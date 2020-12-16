from __future__ import absolute_import

from setuptools import setup, find_packages
from codecs import open



setup(
    name='modex_net',
    version='0.0.0',
    author='Chloi Syranidou (FZJ)',
    author_email='c.syranidou@fz-juelich.de',
    description='Toolset for comparing model results',
    #license='GPLv3',
    packages=find_packages(exclude=['doc', 'test']),
    include_package_data=True,
    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'scikit-learn'
    ])
