from __future__ import absolute_import

import os
from setuptools import setup, find_packages

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, 'requirements.txt') ) as f:
    required_packages = f.read().splitlines()
with open(os.path.join(dir_path,'README.md'), "r") as fh:
    long_description = fh.read()

setup(
    name='modex_net',
    version='0.0.0',
    author='Chloi Syranidou (FZJ)',
    author_email='c.syranidou@fz-juelich.de',
    description='Toolset for comparing electricity markets results',
    long_description=long_description,
    license='GPLv3',
    packages=find_packages(exclude=['test']),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=required_packages)
