# % pip install wheel
# % pip install setuptools
# % pip install twine

import click
import importlib # are these 
import os
import platform
import shutil
import sys
import traceback   # necessary?

from os.path import join
from setuptools import find_packages, setup

setup(
    name='katabatic',
    packages=find_packages(include=['katabatic']),
    version='0.0.1',
    description='An open source framework for tabular data generation',
    author='Jaime Blackwell, Nayyar Zaidi',
    install_requires=[],
    setup_requires=['pytest-runner'],
)
