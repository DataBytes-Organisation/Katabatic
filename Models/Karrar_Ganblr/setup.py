import os
from setuptools import setup, find_packages

# Define the absolute path to the current directory
path = os.path.abspath(os.path.dirname(__file__))

# Attempt to read the README for the long description
try:
    with open(os.path.join(path, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = "Ganblr Toolbox"

setup(
    name="ganblr",
    version="0.1.2",
    description="Ganblr Toolbox",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Karrar Alshanan",
    url="https://github.com/kalshana/ganblr_update",
    license="MIT",
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20", 
        "pandas>=1.3", 
        "tensorflow>=2.3", 
        "scikit-learn>=1.2", 
        "pgmpy>=0.1.19"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ]
)
