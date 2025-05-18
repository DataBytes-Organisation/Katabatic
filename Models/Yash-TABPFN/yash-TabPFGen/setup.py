from setuptools import setup, find_packages
from pathlib import Path

# Get the current directory (where setup.py is)
HERE = Path(__file__).resolve().parent

# Read requirements.txt
def read_requirements(filepath):
    """Read the contents of requirements.txt"""
    with open(HERE / filepath, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="TabPFGen",
    version="0.1.2",
    description="Synthetic tabular data generation using energy-based modeling and TabPFN",
    long_description=(HERE / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Sebastian Haan",
    url="https://github.com/sebhaan/TabPFGen",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements("requirements.txt"),
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    python_requires=">=3.11",
)