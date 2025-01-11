# % pip install wheel
# % pip install setuptools
# % pip install twine

from setuptools import find_packages, setup

setup(
    name="katabatic",
    packages=find_packages(include=["katabatic"]),
    version="0.0.1",
    description="An open source framework for tabular data generation",
    author="Jaime Blackwell, Nayyar Zaidi",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "pyitlib",
        "tensorflow",
        "pgmpy",
        "sdv",
    ],
    extras_require={
        "dev": [
            # Add other development dependencies here
        ],
    },
    python_requires=">=3.9",
    setup_requires=["pytest-runner"],
    long_description=open("README.md").read(),
)
