"""Top-level package for Ganblr."""

__author__ = """Tulip Lab"""
__email__ = 'jhzhou@tuliplab.academy'
__version__ = '0.1.0'

from .kdb import KdbHighOrderFeatureEncoder
from .utils import get_demo_data

__all__ = ['models', 'KdbHighOrderFeatureEncoder', 'get_demo_data']

@inproceedings{ganblr,
    author={Zhang, Yishuo and Zaidi, Nayyar A. and Zhou, Jiahui and Li, Gang},  
    booktitle={2021 IEEE International Conference on Data Mining (ICDM)},   
    title={GANBLR: A Tabular Data Generation Model},   
    year={2021},  
    pages={181-190},  
    doi={10.1109/ICDM51629.2021.00103}
}
@inbook{ganblrpp,
    author = {Yishuo Zhang and Nayyar Zaidi and Jiahui Zhou and Gang Li},
    title = {<bold>GANBLR++</bold>: Incorporating Capacity to Generate Numeric Attributes and Leveraging Unrestricted Bayesian Networks},
    booktitle = {Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
    pages = {298-306},
    doi = {10.1137/1.9781611977172.34},
}