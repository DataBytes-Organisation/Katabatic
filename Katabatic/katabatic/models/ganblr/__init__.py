"""Top-level package for Ganblr."""

__author__ = """Tulip Lab"""
__email__ = 'jhzhou@tuliplab.academy'
__version__ = '0.1.0'

from .kdb import KdbHighOrderFeatureEncoder
from .utils import get_demo_data

__all__ = ['models', 'KdbHighOrderFeatureEncoder', 'get_demo_data']