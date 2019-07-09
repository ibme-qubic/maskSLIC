"""
maskSLIC: Simple linear iterative clustering (SLIC) in a region of interest (ROI)

Copyright (C) 2016-2019, Benjamin Irving
"""

from .slic_superpixels import slic
from ._version import __version__, __timestamp__

__all__ = ['slic', '__version__', '__timestamp__']
