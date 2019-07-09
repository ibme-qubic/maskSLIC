"""
maskSLIC: Cython wrapper for some processing scripts

Copyright (C) 2016-2019, Benjamin Irving
"""

from __future__ import division, print_function, absolute_import

from libcpp.vector cimport vector
import numpy as np
cimport numpy as cnp

cdef extern from "processing.h":
    vector[float] get_mean_point_distance(vector[int] &, vector[int] &, vector[int] &)

def get_mpd(cnp.ndarray[int, ndim=1] x, cnp.ndarray[int, ndim=1] y, cnp.ndarray[int, ndim=1] z):
    """
    Get the mean distance between points in the region

    :param x: X co-ordinates of points
    :param y: Y co-ordinates of points
    :param z: Z co-ordinates of points
    
    :return: Mean distance from a point to it's nearest neighbour, across all points
    """
    dist = get_mean_point_distance(x, y, z)
    return np.array(dist)
