"""
maskSLIC: Wrapper for supervoxel generation which handles 4D perfusion images using PCA reduction

Copyright (C) 2016-2019, Benjamin Irving
"""

from __future__ import division, print_function, unicode_literals

import numpy as np

from .feat_pca import PcaFeatReduce
from . import slic

def preprocess_pca(img, ncomp, normalise_input_image=True, norm_type='perc'):
    """
    Do PCA analysis on 4D image to turn it into a 3D image suitable for SLIC
    """
    # Normalize enhancement curves using first 3 points (FIXME magic number)
    baseline = np.mean(img[:, :, :, :3], axis=-1)
    img = img - np.tile(np.expand_dims(baseline, axis=-1), (1, 1, 1, img.shape[-1]))

    # Run PCA feature extraction
    pca = PcaFeatReduce(n_components=ncomp, norm_modes=True, norm_input=normalise_input_image, norm_type=norm_type)
    feat_image = pca.get_training_features(img, smooth_timeseries=2.0, feature_volume=True)
    return feat_image

def perfslic(img, roi,
             n_supervoxels=None,
             segment_size=None,
             compactness=0.1,
             seed_type='nplace',
             n_pca_components=3,
             **kwargs):
    """
    Just a wrapper for the SLIC interface
    :param compactness:
    :param sigma:
    :param seed_type: 'grid', 'nrandom'
                    Type of seed point initiliasation which is either based on a grid of no points or randomly
                    assigned within an roi
    :param segment_size: Mean supervoxel size (assuming no ROI but can be used with an ROI as standard)
    :param recompute_seeds: True or False
                            Recompute the initialisation points based on spatial distance within a ROI
    :param n_random_seeds:
    :return:
    """

    if img.ndim > 3 and img.shape[3] > 1:
        # For 4D data, use PCA to reduce down to 3D
        img = preprocess_pca(img, n_pca_components)
    else:
        # For 3D data scale to a range of 0-1
        if img.ndim > 3:
            img = np.squeeze(img, -1)
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())

    if n_supervoxels is None and segment_size is not None:
        n_supervoxels = int(img.size / segment_size)
    elif n_supervoxels is None:
        raise ValueError("Either the number of supervoxels or the segment size must be given")

    # FIXME enforce_connectivity=True does not seem to work in ROI mode?
    return slic(img,
                mask=roi,
                n_segments=n_supervoxels,
                compactness=compactness,
                seed_type=seed_type,
                multichannel=kwargs.pop("multichannel", False),
                multifeat=kwargs.pop("multifeat", True),
                recompute_seeds=kwargs.pop("recompute_seeds", True),
                **kwargs)
