"""
maskSLIC: Modified from scikit-image slic method

Original code (C) scikit-image
Modification (C) 2016-2019 Benjamin Irving

See LICENSE.txt for more details
"""

# coding=utf-8
from __future__ import division, absolute_import, unicode_literals, print_function

import warnings
import collections as coll

import numpy as np

from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

from skimage.util import img_as_float, regular_grid
from skimage.color import rgb2lab

from ._slic import _slic_cython, _enforce_label_connectivity_cython
from .processing import get_mpd

def place_seed_points(image, mask, n_segments, spacing):
    """
    Method for placing seed points in an ROI

    Note:
    Optimal point placement problem is somewhat related to the k-center problem
     metric facility location (MFL)
     Maxmin facility location
    https://en.wikipedia.org/wiki/Facility_location_problem

    :param image:
    :param mask:
    :param n_segments:
    :param spacing:

    :return:
    """
    segments_z = np.zeros(n_segments, dtype=np.int64)
    segments_y = np.zeros(n_segments, dtype=np.int64)
    segments_x = np.zeros(n_segments, dtype=np.int64)

    m_inv = np.copy(mask)

    # Cropping to bounding box around ROI
    nonzero_x, nonzero_y, nonzero_z = np.nonzero(m_inv)
    bbox_start = [np.min(nonzero_x), np.min(nonzero_y), np.min(nonzero_z)]
    bbox_end = [np.max(nonzero_x), np.max(nonzero_y), np.max(nonzero_z)]
    m_inv = m_inv[bbox_start[0]:bbox_end[0]+1, bbox_start[1]:bbox_end[1]+1, bbox_start[2]:bbox_end[2]+1]

    # SEED STEP 1:  n seeds are placed as far as possible from every other seed and the edge.
    for seg_idx in range(n_segments):

        # Distance transform
        dtrans = distance_transform_edt(m_inv, sampling=spacing)
        dtrans = gaussian_filter(dtrans, sigma=0.1)

        # Use the maximum locations for the first two points
        coords1 = np.nonzero(dtrans == np.max(dtrans))
        segments_z[seg_idx] = coords1[0][0]
        segments_x[seg_idx] = coords1[1][0]
        segments_y[seg_idx] = coords1[2][0]

        # Adding a new point
        m_inv[segments_z[seg_idx], segments_x[seg_idx], segments_y[seg_idx]] = False

    segments_z = segments_z + bbox_start[0]
    segments_x = segments_x + bbox_start[1]
    segments_y = segments_y + bbox_start[2]

    segments_color = np.zeros((segments_z.shape[0], image.shape[3]))
    segments = np.concatenate([segments_z[..., np.newaxis],
                               segments_x[..., np.newaxis],
                               segments_y[..., np.newaxis],
                               segments_color], axis=1)

    segments_z = np.ascontiguousarray(segments_z, dtype=np.int32)
    segments_x = np.ascontiguousarray(segments_x, dtype=np.int32)
    segments_y = np.ascontiguousarray(segments_y, dtype=np.int32)

    out1 = get_mpd(segments_z, segments_x, segments_y)
    step_z, step_x, step_y = out1[0], out1[1], out1[2]

    return segments, step_x, step_y, step_z

def slic(image, n_segments=100, compactness=10., max_iter=10, sigma=0,
         seed_type='grid', spacing=None, multichannel=True, convert2lab=None,
         enforce_connectivity=False, min_size_factor=0.5, max_size_factor=3,
         slic_zero=False, multifeat=False, return_adjacency=False, mask=None,
         return_segments=False, recompute_seeds=False):
    """
    Segments image using k-means clustering in Color-(x,y,z) space.

    :param image: 2D, 3D or 4D ndarray
        Input image, which can be 2D or 3D, and grayscale or multichannel
        (see `multichannel` parameter).
    :param n_segments: int, optional
        The (approximate) number of labels in the segmented output image.
    :param compactness: float, optional
        Balances color proximity and space proximity. Higher values give
        more weight to space proximity, making superpixel shapes more
        square/cubic. In SLICO mode, this is the initial compactness.
        This parameter depends strongly on image contrast and on the
        shapes of objects in the image. We recommend exploring possible
        values on a log scale, e.g., 0.01, 0.1, 1, 10, 100, before
        refining around a chosen value.
    :param max_iter: int, optional
        Maximum number of iterations of k-means.
    :param sigma: float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each
        dimension of the image. The same sigma is applied to each dimension in
        case of a scalar value. Zero means no smoothing.
        Note, that `sigma` is automatically scaled if it is scalar and a
        manual voxel spacing is provided (see Notes section).
    :param spacing: (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, `slic`
        assumes uniform spacing (same voxel resolution along z, y and x).
        This parameter controls the weights of the distances along z, y,
        and x during k-means clustering.
    :param multichannel: bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.
    :param convert2lab: bool, optional
        Whether the input should be converted to Lab colorspace prior to
        maskslic. The input image *must* be RGB. Highly recommended.
        This option defaults to ``True`` when ``multichannel=True`` *and*
        ``image.shape[-1] == 3``.
    :param enforce_connectivity: bool, optional
        Whether the generated segments are connected or not
    :param min_size_factor: float, optional
        Proportion of the minimum segment size to be removed with respect
        to the supposed segment size ```depth*width*height/n_segments```
    :param max_size_factor: float, optional
        Proportion of the maximum connected segment size. A value of 3 works
        in most of the cases.
    :param slic_zero: bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [2]_
    :param mask: ndarray of bools or 0s and 1s, optional
        Array of same shape as `image`. Supervoxel analysis will only be performed on points at
        which mask == True

    :return: labels : 2D or 3D array
        Integer mask indicating segment labels.

    Raises
    ------
    ValueError
        If ``convert2lab`` is set to ``True`` but the last array
        dimension is not of length 3.

    Notes
    -----
    * If `sigma > 0`, the image is smoothed using a Gaussian kernel prior to
      maskslic.

    * If `sigma` is scalar and `spacing` is provided, the kernel width is
      divided along each dimension by the spacing. For example, if ``sigma=1``
      and ``spacing=[5, 1, 1]``, the effective `sigma` is ``[0.2, 1, 1]``. This
      ensures sensible smoothing for anisotropic images.

    * The image is rescaled to be in [0, 1] prior to processing.

    * Images of shape (M, N, 3) are interpreted as 2D RGB images by default. To
      interpret them as 3D with the last dimension having length 3, use
      `multichannel=False`.

    References
    ----------
    .. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
        Pascal Fua, and Sabine Susstrunk, SLIC Superpixels Compared to
        State-of-the-art Superpixel Methods, TPAMI, May 2012.
    .. [2] http://ivrg.epfl.ch/research/superpixels#SLICO

    Examples
    --------
    >>> from maskslic import slic
    >>> from skimage.data import astronaut
    >>> img = astronaut()
    >>> segments = slic(img, n_segments=100, compactness=10)

    Increasing the compactness parameter yields more square regions:

    >>> segments = slic(img, n_segments=100, compactness=20)

    """
    if slic_zero:
        raise NotImplementedError("Slic zero has not been implemented yet for maskSLIC.")

    if mask is None and seed_type == 'nplace':
        warnings.warn('nplace assignment of seed points should only be used with an ROI. Changing seed type.')
        seed_type = 'grid'

    if seed_type == 'nplace' and recompute_seeds is False:
        warnings.warn('Seeds should be recomputed when seed points are randomly assigned')

    image = img_as_float(image)
    is_2d = False
    if image.ndim == 2:
        # 2D grayscale image
        image = image[np.newaxis, ..., np.newaxis]
        is_2d = True
    elif image.ndim == 3 and multichannel:
        # Make 2D multichannel image 3D with depth = 1
        image = image[np.newaxis, ...]
        is_2d = True
    elif image.ndim == 3 and not multichannel:
        # Add channel as single last dimension
        image = image[..., np.newaxis]

    if spacing is None:
        spacing = np.ones(3)
    elif isinstance(spacing, (list, tuple)):
        spacing = np.array(spacing, dtype=np.double)

    if not isinstance(sigma, coll.Iterable):
        sigma = np.array([sigma, sigma, sigma], dtype=np.double)
        sigma /= spacing.astype(np.double)
    elif isinstance(sigma, (list, tuple)):
        sigma = np.array(sigma, dtype=np.double)
    if (sigma > 0).any():
        # add zero smoothing for multichannel dimension
        sigma = list(sigma) + [0]
        image = ndi.gaussian_filter(image, sigma)

    if multichannel and (convert2lab or convert2lab is None):
        if image.shape[-1] != 3 and convert2lab:
            raise ValueError("Lab colorspace conversion requires a RGB image.")
        elif image.shape[-1] == 3:
            image = rgb2lab(image)

    if multifeat:
        feat_scale = float(image.shape[3])
    else:
        feat_scale = 1.0

    depth, height, width = image.shape[:3]

    if mask is None:
        mask = np.ones(image.shape[:3], dtype=np.bool)
    else:
        mask = np.asarray(mask, dtype=np.bool)

    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]

    if seed_type == 'nplace':

        segments, step_x, step_y, step_z = place_seed_points(image, mask, n_segments, spacing)

    elif seed_type == 'grid':

        # initialize cluster centroids for desired number of segments
        # essentially just outputs the indices of a grid in the x, y and z direction
        grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]
        # returns 3 slices (an object representing an array of slices, see builtin slice)
        slices = regular_grid(image.shape[:3], n_segments)
        step_z, step_y, step_x = [int(s.step) for s in slices]  # extract step size from slices
        segments_z = grid_z[slices]  # use slices to extract coordinates for centre points
        segments_y = grid_y[slices]
        segments_x = grid_x[slices]

        # list of all locations as well as zeros for the color features
        segments_color = np.zeros(segments_z.shape + (image.shape[3],))
        segments = np.concatenate([segments_z[..., np.newaxis],
                                   segments_y[..., np.newaxis],
                                   segments_x[..., np.newaxis],
                                   segments_color],
                                  axis=-1).reshape(-1, 3 + image.shape[3])

        if mask is not None:
            ind1 = mask[segments[:, 0].astype('int'), segments[:, 1].astype('int'), segments[:, 2].astype('int')]
            segments = segments[ind1, :]
    else:
        raise ValueError('seed_type should be nplace or grid')

    segments = np.ascontiguousarray(segments)

    # We do the scaling of ratio in the same way as in the SLIC paper
    # so the values have the same meaning
    step = float(max((step_z, step_y, step_x)))
    ratio = 1.0 / compactness

    image = np.ascontiguousarray(image * ratio, dtype=np.double)
    mask = np.ascontiguousarray(mask, dtype=np.int32)

    #segments_old = np.copy(segments)

    if recompute_seeds:
        # Seed step 2: Run SLIC to reinitialise seeds
        # Runs the supervoxel method but only uses distance to better initialise the method
        labels = _slic_cython(image, mask, segments, step, max_iter, spacing, slic_zero, feat_scale, only_dist=True)

    labels = _slic_cython(image, mask, segments, step, max_iter, spacing, slic_zero, feat_scale, only_dist=False)

    if enforce_connectivity:
        segment_size = mask.sum() / n_segments

        min_size = int(min_size_factor * segment_size)
        max_size = int(max_size_factor * segment_size)

        labels = _enforce_label_connectivity_cython(labels, mask, n_segments, min_size, max_size)

    ret = []
    if is_2d:
        ret.append(labels[0])
    else:
        ret.append(labels)

    if return_adjacency:
        # Also return adjacency map
        labels = np.ascontiguousarray(labels, dtype=np.int32)
        if mask is None:
            adj_mat, border_mat = _find_adjacency_map(labels)
        else:
            adj_mat, border_mat = _find_adjacency_map_mask(labels)

        ret.append(adj_mat)
        ret.append(border_mat)

    if return_segments:
        ret.append(segments)

    if len(ret) == 1:
        return ret[0]
    else:
        return tuple(ret)
