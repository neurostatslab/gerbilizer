"""
Utility functions, like for calibration calculation.
"""

import logging
import math

from typing import Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase

logger = logging.getLogger(__name__)


def subplots(
    n_plots, scale_factor=4, sharex=True, sharey=True, **kwargs
) -> tuple[FigureBase, list[Axes]]:
    """
    Create nicely sized and laid-out subplots for a desired number of plots.
    """
    # essentially we want to make the subplots as square as possible
    # number of rows is the largest factor of n_plots less than sqrt(n_plots)
    options = range(1, int(math.sqrt(n_plots) + 1))
    n_rows = max(filter(lambda n: n_plots % n == 0, options))
    n_cols = int(n_plots / n_rows)
    # now generate the Figure and Axes pyplot objects
    # cosmetic scale factor to make larger plot
    figsize = (n_cols * scale_factor, n_rows * scale_factor)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharey, **kwargs
    )
    flattened_axes = []
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            flattened_axes += list(ax_row)
        else:
            flattened_axes.append(ax_row)
    return fig, flattened_axes


def digitize(locations, bin_edges) -> np.ndarray:
    """
    Wrapper for np.digitize where an error is raised if a value far
    outside the given range is encountered. NOTE: We assume that the
    bins are given in increasing order.
    """
    # check that the bins are in increasing order
    diffs = np.diff(bin_edges)
    if (diffs <= 0).any():
        raise ValueError("Expected array `bins` to be in increasing order.")
    # get max distance between bins
    max_dx = diffs.max()
    # define a new bin array where the highest bin
    # is bins[-1] + tol, with our tolerance as
    # 0.01 * max_dx.
    # say values less than this are in the highest bin.
    # this is to catch floating point errors that push values
    # greater than bin_edges[-1], while still
    # letting us catch extreme values that are way too high.
    tol = 0.01 * max_dx
    extended_bins = np.append(bin_edges, bin_edges[-1] + tol)
    # digitize locations using these new bins, removing the
    # leftmost bin edge to avoid off-by-one errors.
    edges_to_use = extended_bins[1:]
    bin_idxs = np.digitize(locations, edges_to_use)
    # if any value was greater than bins[-1] + max_dx,
    # raise a value error.
    if (bin_idxs == len(edges_to_use)).any():
        positions = bin_idxs == len(edges_to_use)
        values = locations[positions]
        err_display = [f"idx: {p} | value: {v}" for (p, v) in zip(positions, values)]
        raise ValueError(
            f"Encountered value far greater than the largest bin edge! "
            f"Largest bin edge: {bin_edges[-1]}; Invalid values and their "
            f"positions: {err_display}"
        )
    # if not, say that the values were sufficiently close to the bin edges
    # and clip them to match the number of bins
    num_bins = len(bin_edges) - 1
    highest_bin_idx = num_bins - 1
    return bin_idxs.clip(0, highest_bin_idx)


def assign_to_bin_2d(locations, xgrid, ygrid):
    """
    Return an array of indices of the 2d bins to which each input in
    `locations` corresponds.

    The indices correspond to the "flattened" version of the grid. In essence,
    for a point in bin (i, j), the output is i + (n_x_bins * j), where n_x_bins
    is the number of grid bins in the x direction--essentially the number
    of gridpoints in that direction - 1.
    """
    # locations: (NUM_SAMPLES, 2)
    # xgrid: (n_y_pts, n_x_pts)
    # xgrid: (n_y_pts, n_x_pts)
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    # 1d array of numbers representing x coord of each bin
    x_bin_edges = xgrid[0]
    # same for y coord
    y_bin_edges = ygrid[:, 0]
    # assign each coord to a bin in one dimension
    x_idxs = digitize(x_coords, x_bin_edges)
    y_idxs = digitize(y_coords, y_bin_edges)
    # NOTE: we expect model output to have shape (NUM_SAMPLES, n_y_bins, n_x_bins)
    # where n_y_bins = len(y_bin_edges) - 1, and similar for n_x_bins.
    # so when we flatten, the entry at coordinate (i, j) gets mapped to
    # (n_x_bins * j) + i
    n_x_bins = len(x_bin_edges) - 1
    return (n_x_bins * y_idxs) + x_idxs


def make_xy_grids(
    arena_dims: Union[Tuple[float, float], np.ndarray],
    resolution: Optional[float] = None,
    shape: Optional[Union[Tuple[float, float], np.ndarray]] = None,
    return_center_pts: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a two-tuple or np array storing the x and y dimensions and a desired
    grid resolution, return a tuple of arrays stooring the x and y coordinates
    at every point in the arena spaced apart by `desired_resolution`.

    Optionally, calculate those gridpoints and instead return the CENTER of each
    bin based on the flag `return_center_pts`.

    Note that this function expects shape in the format (n_y_pts, n_x_pts).

    Examples:
    ```
    >>> make_xy_grids((4, 3), shape=(2, 3))
    (
        array([
            [0., 2., 4.],
            [0., 2., 4.]
        ]),
        array([
            [0., 0., 0.],
            [3., 3., 3.]
        ])
    )
    >>> make_xy_grids((4, 3), shape=(1, 2), return_center_pts=True)
    (
        array([
            [0.66666667, 2.        , 3.33333333],
            [0.66666667, 2.        , 3.33333333]
        ]),
        array([
            [0.75, 0.75, 0.75],
            [2.25, 2.25, 2.25]
        ])
    )
    ```

    """
    if resolution is None and shape is None:
        raise ValueError("One of `resolution`, `shape` is required!")

    if not resolution:
        # np.meshgrid returns a shape of (n_y_pts, n_x_pts)
        # but expects (xs, ys) as arguments.
        # reverse the shape so we match this convention.
        pts_per_dim = np.array(shape)[::-1]
    else:
        pts_per_dim = np.ceil(np.array(arena_dims) / resolution).astype(int)

    def _coord_array(dim_pts):
        """
        Get an array of coordinates along one axis.

        Expects `dim_pts` to be a tuple (dim, n_pts), where `dim`
        is the length of the grid along the current axis, and `n_pts`
        is the desired number of points to be placed along the axis.
        """
        dimension, n_pts = dim_pts
        # if the user requested to return the CENTER of each bin,
        # create one extra gridpoint in each direction
        # then return the average of each successive bin
        if return_center_pts:
            edge_coords = np.linspace(0, dimension, n_pts + 1)
            # add half the successive differences to get avgs
            # between edge_pts[i] and edge_pts[i+1]
            coords = edge_coords[:-1] + (np.diff(edge_coords) / 2)
        else:
            coords = np.linspace(0, dimension, n_pts)
        return coords

    xs, ys = map(_coord_array, zip(arena_dims, pts_per_dim))

    xgrid, ygrid = np.meshgrid(xs, ys)

    return (xgrid, ygrid)
