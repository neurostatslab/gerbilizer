import numpy as np

from matplotlib import pyplot as plt


def assign_to_bin_1d(arr, bins):
    """
    Return an array of indices of the bins to which each input in
    `arr` corresponds.
    
    Note: this method assumes that all bins are evenly spaced apart.
    """
    x_0 = bins[0]
    dx = bins[1] - x_0
    i = int((arr - x_0) / dx)
    return i


def assign_to_bin_2d(locations, xgrid, ygrid):
    # locations: (NUM_SAMPLES, 2)
    # xgrid: (n_y_pts, n_x_pts)
    # xgrid: (n_y_pts, n_x_pts)
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    # 1d array of numbers representing x coord of each bin
    x_bins = xgrid[0]
    # same for y coord
    y_bins = ygrid[:, 0]
    x_idxs = [assign_to_bin_1d(x_coord, x_bins) for x_coord in x_coords]
    y_idxs = [assign_to_bin_1d(y_coord, y_bins) for y_coord in y_coords]
    # NOTE: we expect model output to have shape (NUM_SAMPLES, n_x_pts, n_y_pts)
    # so when we flatten, the entry at coordinate (i, j) gets mapped to
    # i * n_y_pts + j
    n_y_pts = len(y_bins)
    # print(list(zip(x_idxs, y_idxs)))
    return np.array([i * n_y_pts + j for i, j in zip(x_idxs, y_idxs)])


def f_2d(
    maps: np.ndarray,
    locations: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray
    ):
    # maps: (NUM_SAMPLES, n_x_pts, n_y_pts)
    # locations: (NUM_SAMPLES, 2)
    # coord_bins: (n_y_pts, n_x_pts, 2)  ( output of meshgrid then dstack ) 
    # reshape maps to (NUM_SAMPLES, N_BINS)
    num_samples = maps.shape[0]
    flattened_maps = maps.reshape((num_samples, -1))
    idx_matrix = flattened_maps.argsort(axis=1)[:, ::-1]
    # bin number for each location
    loc_idxs = assign_to_bin_2d(locations, xgrid, ygrid)
    # bin number for first interval containing location
    bin_idxs = (idx_matrix == loc_idxs[:, np.newaxis]).argmax(axis=1)
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = xgrid.shape[0] * xgrid.shape[1]
    x_idx = np.arange(num_bins)[np.newaxis, :].repeat(num_samples, axis=0)
    condition = x_idx > bin_idxs[:, np.newaxis]
    sorted_maps = np.take_along_axis(flattened_maps, idx_matrix, axis=1)
    s = np.where(condition, 0, sorted_maps).sum(axis=1)
    return s

def calibration_curve(
    model_output,
    true_coords,
    xgrid,
    ygrid,
    ax = None,
    n_bins=10
    ):
    s = f_2d(model_output, true_coords, xgrid, ygrid)
    counts, bin_edges = np.histogram(
        s,
        bins=n_bins,
        range=(0, 1)
    )
    observed_props = counts.cumsum() / counts.sum()
    use_to_plot = ax if ax else plt
    use_to_plot.scatter(bin_edges[1:], observed_props)
    use_to_plot.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
    return bin_edges[1:], observed_props