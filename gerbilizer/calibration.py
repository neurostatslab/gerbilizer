"""
Functions used in calculating the calibration of a model outputting mean + cov.
"""

import logging
from typing import Tuple, Union

import numpy as np

from scipy.stats import multivariate_normal

from gerbilizer.util import assign_to_bin_2d, digitize, make_xy_grids

logger = logging.getLogger(__name__)


def min_mass_containing_location(
    pmf: np.ndarray, location: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray
):
    """
    Return the minimum probability mass containing the true location. This
    function is used to calculate the 95% confidence set, as well as calibration
    curves.
    """
    # maps: (n_y_bins, n_x_bins)
    # locations: (NUM_SAMPLES, 2)
    # coord_bins: (n_y_bins + 1, n_x_bins + 1, 2)  ( output of meshgrid then dstack )
    n_y_bins, n_x_bins = pmf.shape

    # first verify that xgrid and ygrid have correct shapes
    # since xgrid and ygrid should represent the EDGES of the gridpoints at
    # which the pmf is evaluated
    expected_grid_shape = (n_y_bins + 1, n_x_bins + 1)
    if xgrid.shape != expected_grid_shape or ygrid.shape != expected_grid_shape:
        raise ValueError(
            f"Expected `xgrid` and `ygrid` to have shape {expected_grid_shape}, "
            f"since the pmf has shape {pmf}. "
            f"Instead, encountered `xgrid` shape {xgrid.shape} and "
            f"`ygrid` shape {ygrid.shape}"
        )

    # flatten the pmf
    flattened = pmf.flatten()
    # argsort in descending order
    argsorted = flattened.argsort()[::-1]
    # assign the true location to a coordinate bin
    # reshape loc to a (1, 2) array so the vectorized function
    # assign_to_bin_2d still works
    location = location.reshape(1, 2)
    loc_idx = assign_to_bin_2d(location, xgrid, ygrid)
    # bin number for first interval containing location
    bin_idx = (argsorted == loc_idx).argmax()
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = pmf.size
    sorted_maps = flattened[argsorted]
    s = np.where(np.arange(num_bins) > bin_idx, 0, sorted_maps).sum()
    if (s < 0).any():
        logger.warning(f"negative min masses containing locations found: {s}")
    # clip values to remove any floating point errors
    # where the sum is greater than 1
    return s.clip(0, 1)


def calculate_confidence_set(pmf: np.ndarray, threshold: float):
    """
    Return the region around the mean to which the model assigns
    `threshhold` proprtion of the probability mass.
    """
    flattened = pmf.flatten()
    sorted_bins = flattened.argsort()[::-1]

    # select bins in descending order by mass until we hit `threshhold`
    sorted_masses = flattened[sorted_bins]
    total_mass_by_bins = sorted_masses.cumsum()

    bins_in_confidence_set = total_mass_by_bins <= threshold

    # reorder these values from their descended sorting pattern
    # back to the original places of the bins
    restore_bin_order = sorted_bins.argsort()
    confidence_set = bins_in_confidence_set[restore_bin_order]

    confidence_set = confidence_set.reshape(pmf.shape)

    return confidence_set


def confidence_set_stats(
    pmf: np.ndarray,
    threshhold: float,
    arena_dims: Union[Tuple[float, float], np.ndarray],
    true_location: np.ndarray,
) -> tuple[np.ndarray, float, bool, float]:
    """
    Given a pmf, compute and return the `threshhold` confidence set, along with
    relevant info like its area, whether the true location is in the set, etc.
    """
    # find the mean of each pmf
    center_xgrid, center_ygrid = make_xy_grids(
        arena_dims, shape=pmf.shape, return_center_pts=True
    )

    # pmf shape: (n_y_pts, n_x_pts)
    # marginal distributions over x: pmf.sum(axis=0)
    p_X = pmf.sum(axis=0)
    x_coords = (p_X * center_xgrid[0]).sum()
    # marginal distribution over y: pmf.sum(axis=1)
    p_Y = pmf.sum(axis=1)
    y_coords = (p_Y * center_ygrid[:, 0]).sum()
    means = np.column_stack((x_coords, y_coords))

    # get the confidence sets
    confidence_set = calculate_confidence_set(pmf, threshhold)

    # was the true location in the confidence sets?
    edge_xgrid, edge_ygrid = make_xy_grids(arena_dims, shape=np.array(pmf.shape) + 1)
    loc_bin = assign_to_bin_2d(true_location.reshape(1, 2), edge_xgrid, edge_ygrid)
    (y_idx, x_idx) = np.unravel_index(loc_bin, shape=pmf.shape)
    loc_in_confidence_set = bool(confidence_set[y_idx, x_idx])

    # find the area of each set
    arena_area = arena_dims[0] * arena_dims[1]
    num_bins_per_set = confidence_set.sum()
    area = float((num_bins_per_set / confidence_set.size) * arena_area)

    # find the distances from the furthest point in the calibration set
    # to the mean
    center_coord_grid = np.dstack((center_xgrid, center_ygrid))
    distances = np.linalg.norm(center_coord_grid - means[None], axis=-1)

    distances_in_set = np.where(confidence_set, distances, 0)
    dist_to_furthest_point = float(distances_in_set.max())

    return (confidence_set, area, loc_in_confidence_set, dist_to_furthest_point)


class CalibrationAccumulator:
    """
    Helper class to simplify the calculation of calibration,
    with support for multiple output types and applying various
    smoothing methods to each output type.
    """

    def __init__(
        self,
        arena_dims: Union[Tuple[float, float], np.ndarray],
        n_calibration_bins: int = 10,
        confidence_set_threshold: float = 0.95,
    ):
        """
        Initialize a CalibrationAccumulator object, which is a helper class
        to simplify the calculation of calibration in an online manner (iterating
        through or asynchronously going through a dataset). Currently only
        defined for models outputting a mean and a covariance matrix.
        """
        self.n_calibration_bins = n_calibration_bins

        self.arena_dims = arena_dims

        # initialize the internal mass counts tracking arrays
        self.mass_counts = np.zeros(n_calibration_bins)

        # and the outputs for confidence set calculation
        self.confidence_set_threshold = confidence_set_threshold

        self.confidence_sets = []
        self.confidence_set_areas = []
        self.location_in_confidence_set = []
        self.distances_to_furthest_point = []

    def calculate_step(self, model_output: np.ndarray, true_location: np.ndarray):
        """
        Perform one step of the calibration process on `model_output`.

        Essentially, this function calculates the probability assigned to the
        smallest region in the xy plane containing the true location. These
        regions are defined by progressively taking the location bins to which
        the model assigns the highest probability mass.
        """
        if model_output.shape == (2,):
            raise ValueError(
                f"Encountered output with shape {model_output.shape}. "
                "Calibration only supported for models that output a mean and covariance "
                "or a probability mass function."
            )
        elif model_output.shape == (3, 2):
            # assume model_output[0] is the mean
            # and model_output[1:] is the Cholesky factor
            # of the covariance matrix
            coords = self._make_coord_array()

            mean, cov = model_output[0], model_output[1:]
            # PREVIOUSLY INPUTTED CHOLESKY COV, NOW INPUTS COV.
            # mean, cholesky_cov = model_output[0], model_output[1:]
            # cov = cholesky_cov @ cholesky_cov.T

            # evaluate distribution at the gridpoints
            pmf = multivariate_normal(mean=mean, cov=cov).pdf(coords)
            # and renormalize
            pmf /= pmf.sum()
        elif model_output.shape[1:] == (3, 2):
            # if we have a batch of means and covariance matrices
            # make the mixture density pmf
            coords = self._make_coord_array()

            means, covs = model_output[:, 0], model_output[:, 1:]
            # means, cholesky_covs = model_output[:, 0], model_output[:, 1:]
            # covs = cholesky_covs @ cholesky_covs.swapaxes(-2, -1)

            gaussian_pmfs = [
                multivariate_normal(mean=mean, cov=cov).pdf(coords)
                for mean, cov in zip(means, covs)
            ]
            # average the pdfs with uniform weights to get the outputted density
            pmf = np.mean(gaussian_pmfs, axis=0)
            pmf /= pmf.sum()
        elif model_output.ndim == 2:
            # assume the model output is just a pmf
            pmf = model_output
            pmf /= pmf.sum()
        else:
            raise ValueError(f"Model output shape {model_output.shape} not understood!")

        # calculate the confidence set for this prediction
        # and store useful stats about it
        (
            confidence_set,
            area,
            loc_in_confidence_set,
            dist_to_furthest_point,
        ) = confidence_set_stats(
            pmf, self.confidence_set_threshold, self.arena_dims, true_location
        )

        self.confidence_sets.append(confidence_set)
        self.confidence_set_areas.append(area)
        self.location_in_confidence_set.append(loc_in_confidence_set)
        self.distances_to_furthest_point.append(dist_to_furthest_point)

        # get our x and ygrids to match the shape of the pmfs
        # since the grids track the edge points, we should have
        # one more point in each coordinate direction.
        grid_shape = np.array(pmf.shape) + 1
        xgrid, ygrid = make_xy_grids(self.arena_dims[:2], shape=grid_shape)

        # reshape location to (1, 2) if necessary
        # we do this so the repeat function works out correctly
        if true_location.shape != (1, 2):
            true_location = true_location.reshape((1, 2))

        # perform the calibration calculation step
        mass = min_mass_containing_location(pmf, true_location, xgrid, ygrid)

        # transform to the bin in [0, 1] to which each value corresponds,
        # essentially iteratively building a histogram with each step
        bins = np.arange(self.n_calibration_bins + 1) / self.n_calibration_bins
        bin_idx = digitize(mass, bins)

        # update mass counts array
        self.mass_counts[bin_idx] += 1

    def results(self):
        """
        Calculate calibration curves and error from the collected
        results of all the calibration steps.
        """
        calibration_curve = self.mass_counts.cumsum() / self.mass_counts.sum()
        # prepend a zero to the calibration curve so each value represents
        # the proportion of values whose min_mass is less than the bin
        calibration_curve = np.pad(calibration_curve, (1, 0), "constant")

        results = {}
        results["calibration_curve"] = calibration_curve

        results["confidence_sets"] = self.confidence_sets
        results["confidence_set_areas"] = self.confidence_set_areas
        results["location_in_confidence_set"] = self.location_in_confidence_set
        results["distances_to_furthest_point"] = self.distances_to_furthest_point

        return results

    def _make_coord_array(self):
        """
        Return a grid of evenly spaced points on the arena floor.
        """
        xdim = self.arena_dims[0]
        ydim = self.arena_dims[1]

        # change xgrid / ygrid size to preserve aspect ratio
        ratio = ydim / xdim
        desired_shape = (int(ratio * 100), 100)
        xgrid, ygrid = make_xy_grids(
            (xdim, ydim), shape=desired_shape, return_center_pts=True
        )
        coords = np.dstack((xgrid, ygrid))
        return coords
