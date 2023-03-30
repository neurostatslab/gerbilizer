"""Loss functions for the DNNs."""

import logging

from typing import Literal, Optional

import torch
from torch.nn import functional as F
from torch.distributions import MultivariateNormal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# MARK: Loss functions
# All loss functions are expected to accept the prediction as the first argument
# and the ground truth as the second argument
def se_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    # Assumes the inputs have shape (B, 2), representing a batch of `B` 2-dimensional coordinates
    return 2 * torch.mean(torch.square(target - pred))  # Should be a scalar tensor


def map_se_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    """MSE loss over a location map."""
    target = torch.flatten(target, start_dim=1)
    pred = torch.flatten(pred, start_dim=1)
    return torch.mean(torch.square(target - pred).mean(dim=1))


def wass_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    """Calculates the earth mover's distance between the target and predicted locations.
    This is done by element-wise multiplying the prediction map by a (pre-computed) matrix
    of distances, in which each element contains the distance of that bin from the target
    location.
    This implementation is not symmetric because it scales the input 'pred' via a softmax.
    What is returned is not the true distance, but a numerically stable analogue that shares
    minima with the true distance w.r.t model parameters.
    """
    flat_pred = torch.flatten(pred, start_dim=1)
    flat_target = torch.flatten(target, start_dim=1)
    # Add 1 to target to make the smallest value 0
    error_map = F.log_softmax(flat_pred, dim=1) + torch.log(flat_target + 1)
    return torch.logsumexp(error_map, dim=1)


def gaussian_NLL(pred: torch.Tensor, target: torch.Tensor):
    """
    Negative log likelihood of the Gaussian parameterized by the model prediction.

    Assumes that the conditional distribution of a location y
    given an example x can be approximated by a multivariate gaussian

        p(y | x) = N(mu, Sigma)

    where mu(x) and Sigma(x) are models of the mean and covariance as a
    function of the audio.

    Expects pred to be a tensor of shape (B, 3, 2), where pred[i][0]
    represents the predicted mean for example i and the entries pred[i][1:]
    and the remaining entries represent a lower triangular matrix L such that
    L @ L.T = Sigma(x).
    """
    # make sure that preds has shape: (B, 3, 2) where B is the batch size
    if pred.ndim != 3:
        raise ValueError(
            "Expected `pred` to have shape (B, 3, 2) where B is the batch size, "
            f"but encountered shape: {pred.shape}"
        )

    y_hat = pred[:, 0]  # output, shape: (B, 2)
    L = pred[:, 1:]  # cholesky factor of covariance, shape: (B, 2, 2)

    # create the distribution corresponding to the outputted predictive
    # density
    multivariate_normal = MultivariateNormal(loc=y_hat, scale_tril=L)

    loss = -multivariate_normal.log_prob(target)

    return loss


def gaussian_NLL_half_normal_variances(
    pred: torch.Tensor, target: torch.Tensor, arena_dims: tuple[float, float]
):
    """
    Regularized version of the gaussian_NLL loss function. Specifically, this
    is the negative log posterior p(mu, Sigma | x) where we place half-Normal
    priors on the diagonal entries of the covariance matrix.

    Motivation: no prior belief about the skew direction of the model's
    confidence ellipses, so we should have a uniform prior over the correlation
    structure — reasonable choice is the LJK distribution with shape parameter
    eta = 1. Technically, we also implicitly put a uniform prior over the
    means, which are constrained to be on the unit square [-1, 1]^2. But since
    these are uniform, they don't affect the negative log posterior. The only
    thing that does is our half-Normal priors on the variances sigma^2_x and
    sigma^2_y — after taking the negative log, this results in just adding

        lambda_x sigma^2_x + lambda_y sigma^2_y

    to the output of the gaussian_NLL function, where lambda_i is half the
    squared scale paramater to the half-Normal. If we pick this scale parameter
    to be half the arena dimension in each direction, we express the belief
    that about 68% of the time, the variance in the x and y direction should be
    less than half the arena size.
    """
    NLL_term = gaussian_NLL(pred, target)

    cholesky_cov = pred[:, 1:]
    cov = cholesky_cov @ cholesky_cov.swapaxes(-2, -1)

    variance_x = cov[:, 0, 0]
    variance_y = cov[:, 1, 1]

    x_dim, y_dim = arena_dims

    get_lambda = lambda dimension: 1 / 2 * (dimension**2)

    lambda_x = get_lambda(x_dim)
    lambda_y = get_lambda(y_dim)

    return NLL_term + (lambda_x * variance_x) + (lambda_y * variance_y)


def gaussian_NLL_entropy_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    penalty: float = 1.0,
):
    """
    Regularized version of gaussian_NLL, with an entropy penalty.
    """
    # make sure that preds has shape: (B, 3, 2) where B is the batch size
    if pred.ndim != 3:
        raise ValueError(
            "Expected `pred` to have shape (B, 3, 2) where B is the batch size, "
            f"but encountered shape: {pred.shape}"
        )

    y_hat = pred[:, 0]  # output, shape: (B, 2)
    L = pred[:, 1:]  # cholesky factor of covariance, shape: (B, 2, 2)

    # create the distribution corresponding to the outputted predictive
    # density
    multivariate_normal = MultivariateNormal(loc=y_hat, scale_tril=L)

    loss = -multivariate_normal.log_prob(target) + (
        penalty * multivariate_normal.entropy()
    )

    return loss
