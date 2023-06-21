"""Initialize model and loss function from configuration."""
from collections import namedtuple
from functools import partial
from typing import Any, Callable, Tuple, Union

import numpy as np
import torch

from ..architectures.attentionnet import GerbilizerTransformer
from ..architectures.densenet import GerbilizerDenseNet
from ..architectures.ensemble import GerbilizerEnsemble
from ..architectures.simplenet import GerbilizerSimpleNetwork
from .losses import (
    se_loss_fn,
    map_se_loss_fn,
    gaussian_NLL,
    gaussian_NLL_half_normal_variances,
    gaussian_NLL_entropy_penalty,
    wass_loss_fn,
)

ModelType = namedtuple("ModelType", ("model", "non_cov_loss_fn", "can_output_cov"))

LOOKUP_TABLE = {
    "GerbilizerDenseNet": ModelType(GerbilizerDenseNet, se_loss_fn, True),
    "GerbilizerSimpleNetwork": ModelType(GerbilizerSimpleNetwork, se_loss_fn, True),
    "GerbilizerTransformer": ModelType(GerbilizerTransformer, wass_loss_fn, True),
}


def build_model(config: dict[str, Any]) -> tuple[torch.nn.Module, Callable]:
    """
    Specifies model and loss funciton.

    Parameters
    ----------
    config : dict
        Dictionary holding training hyperparameters.

    Returns
    -------
    model : torch.nn.Module
        Model instance with hyperparameters specified
        in config.

    loss_function : function
        Loss function mapping network output to a
        per-instance. That is, loss_function should
        take a torch.Tensor with shape (batch_size, ...)
        and map it to a shape of (batch_size,) holding
        losses.
    """
    arch = config["ARCHITECTURE"]
    if arch in LOOKUP_TABLE:
        model, loss_fn, can_output_cov = LOOKUP_TABLE[arch]
        model = model(config)
    elif arch == "GerbilizerEnsemble":
        # None out the other parameters
        loss_fn, can_output_cov = None, None
        built_submodels = []
        for sub_model_config in config["MODELS"]:
            submodel, _ = build_model(sub_model_config)
            built_submodels.append(submodel)
        model = GerbilizerEnsemble(config, built_submodels)
    else:
        raise ValueError(f"ARCHITECTURE {arch} not recognized.")

    loss = loss_fn

    if loss == wass_loss_fn:
        # Converted from mm to m for stability
        arena_dims = np.array(config["DATA"]["ARENA_DIMS"]) / 1000
        transport_cost = partial(transport_cost_fn, width=40, height=30)
        loss = partial(wass_loss_fn, transport_cost=transport_cost)

    # change the loss function depending on whether a model outputting covariance
    # was chosen
    if config.get("MODEL_PARAMS", {}).get("OUTPUT_COV", True):
        # some models, like a model that outputs a 2d map, don't have the ability to output a
        # cov matrix.
        if not can_output_cov:
            raise ValueError(
                f"Flag `OUTPUT_COV` was passed to a model architecture that can't output a covariance matrix ({arch})."
            )

        loss = gaussian_NLL

        # change the loss function depending on whether the "REGULARIZE_COV" parameter was provided
        # in the JSON config.
        if config.get("MODEL_PARAMS", {}).get("REGULARIZE_COV", False):
            reg = config["MODEL_PARAMS"]["REGULARIZE_COV"]
            if reg == "HALF_NORMAL":
                loss = gaussian_NLL_half_normal_variances
            elif reg == "ENTROPY":
                loss = gaussian_NLL_entropy_penalty
            else:
                raise ValueError(
                    f"Unrecognized value {reg} passed as `REGULARIZE_COV` parameter in model config!"
                )

    return model, loss


def transport_cost_fn(target: torch.Tensor, width: int, height: int) -> torch.Tensor:
    arena_dims = [2, 2]
    spacing_x = arena_dims[0] / width
    spacing_y = arena_dims[1] / height
    x_points = np.linspace(-arena_dims[0] / 2 + spacing_x / 2, arena_dims[0] / 2 - spacing_x / 2, width, endpoint=True)  # evaluate at center of each grid cell
    y_points = np.linspace(-arena_dims[1] / 2 + spacing_y / 2, arena_dims[1] / 2 - spacing_y / 2, height, endpoint=True)
    x_grid, y_grid = np.meshgrid(x_points, y_points)
    grid = np.stack([x_grid, y_grid], axis=-1).astype(np.float32)
    grid = torch.from_numpy(grid).float().to(target.device)
    # fill each cell with the distance between the cell center and the target
    # grid is of shape (height, width, 2)
    # target is of shape (n_samples, 2)
    dists = torch.norm(target[:, None, None, :] - grid[None, :, :, :], dim=-1, p=2)  # yields shape (n_samples, height, width)

    # warning: if displayed as an image, this will be upside down
    return dists


def __apply_affine(locations: np.ndarray, A: np.ndarray, b: np.ndarray):
    """
    Helper function to apply an affine transformation Ax + b
    to a list of vectors.

    Args:
        locations: Array of shape (n_samples, k)
        A: Array of shape (k, k)
        b: Array of shape (k,)
    """
    return (A @ locations[..., None]).squeeze() + b


def unscale_output(
    model_output: np.ndarray, arena_dims: Union[tuple[float, float], np.ndarray]
) -> np.ndarray:
    """
    Transform model output from arbitrary units on the square [-1, 1]^2 to
    the size of the arena, accounting correctly for the shape of the input.

    Note: expects the first dimension of model_output to be the number of
    samples per vocalization.

    This transformation and recentering is given by the affine transform

        z_i = Ay_i + b,

    where z_i is the new location and

        b = 1/2 * (a_1, a_2),

        A = 1/2 * [[a_1 ,  0  ],
                   [0   ,  a_2]].
    """
    A = 0.5 * np.diag(arena_dims)  # rescaling matrix
    b = 0.5 * np.array(arena_dims)  # recentering vector

    unscaled = np.zeros(model_output.shape)

    # if the model outputs a mean y and a covariance matrix S, the transformed
    # mean is given by Ay + b, and the cov matrix is given by A @ S @ A.T
    # (result from the transformation of a random variable)
    if model_output.ndim == 3 and model_output.shape[1:] == (3, 2):
        means = model_output[:, 0]  # shape: (len(model_output), 2)
        cholesky = model_output[:, 1:]  # shape: (len(model_output), 2, 2)
        covs = cholesky @ cholesky.swapaxes(-1, -2)

        unscaled[:, 0] = __apply_affine(means, A, b)
        unscaled[:, 1:] = A @ covs @ A.T

    # similar if model outputs a batch of means + cholesky covariances
    # this is the case for ensemble models
    elif model_output.ndim == 4 and model_output.shape[2:] == (3, 2):
        means = model_output[:, :, 0]  # shape: (batch_size, n_models, 2)
        cholesky = model_output[:, :, 1:]  # shape: (batch_size, n_models, 2, 2)
        covs = cholesky @ cholesky.swapaxes(-1, -2)

        unscaled[:, :, 0] = __apply_affine(means, A, b)
        unscaled[:, :, 1:] = A @ covs @ A.T

    # otherwise, just apply the affine transformation
    elif model_output.ndim == 2 and model_output.shape[1] == 2:
        unscaled = __apply_affine(model_output, A, b)
    else:
        raise ValueError(
            f"Unscaling not currently supported for output of shape {model_output.shape}!"
        )

    return unscaled
