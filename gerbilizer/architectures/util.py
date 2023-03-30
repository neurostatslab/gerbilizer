"""
Common utility functions for things like embedding covariance model output into
an array containing the outputted mean and Cholesky covariance factor.
"""

import torch
from torch.nn import functional as F


def build_cov_output(raw_output, device):
    """
    Given a batch of length-5 vectors, return a batch of (3, 2) arrays
    where for each array, arr[0] is the mean, and arr[1:] is the Cholesky
    factor of the covariance matrix.
    """
    y_hat = raw_output[:, :2]  # (batch, 2)
    # calculate the lower triangular Cholesky factor
    # of the covariance matrix
    len_batch = raw_output.shape[0]
    # initialize an array of zeros into which we'll put
    # the model output
    L = torch.zeros(len_batch, 2, 2, device=device)
    # embed the elements into the matrix
    idxs = torch.tril_indices(2, 2)
    L[:, idxs[0], idxs[1]] = raw_output[:, 2:]
    # apply softplus to the diagonal entries to guarantee the resulting
    # matrix is positive definite
    new_diagonals = F.softplus(L.diagonal(dim1=-2, dim2=-1))
    L = L.diagonal_scatter(new_diagonals, dim1=-2, dim2=-1)
    # reshape y_hat so we can concatenate it to L
    y_hat = y_hat.reshape((-1, 1, 2))  # (batch, 1, 2)
    # concat the two to make a (batch, 3, 2) tensor
    concatenated = torch.cat((y_hat, L), dim=-2)
    return concatenated
