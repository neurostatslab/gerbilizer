import enum
import sys
from typing import List, Optional, Union

import torch


class Unit(enum.Enum):
    """
    Enum type for the different units understood by the program.
    """

    ARBITRARY = enum.auto()
    CM = enum.auto()
    MM = enum.auto()


class ModelOutput:
    """
    Base class packaging the raw Tensor output from a deep neural network
    with information about its unit and potentially the way it parameterizes
    a (batch of) response distributions, if probabilistic.
    """

    def __init__(
        self,
        raw_output,
        arena_dims: torch.Tensor,
        arena_dim_units: Union[str, Unit],
    ):
        """Initialize a ModelOutput object."""
        self.raw_output = raw_output
        if isinstance(raw_output, torch.Tensor):
            self.batch_size = raw_output.shape[0]
            self.device = raw_output.device
        else:
            self.device = raw_output[0].raw_output.device

        arena_dims = arena_dims.to(self.device)

        if isinstance(arena_dim_units, str):
            try:
                arena_dim_units = Unit[arena_dim_units]
            except KeyError:
                raise ValueError(
                    "Arena dimensions must be provided in either centimeters or millimeters! "
                    f"Instead encountered unit: {arena_dim_units}."
                )

        self.arena_dims = {
            Unit.MM: self._convert(arena_dims, arena_dim_units, Unit.MM),
            Unit.CM: self._convert(arena_dims, arena_dim_units, Unit.CM),
        }

    def _convert(
        self, x: torch.Tensor, in_units: Unit, out_units: Unit
    ) -> torch.Tensor:
        """
        Convert an input array-like representing a (collection of) point(s)
        from one unit to another.
        """
        if in_units == out_units:
            return x
        elif in_units == Unit.MM and out_units == Unit.CM:
            return x / 10
        elif in_units == Unit.CM and out_units == Unit.MM:
            return x * 10
        elif in_units == Unit.ARBITRARY:
            return (x + 1) * 0.5 * self.arena_dims[out_units].to(x.device)
        elif out_units == Unit.ARBITRARY:
            return 2 * (x / self.arena_dims[in_units].to(x.device)) - 1
        else:
            raise ValueError(
                "Expected both `in_units` and `out_units` to be instances of "
                f"the Unit class. Instead encountered in_units: {in_units} and "
                f"out_units: {out_units}."
            )

    def _point_estimate(self):
        """Return a single point estimate in arbitrary units."""
        raise NotImplementedError

    def point_estimate(self, units: Unit = Unit.ARBITRARY):
        """Return a single point estimate in the specified units."""
        return self._convert(self._point_estimate(), Unit.ARBITRARY, units)


class PointOutput(ModelOutput):
    """
    Class representing a batch of point estimates.
    """

    N_OUTPUTS_EXPECTED = 2

    config_name = "POINT"

    def _point_estimate(self):
        return torch.clamp(self.raw_output, -1, 1)


class ProbabilisticOutput(ModelOutput):
    """
    Base class packaging the raw Tensor output from a deep neural network that
    parameterizes a (batch of) distributions with its unit and choice of
    parameterization.
    """

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return log p(x) under the distribution p parameterized by the model
        output, where x is an array-like with expected shape (..., 2). Importantly,
        x is expected to be provided in arbitrary units (i.e. on the square [-1, 1]^2).
        """
        raise NotImplementedError

    def log_p(self, x: torch.Tensor, units: Unit) -> torch.Tensor:
        """
        Return log p(x) for x provided in units `units`.
        """
        # convert coordinate grid to arbitrary units
        x = x.to(self.device)
        x_arbitrary = self._convert(x, units, Unit.ARBITRARY)
        log_prob = self._log_p(x_arbitrary)
        # scale probability accordingly
        # jacobian determinant of the affine transform from [-1, 1]^2 \to
        # [0,xdim] x [0, ydim] is xdim * ydim / 4, so we divide the density by
        # that.
        scale_factor = self.arena_dims[units].prod() / 4
        # equivalently, subtract the log of the scale factor from log_prob.
        return log_prob - torch.log(scale_factor)

    def pmf(self, coordinate_grid: torch.Tensor, units: Unit) -> torch.Tensor:
        """
        Calculate p(x) at each point on the coordinate grid for the
        distribution p parameterized by this model output instance, in a
        vectorized and numerically stable way. Normalizes to sum to 1.
        """
        probs = torch.exp(self.log_p(coordinate_grid, units=units))
        # normalize to 1
        return probs / probs.sum()


class BaseDistributionOutput(ProbabilisticOutput):
    """
    Utility subclass expressing that the model output
    parameterizes a single distribution rather than a mixture.
    """

    N_OUTPUTS_EXPECTED: int


class UniformOutput(BaseDistributionOutput):
    N_OUTPUTS_EXPECTED = 0
    config_name = "UNIFORM"

    def __init__(
        self,
        raw_output: None,
        arena_dims: torch.Tensor,
        arena_dim_units: Union[str, Unit],
        batch_size: int,
        device: str,
    ):
        """
        Inits UniformOutput.

        Args:
            raw_output: expected to be None. the argument is kept for consistency
                with other init methods, but it's ignored even if provided.
            arena_dims: array-like (x, y) storing arena dimensions in either CM or MM
            arena_dim_units: str or Unit indicating in which unit the arena dimensions
                are provided.
            batch_size: batch size. usually inferred from model output shape,
                but since no output is expected or considered here, it must be
                provided separately.
            device: device on which outputs should be stored. usually inferred as well.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)
        self.batch_size = batch_size
        self.device = device

    def _point_estimate(self):
        print(
            "Warning: method `_point_estimate` was called on a UniformOutput object. "
            "While supported for consistency, this method is almost nonsensical and "
            "should not be used in general.",
            file=sys.stderr,
        )
        return torch.zeros(self.batch_size, 2, device=self.device)

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.batch_size:
            raise ValueError(
                f"Incorrect shape for input! Since batch size is {self.batch_size}, "
                f"expected second-to-last dim of input `x` to have the same shape. Instead "
                f"found shape {x.shape}."
            )
        output_shape = x.shape[:-1]
        return torch.log(torch.tensor(0.25)) * torch.ones(output_shape, device=x.device)


class MDNOutput(ProbabilisticOutput):
    """
    Class storing the output of a mixture density network, which
    assumes a mixture response and parameterizes both the constituent
    distributions of the mixture as well as the weights by which the constituent
    distributions should be combined.
    """

    config_name = "MDN"

    def __init__(
        self,
        raw_output: torch.Tensor,
        arena_dims: torch.Tensor,
        arena_dim_units: Union[str, Unit],
        constituent_response_types: List[type[BaseDistributionOutput]],
        constituent_extra_kwargs: Optional[dict[int, dict]] = None,
    ):
        """
        Inits MDNOutput.

        Args:
            raw_output: batched tensor output from a GerbilizerArchitecture model
            arena_dims: array-like (x, y) storing arena dimensions in either CM or MM
            arena_dim_units: str or Unit indicating which unit the arena dimensions are provided in
            constituent_response_types: a list of class names indicating the distributions
                that are assumed to be components of a mixture response, with mixture weights specified
                on a per-stimulus basis by the model.
            constituent_extra_kwargs: an OPTIONAL extra dictionary mapping integer indices
                of the list `constituent_response_types` to keyword argument dictionaries
                that should be passed to the constituent distributions' constructors.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)

        # sum the number of parameters each constituent response distribution expects
        total_parameters_for_constituents = sum(
            t.N_OUTPUTS_EXPECTED for t in constituent_response_types
        )
        # then add on the number of mixing weights expected (R for R
        # the number of response distributions)
        num_responses = len(constituent_response_types)
        n_parameters_expected = total_parameters_for_constituents + num_responses

        if self.raw_output.shape[-1] != n_parameters_expected:
            raise ValueError(
                f"Given constituent response types {constituent_response_types}, expected "
                f"to recieve {n_parameters_expected} parameters per stimulus, with "
                f"{total_parameters_for_constituents} params to specify the constituent "
                f"distributions, and {num_responses} params to specify the mixing weights. "
                f"Instead encountered {raw_output.shape[-1]}, with raw_output shape "
                f"of {raw_output.shape}."
            )

        # we don't set `constituent_extra_kwargs` to the empty dict by default
        # because that's dangerous and stateful in python.
        if not constituent_extra_kwargs:
            constituent_extra_kwargs = {}

        curr_idx = 0
        self.responses = []
        for i, response_type in enumerate(constituent_response_types):
            n_params = response_type.N_OUTPUTS_EXPECTED
            next_idx = curr_idx + n_params

            additional_kwargs = constituent_extra_kwargs.get(i, {})

            if response_type == UniformOutput:
                additional_kwargs["batch_size"] = self.batch_size
                additional_kwargs["device"] = self.device

            response = response_type(
                raw_output[:, curr_idx:next_idx],
                arena_dims,
                arena_dim_units,
                **additional_kwargs,
            )
            self.responses.append(response)
            curr_idx = next_idx

        # the parameters representing the weights are expected to be
        # passes as unnormalized logits. call these w_ij for i the batch index
        # and j the logit for response distribution j.
        unnormalized_logits = raw_output[:, curr_idx:]
        # the actual weights theta_i are given by
        # theta_ij = exp(w_ij) / [ \sum_k exp(w_ik) ]
        # but we're ultimately going to work on log scale, so
        # instead compute log theta)ij = w_ij - log(sum_k exp(w_ik))
        normalizer = torch.logsumexp(unnormalized_logits, -1, keepdim=True)
        self.log_weights = unnormalized_logits - normalizer

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        # with the log weights log(theta_ij) defined in the init
        # method, we can now compute the mixture density evaluated at a
        # certain (collection of) point(s)
        if x.shape[-2] != self.batch_size:
            raise ValueError(
                f"Incorrect shape for input! Since batch size is {self.batch_size}, "
                f"expected second-to-last dim of input `x` to have the same shape. Instead "
                f"found shape {x.shape}."
            )
        # stacks R tensors each of shape (..., self.batch_size)
        # into one big tensor of shape (..., self.batch_size, R)
        # this is for compatibility with self.log_weights, which has
        # shape (..., self.batch_size)
        individual_log_probs = torch.stack([r._log_p(x) for r in self.responses], -1)
        # the result is given by p_i = \sum_j theta_ij p_ij
        # for p_i the mixture density for batch element i,
        # and p_ij the probability assigned from response j
        # calculating this using only the log probs and log weights,
        # the formula becomes
        # log p_i = log \sum_j exp( log theta_ij + log p_ij)

        # tensor shape wise, this operation reduces along the last dimension
        # across the different responses as expected
        #     (..., self.batch_size, R) --> (..., self.batch_size)
        return torch.logsumexp(self.log_weights + individual_log_probs, -1)

    def _point_estimate(self):
        # average the component distribution's point estimates
        # throwing out any uniform distributions because a point estimate
        # from one is sort of nonsensical
        to_include = []
        for response in self.responses:
            if not isinstance(response, UniformOutput):
                to_include.append(response.point_estimate())

        point_estimates = torch.stack(to_include, dim=1)  # (self.batch_size, R, 2)
        weights = torch.exp(self.log_weights)  # (self.batch_size, R)
        reweighted_estimates = (
            point_estimates * weights[..., None]
        )  # (self.batch_size, R, 2)

        return reweighted_estimates.sum(dim=1)  # (self.batch_size, 2)


class EnsembleOutput(ProbabilisticOutput):
    """
    Class storing the output of an ensemble of separate models.
    """

    config_name = "ENSEMBLE"

    def __init__(
        self,
        raw_output: List[ProbabilisticOutput],
        arena_dims: torch.Tensor,
        arena_dim_units: Union[str, Unit],
    ):
        """
        Initialize an object representing a mixture density created by
        averaging multiple separate probabilistic outputs, oftentimes from
        separate models.
        """
        super().__init__(raw_output, arena_dims, arena_dim_units)
        # make sure all constituent distributions have the same batch size
        self.distributions = raw_output
        self.n_distributions = len(self.raw_output)
        self.batch_size = self.raw_output[0].batch_size

        for output in self.raw_output:
            if output.batch_size != self.batch_size:
                raise ValueError(
                    "Not all input model output objects have the same batch size! "
                    f"The encountered batch sizes are: {[o.batch_size for o in self.raw_output]}."
                )

    def _log_p(self, x: torch.Tensor) -> torch.Tensor:
        # output probability should be the average of the individual
        # distribution's probabilities. say i is batch index and p_ij is
        # probability from distribution j on batch element i
        # log p_i = log (1/N \sum_j p_ij) = log(1/N) + logsumexp(p_i1, ..., p_iN)
        # get each p_ij
        individual_log_probs = torch.stack(
            [r._log_p(x) for r in self.distributions], -1
        )
        # shape (..., self.batch_size, self.n_distributions) |--> (..., self.batch_size)
        unnormalized = torch.logsumexp(individual_log_probs, -1)
        normalizer = torch.log(torch.tensor(1 / self.n_distributions, device=x.device))
        return normalizer + unnormalized

    def _point_estimate(self):
        # average the component distribution's point estimates
        # throwing out any uniform distributions because a point estimate
        # from one is sort of nonsensical
        to_include = []
        for distr in self.distributions:
            if not isinstance(distr, UniformOutput):
                to_include.append(distr.point_estimate())
        return sum(to_include) / len(to_include)
