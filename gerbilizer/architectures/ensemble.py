import pathlib

import torch
from torch import nn

# from torch.nn import functional as F
from gerbilizer.training.configs import build_config

# inference only for now
class GerbilizerEnsemble(nn.Module):
    # config should just tell us model architectures
    def __init__(self, config, built_models):
        """
        Inputs the loaded config dictionary object, as well as each submodel
        built using `build_model`.
        """
        super().__init__()

        for model_config in config["MODELS"]:
            if "OUTPUT_COV" not in model_config:
                raise ValueError(
                    "Ensembling not yet available for models without uncertainty estimates."
                )

            if "WEIGHTS_PATH" not in model_config:
                raise ValueError(
                    "Cannot construct ensemble! Not all models have config paramter `WEIGHTS_PATH`."
                    )

        self.models = nn.ModuleList(built_models)
        self.average_outputs = bool(config.get("AVERAGE_OUTPUTS"))

#     @classmethod
#     def from_ensemble_dir(cls, ensemble_dir):
#         """
#         Load in an ensemble of models.
#         """
#         # load in jsons
#         config_paths = sorted(pathlib.Path(ensemble_dir).glob("*/config.json"))
#         # load in model weights
#         configs = []
#         for path in config_paths:
#             configs.append(build_config(str(path)))
#         # call regular constructor
#         ensemble = cls({"MODELS": configs})
#         # load state dict for each model
#         # for model, config in zip(ensemble.models, configs):
#         #     if "WEIGHTS_PATH" not in config:
#         #         raise ValueError(
#         #             "Cannot construct ensemble! Not all models have config paramter `WEIGHTS_PATH`."
#         #         )
#         #     model.load_state_dict(torch.load(config["WEIGHTS_PATH"]))
#         return ensemble

    def forward(self, x):
        # return mean and cholesky covariance of gaussian mixture
        # created from the ensemble of models
        batch_size = x.shape[0]
        means = torch.zeros(batch_size, len(self.models), 2)
        cholesky_covs = torch.zeros(batch_size, len(self.models), 2, 2)
        for i, model in enumerate(self.models):
            output = model(x)
            means[:, i] = output[:, 0]
            cholesky_covs[:, i] = output[:, 1:]

        # based on the model configuration we either just return these
        # means and covs
        if not self.average_outputs:
            # add extra dim so means is of shape (batch_size, len(self.models), 1, 2)
            return torch.cat((means[:, :, None], cholesky_covs), dim=-2)

        # or whether we make a "simplifying" assumption
        # as in (Lakshminarayanan, 2017) and say the mixture density is just
        # a Gaussian

        # calculate the mean and covariance matrix of the mixture density
        # mean is just the average of the means:
        mixture_means = means.mean(dim=1)

        # by introducing a new variable z representing the mixture
        # assignment, we can use the law of total variance to decompose the
        # covariance matrix as Var(Y) = E[Var(Y | Z = z)] + Var(E[Y | Z = z])

        # Cholesky factor of E[Var(Y|Z=z)]
        avg_cholesky_cov = cholesky_covs.mean(dim=1)
        # Var(E[Y | Z = z])
        cov_means_over_assigment = torch.zeros(batch_size, 2, 2)
        for i in range(batch_size):
            # take the transpose since torch.cov expects rows to be variables
            # and columns to be observations
            # here, one observation is one predicted mean from a given model
            cov_means_over_assigment[i] = torch.cov(means[i].T)
        # take the cholesky decomposition
        chol_cov_over_assignment = torch.linalg.cholesky(cov_means_over_assigment)
        # sum the two to find the full cholesky mixture covariance
        chol_mixture_cov = avg_cholesky_cov + chol_cov_over_assignment
        # concatenate mixture_means and chol_mixture_cov to make a (batch, 3, 2) tensor
        # reshape mixture_means to be (batch_size, 1, 2)
        return torch.cat((mixture_means[:, None], chol_mixture_cov), dim=-2)

