from itertools import combinations
from math import comb
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from gerbilizer.architectures.base import GerbilizerArchitecture
from gerbilizer.outputs import ModelOutputFactory

from .simplenet import GerbilizerSimpleLayer


class SkipConnection(torch.nn.Module):
    def __init__(self, submodule: nn.Module):
        super(SkipConnection, self).__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class CorrSimpleNetwork(GerbilizerArchitecture):
    defaults = {
        "USE_BATCH_NORM": True,
        "SHOULD_DOWNSAMPLE": [False, True] * 5,
        "CONV_FILTER_SIZES": [33] * 10,
        "CONV_NUM_CHANNELS": [16, 16, 32, 32, 64, 64, 128, 128, 256, 256],
        "CONV_DILATIONS": [1] * 10,
        "OUTPUT_COV": True,
        "REGULARIZE_COV": False,
        "CPS_NUM_LAYERS": 3,
        "CPS_HIDDEN_SIZE": 1024,
    }

    def __init__(self, CONFIG, output_factory: ModelOutputFactory):
        super(CorrSimpleNetwork, self).__init__(CONFIG, output_factory)

        N = CONFIG["DATA"]["NUM_MICROPHONES"]

        self.xcorr_length = 256

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = CorrSimpleNetwork.defaults.copy()
        model_config.update(CONFIG.get("MODEL_PARAMS", {}))
        CONFIG["MODEL_PARAMS"] = (
            model_config  # Save the parameters used in this run for backward compatibility
        )

        should_downsample = model_config["SHOULD_DOWNSAMPLE"]
        self.n_channels = model_config["CONV_NUM_CHANNELS"]
        filter_sizes = model_config["CONV_FILTER_SIZES"]
        dilations = model_config["CONV_DILATIONS"]

        min_len = min(
            len(self.n_channels),
            len(filter_sizes),
            len(should_downsample),
            len(dilations),
        )
        self.n_channels = self.n_channels[:min_len]
        filter_sizes = filter_sizes[:min_len]
        should_downsample = should_downsample[:min_len]
        dilations = dilations[:min_len]

        use_batch_norm = model_config["USE_BATCH_NORM"]

        self.n_channels.insert(0, N)

        convolutions = [
            GerbilizerSimpleLayer(
                in_channels,
                out_channels,
                filter_size,
                downsample=downsample,
                dilation=dilation,
                use_bn=use_batch_norm,
            )
            for in_channels, out_channels, filter_size, downsample, dilation in zip(
                self.n_channels[:-1],
                self.n_channels[1:],
                filter_sizes,
                should_downsample,
                dilations,
            )
        ]
        self.conv_layers = torch.nn.Sequential(*convolutions)

        self.final_pooling = nn.AdaptiveAvgPool1d(1)

        # layers for the cps branch of the network:
        cps_initial_channels = comb(N, 2) * 256  # Number of microphone pairs
        cps_num_layers = model_config["CPS_NUM_LAYERS"]
        cps_hidden_size = model_config["CPS_HIDDEN_SIZE"]

        # Add a batch norm to handle scaling the cps
        cps_network = [
            nn.BatchNorm1d(cps_initial_channels),
            nn.Linear(cps_initial_channels, cps_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(cps_num_layers - 1):
            cps_network.append(
                SkipConnection(
                    nn.Sequential(
                        nn.Linear(cps_hidden_size, cps_hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        self.cps_network = nn.Sequential(*cps_network)

        if not isinstance(self.n_outputs, int):
            raise ValueError(
                "Number of parameters to output is undefined! Maybe check the model configuration and ModelOutputFactory object?"
            )
        self.coord_readout = nn.Sequential(
            nn.Linear(
                self.n_channels[-1] + cps_hidden_size,
                self.n_channels[-1] + cps_hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(self.n_channels[-1] + cps_hidden_size, self.n_outputs),
        )

    def make_cps(self, audio: torch.Tensor):
        # audio shape: (batch, n_samples, n_channels)
        # Generates the cross power spectrum of the audio channels in a pairwise fasion

        output_length = self.xcorr_length

        seq_len, n_channels = audio.shape[-2:]

        # Get all pairwise combinations of channels
        channel_pairs = list(combinations(range(n_channels), 2))
        channel_pairs = torch.tensor(channel_pairs)  # for 4 mics, shape is (6, 2)

        rfft_audio = torch.fft.rfft(audio, dim=-2)  # (batch, n_rfreq, n_channels)

        lhs = rfft_audio[:, :, channel_pairs[:, 0]]
        rhs = rfft_audio[:, :, channel_pairs[:, 1]]
        f_xcorr = lhs * rhs.conj()  # both complex, shape is (batch, n_rfreq, n_pairs)
        cps = torch.fft.irfft(f_xcorr, dim=-2)
        # shape is (batch, n_samples, n_pairs)

        # Take the central 256 samples
        center = cps.shape[-2] // 2
        width = output_length
        low_end = center - width // 2
        high_end = center + width // 2
        cps = cps[..., low_end:high_end, :]
        # shape is (batch, 256, n_pairs)

        return cps.flatten(start_dim=1)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        audio = x
        cps = self.make_cps(audio)

        audio = audio.transpose(
            -1, -2
        )  # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        h1 = self.conv_layers(audio)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)

        cps_branch = self.cps_network(cps)
        h2 = torch.cat((h2, cps_branch), dim=-1)

        coords = self.coord_readout(h2)
        return coords

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
