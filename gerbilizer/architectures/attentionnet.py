from itertools import chain
from math import comb

import torch
from torch import nn
from torch.nn import functional as F

from gerbilizer.architectures.util import build_cov_output

from .encodings import LearnedEncoding, FixedEncoding
from .sparse_attn import SparseTransformerEncoder, SparseTransformerEncoderLayer


class Transpose(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.a = dim_a
        self.b = dim_b

    def forward(self, tensor):
        return tensor.transpose(self.a, self.b)


class Skip(nn.Module):
    def __init__(self, submodule):
        super().__init__()
        self.submodule = submodule

    def forward(self, x):
        return x + self.submodule(x)


class LocalAttentionTransformerLayer(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        return x


class GerbilizerAttentionNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_mics = config["NUM_MICROPHONES"]
        if config["COMPUTE_XCORRS"]:
            n_mics += comb(n_mics, 2)
        d_model = config[f"CONV_NUM_CHANNELS"][0]
        dilation = config[f"CONV_DILATIONS"][0]
        stride = config[f"CONV_STRIDES"][0]
        kernel_size = config[f"CONV_FILTER_SIZES"][0]

        # Keeping the module list for compatibility with the existing state dictionaries
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=n_mics,
                out_channels=d_model,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding="same",
            )
        )

        n_transformer_layers = config["N_TRANSFORMER_LAYERS"]
        n_attn_heads = config["N_ATTENTION_HEADS"]

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_attn_heads, batch_first=True, activation="relu"
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, n_transformer_layers
        )
        # No dropout here since the TransformerEncoder implements its own dropout
        # Add 1 to account for the additional cls token appended in forward()
        # self.p_encoding = FixedEncoding(n_features, max_len=config['SAMPLE_LEN'] + 1)
        self.p_encoding = LearnedEncoding(
            d_model=d_model, max_seq_len=config["SAMPLE_LEN"] + 1
        )

        self.linear_dim = 1024

        self.dense = nn.Sequential(nn.Linear(d_model, self.linear_dim), nn.ReLU())

        self.output_cov = bool(config.get("OUTPUT_COV"))
        N_OUTPUTS = 5 if self.output_cov else 2

        self.coord_readout = nn.Sequential(
            nn.Linear(self.linear_dim, N_OUTPUTS), nn.Tanh()
        )

    def _clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def embed(self, x):
        """I'm splitting the forward pass into this 'embedding' segment
        and the final x-y coord output to make inheritance easier for the
        hourglass subclass.
        """
        conv_out = self.conv_layers[0](x)
        cls_token = torch.zeros(
            (conv_out.shape[0], conv_out.shape[1], 1), device=conv_out.device
        )
        conv_out = torch.cat([cls_token, conv_out], dim=2)
        # (batch, channel, seq) -> (batch, seq, channel)
        transformer_input = conv_out.transpose(1, 2)
        encoded = self.p_encoding(transformer_input)
        transformer_out = self.transformer(encoded)[:, 0, :]
        return self.dense(transformer_out)

    def forward(self, x):
        linear_out = self.embed(x)
        coords = self.coord_readout(linear_out)
        return build_cov_output(coords, x.device) if self.output_cov else coords

    def trainable_params(self):
        return self.parameters()


class GerbilizerSparseAttentionNet(GerbilizerAttentionNet):
    def __init__(self, config):
        super().__init__(config)

        del self.encoder_layer
        del self.transformer

        d_model = config[f"CONV_NUM_CHANNELS"][0]
        n_attn_heads = config["N_ATTENTION_HEADS"]
        n_transformer_layers = config["N_TRANSFORMER_LAYERS"]

        n_global = config["TRANSFORMER_GLOBAL_BLOCKS"]
        n_window = config["TRANSFORMER_WINDOW_BLOCKS"]
        n_random = config["TRANSFORMER_RANDOM_BLOCKS"]
        block_size = config["TRANSFORMER_BLOCK_SIZE"]

        encoder_layer = SparseTransformerEncoderLayer(
            d_model,
            n_attn_heads,
            block_size=block_size,
            n_global=n_global,
            n_window=n_window,
            n_random=n_random,
            dim_feedforward=2048,
            dropout=0.1,
        )

        self.transformer = SparseTransformerEncoder(encoder_layer, n_transformer_layers)

    def trainable_params(self):
        return self.parameters()
