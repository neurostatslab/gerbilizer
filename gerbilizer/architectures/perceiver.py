from math import comb
from typing import Optional

from numpy import sqrt
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from gerbilizer.architectures.simplenet import GerbilizerSimpleLayer
from gerbilizer.architectures.util import build_cov_output

from .encodings import FixedEncoding, LearnedEncoding


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        positional_encoding: Optional[nn.Module] = None,
        *,
        attn_steps_per_conv: int = 1,
        feedforward_dim: int = 1024,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()

        factory = {"dtype": dtype, "device": device}
        self.num_heads = nhead
        self.positional_encoding = positional_encoding
        self.attn_steps_per_conv = attn_steps_per_conv

        if (d_model % nhead) != 0:
            raise ValueError("Num_heads should divide d_model")

        self.sattn = nn.MultiheadAttention(d_model, nhead, batch_first=True, **factory)

        self.q_proj_weight = Parameter(torch.empty((d_model, d_model), **factory))
        self.k_proj_weight_T = Parameter(torch.empty((d_model, d_model), **factory))
        self.v_proj_weight = Parameter(torch.empty((d_model, d_model), **factory))

        self.q_proj_bias = Parameter(torch.empty((d_model,), **factory))
        self.k_proj_bias = Parameter(torch.empty((d_model,), **factory))
        self.v_proj_bias = Parameter(torch.empty((d_model,), **factory))

        self.out_proj = nn.Linear(d_model, d_model, **factory)

        self.norm1 = nn.LayerNorm((d_model,), **factory)
        self.norm2 = nn.LayerNorm((d_model,), **factory)
        self.norm3 = nn.LayerNorm((d_model,), **factory)

        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim, **factory),
            nn.ReLU(),
            nn.Linear(feedforward_dim, d_model, **factory),
        )

        self.conv = GerbilizerSimpleLayer(
            in_channels, out_channels, kernel_size, downsample=False, dilation=dilation
        )

        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.q_proj_weight)
        torch.nn.init.xavier_normal_(self.k_proj_weight_T)
        torch.nn.init.xavier_normal_(self.v_proj_weight)

        torch.nn.init.constant_(self.q_proj_bias, 0.0)
        torch.nn.init.constant_(self.k_proj_bias, 0.0)
        torch.nn.init.constant_(self.v_proj_bias, 0.0)

    def xattn(self, query, key, value):
        batch, mem_size, d_model = query.shape
        _, _, seq_len = key.shape  # (batch, d_model, seq_len)

        if key is not value:
            raise ValueError("Expects key and value to be the same Tensor")

        scale_factor = sqrt(1 / d_model)
        Q = (
            torch.matmul(query, self.q_proj_weight) + self.q_proj_bias[None, None, :]
        )  # Has shape (batch, M, d_model)
        K_T = (
            torch.matmul(self.k_proj_weight_T, key) + self.k_proj_bias[None, :, None]
        )  # (batch, D, L)
        V = (
            torch.einsum("bji,jk->bik", value, self.v_proj_weight)
            + self.v_proj_bias[None, None, :]
        )  # (batch, L, d_model)

        scores = (
            torch.einsum(
                "BMHD,BHDL->BHML",
                Q.view(batch, mem_size, self.num_heads, -1),
                K_T.view(batch, self.num_heads, -1, seq_len),
            )
            * scale_factor
        )  # Has shape (batch, num_heads, mem_size, seq_len)
        scores = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum(
            "BHML,BLHD->BMHD",
            scores.view(batch, self.num_heads, mem_size, seq_len),
            V.view(batch, seq_len, self.num_heads, -1),
        ).reshape(batch, mem_size, d_model)
        return self.out_proj(attn_output)  # Should be (batch, mem_size, d_model)

    def attn_step(self, memory, sequence):
        memory = self.norm1(
            memory + self.sattn(memory, memory, memory, need_weights=False)[0]
        )
        memory = self.norm2(memory + self.xattn(memory, sequence, sequence))
        memory = self.norm3(memory + self.ff(memory))

        return memory

    def forward(self, memory, sequence):
        # Memory has shape (batch, seq, d_model)
        # Unlike the standard transformer decoder, sequence is assumed to have shape (batch, d_model, seq) rather than (batch, seq, d_model)
        sequence = self.conv(sequence)

        if self.positional_encoding is None:
            encoded_sequence = sequence
        else:
            encoded_sequence = self.positional_encoding(sequence)

        for _ in range(self.attn_steps_per_conv):
            memory = self.attn_step(memory, encoded_sequence)

        return memory, sequence


class GerbilizerPerceiver(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_mics = config["NUM_MICROPHONES"]
        if config["COMPUTE_XCORRS"]:
            n_mics += comb(n_mics, 2)

        d_model = config[f"TRANSFORMER_D_MODEL"]
        self.d_model = d_model
        n_attn_heads = config["N_ATTENTION_HEADS"]
        n_transformer_layers = config["N_TRANSFORMER_LAYERS"]

        use_learned_encoding = config["POSITIONAL_ENCODING"].lower() == "learned"
        self.memory_size = config["PERCEIVER_MEMORY_SIZE"]

        kernel_size = config["PERCEIVER_KERNEL_SIZE"]
        dilation = config["PERCEIVER_DILATION"]
        attn_steps = config["PERCEIVER_ATTN_STEPS_PER_CONV"]

        max_seq_len = config["MAX_SEQ_LEN"]
        if use_learned_encoding:
            self.pos_encoding = LearnedEncoding(
                d_model=d_model, max_seq_len=max_seq_len, transpose=True
            )
        else:
            self.pos_encoding = FixedEncoding(
                d_model=d_model, max_len=max_seq_len, transpose=True
            )

        layers = [
            PerceiverLayer(
                d_model,
                n_attn_heads,
                kernel_size,
                n_mics,
                d_model,
                dilation,
                self.pos_encoding,
                attn_steps_per_conv=attn_steps,
            )
        ]
        layers.extend(
            [
                PerceiverLayer(
                    d_model,
                    n_attn_heads,
                    kernel_size,
                    d_model,
                    d_model,
                    dilation,
                    attn_steps_per_conv=attn_steps,
                )
                for _ in range(n_transformer_layers - 1)
            ]
        )
        self.layers = nn.ModuleList(layers)

        self.output_cov = bool(config.get("OUTPUT_COV"))
        N_OUTPUTS = 5 if self.output_cov else 2

        self.linear = nn.Sequential(
            nn.Linear(d_model, 1024), nn.ReLU(), nn.Linear(1024, N_OUTPUTS)
        )

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def trainable_params(self):
        return self.parameters()

    def forward(self, x):
        x = x.transpose(
            -1, -2
        )  # (batch, seq_len, channels) -> (batch, channels, seq len) needed by conv1d
        memory = torch.randn(
            (x.shape[0], self.memory_size, self.d_model), dtype=x.dtype, device=x.device
        )

        for layer in self.layers:
            memory, x = layer(memory, x)

        output = self.linear(memory[:, 0])

        return build_cov_output(output, x.device) if self.output_cov else output
