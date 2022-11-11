import numpy as np
import torch
from torch import nn

# Positional Encoding module copied from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
# and modified slightly to accomodate a batch-first data shape
class FixedEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len=8192, transpose=False):
        super(FixedEncoding, self).__init__()
        self.transpose = transpose

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if transpose:
            pe = pe.T
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if self.transpose:
            x = x + self.pe[None, :, : x.shape[2]]
        else:
            x = x + self.pe[None, : x.shape[1], :]
        return x


class LearnedEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, transpose=False):
        super().__init__()
        enc = torch.empty((max_seq_len, d_model))
        if transpose:
            enc = enc.T
        self.transpose = transpose
        nn.init.normal_(enc)
        self.encoding = nn.Parameter(enc, requires_grad=True)

    def forward(self, x):
        if self.transpose:
            return x + self.encoding[:, : x.shape[-1]].unsqueeze(0)
        return x + self.encoding[: x.shape[1], :].unsqueeze(0)


class LearnedImageEncoding(nn.Module):
    def __init__(self, d_model, max_width, max_height):
        super().__init__()
        enc = torch.empty((max_height, max_width, d_model))
        nn.init.normal_(enc)
        self.encoding = nn.Parameter(enc, requires_grad=True)

    def forward(self, x):
        height, width = x.shape[-3], x.shape[-2]
        return x + self.encoding[:height, :width, :].unsqueeze(0)