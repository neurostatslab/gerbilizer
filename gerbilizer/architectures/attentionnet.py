import torch
from torch import nn
from torch.nn import functional as F

from gerbilizer.architectures.util import build_cov_output


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

