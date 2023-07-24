import logging
from math import comb

import torch
from torch import nn

from gerbilizer.architectures.base import GerbilizerArchitecture
from gerbilizer.outputs import ModelOutputFactory

logging.basicConfig(level=logging.INFO)

"""
resnet for 1-d signal data, pytorch version
 
Shenda Hong, Oct 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_spec_norm=False, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )
        if use_spec_norm:
            self.conv = nn.utils.parametrizations.spectral_norm(self.conv)

    def forward(self, x):
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net

class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)
    def forward(self, x):
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.max_pool(net)
        return net

class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
        use_spec_norm=False,
    ):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
            use_spec_norm=use_spec_norm
        )
        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
            use_spec_norm=use_spec_norm
        )
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)
        # shortcut
        out += identity
        return out

class GatedBasicBlock(BasicBlock):
    """
    ResNet Basic Block
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        downsample,
        use_bn,
        use_do,
        is_first_block=False,
        use_spec_norm=False,
    ):
        super(GatedBasicBlock, self).__init__(
            in_channels, out_channels, kernel_size, stride, groups, downsample,
            use_bn, use_do, is_first_block, use_spec_norm
            )
        self.left_conv1 = self.conv1
        self.left_conv2 = self.conv2

        self.right_conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            groups=self.groups,
            use_spec_norm=use_spec_norm
        )

        self.right_conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=self.groups,
            use_spec_norm=use_spec_norm
        )

    def forward(self, x):
        identity = x
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)

        left_out = torch.tanh(self.left_conv1(out))
        right_out = torch.sigmoid(self.right_conv1(out))

        out = left_out * right_out

        # the second conv
        if self.use_bn:
            out = self.bn2(out)

        if self.use_do:
            out = self.do2(out)

        left_out = torch.tanh(self.left_conv2(out))
        right_out = torch.sigmoid(self.right_conv2(out))

        out = left_out * right_out

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity
        return out

class ResNet1D(GerbilizerArchitecture):
    """
    Input:
        X: (n_samples, n_length, n_channel)
        Y: (n_samples)
    Output:
        out: (n_samples)

    Parameters:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes

    """

    defaults = {
        "USE_BATCH_NORM": True,
        "SHOULD_DOWNSAMPLE": [False, True, True, True, True, True, False],
        "CONV_BASE_KERNEL_SIZE": 16,
        "CONV_BASE_NUM_CHANNELS": 16,
        "CONV_STRIDE": 2,
        "USE_SPECTRAL_NORM": False,
        "NONLINEARITY": "ReLU" # one of ReLU, Gated
    }


    def __init__(
        self,
        CONFIG: dict,
        output_factory: ModelOutputFactory,
    ):
        super(ResNet1D, self).__init__(CONFIG, output_factory)

        N = CONFIG["DATA"]["NUM_MICROPHONES"]

        if CONFIG["DATA"].get("COMPUTE_XCORRS", False):
            N += comb(N, 2)

        # Obtains model-specific parameters from the config file and fills in missing entries with defaults
        model_config = ResNet1D.defaults.copy()
        model_config.update(CONFIG.get("MODEL_PARAMS", {}))
        CONFIG[
            "MODEL_PARAMS"
        ] = model_config  # Save the parameters used in this run for backward compatibility

        in_channels = N
        base_filters = model_config["CONV_BASE_NUM_CHANNELS"]
        kernel_size = model_config["CONV_BASE_KERNEL_SIZE"]
        stride = model_config["CONV_STRIDE"]
        groups = model_config.get("groups" , 8)
        n_block = model_config.get("n_block" , 8)
        downsample_gap = 2
        increasefilter_gap = 4
        use_bn = model_config["USE_BATCH_NORM"]
        use_do = False
        verbose = False
        nonlinearity = model_config["NONLINEARITY"]

        use_spec_norm = model_config.get("USE_SPECTRAL_NORM", False)

        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        if nonlinearity == "ReLU":
            self.basic_block = BasicBlock
        elif nonlinearity == "Gated":
            self.basic_block = GatedBasicBlock
        else:
            raise ValueError(
                f"Unrecognized value for `nonlinearity`: {nonlinearity}! "
                "Expected one of 'ReLU', 'Gated'."
                )

        self.downsample_gap = downsample_gap  # 2 for base model
        self.increasefilter_gap = increasefilter_gap  # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=base_filters,
            kernel_size=self.kernel_size,
            stride=1,
            use_spec_norm=use_spec_norm
        )
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(
                    base_filters * 2 ** ((i_block - 1) // self.increasefilter_gap)
                )
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels

            tmp_block = self.basic_block(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=downsample,
                use_bn=self.use_bn,
                use_do=self.use_do,
                is_first_block=is_first_block,
                use_spec_norm=use_spec_norm
            )
            self.basicblock_list.append(tmp_block)
        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        if not isinstance(self.n_outputs, int):
            raise ValueError(
                "Number of parameters to output is undefined! Maybe check the model configuration and ModelOutputFactory object?"
            )
        self.dense = nn.Linear(out_channels, self.n_outputs)
        # self.softmax = nn.Softmax(dim=1)

    def _forward(self, x: torch.Tensor, return_hidden: bool = False):
        # (batch, seq_len, channels) -> (batch, channels, seq_len) needed by conv1d
        out = x.transpose(-1, -2) 

        # first conv
        if self.verbose:
            print("input shape", out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print("after first conv", out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print(
                    "i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}".format(
                        i_block, net.in_channels, net.out_channels, net.downsample
                    )
                )
            out = net(out)
            if self.verbose:
                print(out.shape)
        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print("final pooling", out.shape)
        # # out = self.do(out)
        if return_hidden:
            return out
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # # out = self.softmax(out)
        # if self.verbose:
        #     print('softmax', out.shape)
        return out

    def clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)
