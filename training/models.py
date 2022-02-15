import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_model(CONFIG):
    """
    Specifies model and loss funciton.

    Parameters
    ----------
    CONFIG : dict
        Dictionary holding training hyperparameters.

    Returns
    -------
    model : torch.nn.Module
        Model instance with hyperparameters specified
        in CONFIG.

    loss_function : function
        Loss function mapping network output to a
        per-instance. That is, loss_function should
        take a torch.Tensor with shape (batch_size, ...)
        and map it to a shape of (batch_size,) holding
        losses.
    """

    if CONFIG["ARCHITECTURE"] == "GerbilizerDenseNet":
        model = GerbilizerDenseNet(CONFIG)
    elif CONFIG["ARCHITECTURE"] == "GerbilizerSimpleNetwork":
        model = GerbilizerSimpleNetwork(CONFIG)
    elif CONFIG["ARCHITECTURE"] == "GerbilizerHourglassNet":
        model = GerbilizerHourglassNet(CONFIG)
    else:
        raise ValueError("ARCHITECTURE not recognized.")

    if CONFIG["ARCHITECTURE"] == "GerbilizerHourglassNet":
        def loss_function(x, y):
            return x * y
    else:
        def loss_function(x, y):
            return torch.mean(torch.square(x - y), axis=-1)

    return model, loss_function


class GerbilizerSimpleLayer(torch.nn.Module):

    def __init__(
            self, channels_in, channels_out, filter_size, *,
            downsample, dilation
        ):
        super(GerbilizerSimpleLayer, self).__init__()

        self.fc = torch.nn.Conv1d(
            channels_in, channels_out, filter_size,
            padding=(filter_size * dilation - 1) // 2,
            stride=(2 if downsample else 1),
            dilation=dilation
        )
        self.gc = torch.nn.Conv1d(
            channels_in, channels_out, filter_size,
            padding=(filter_size * dilation - 1) // 2,
            stride=(2 if downsample else 1),
            dilation=dilation
        )
        self.batch_norm = torch.nn.BatchNorm1d(channels_out)

    def forward(self, x):
        fcx = self.fc(x)
        return self.batch_norm(
            (torch.tanh(fcx) + 0.05 * fcx) * torch.sigmoid(self.gc(x))
        )

def ceiling_division(n, d):
    q, r = divmod(n, d)
    return q + bool(r)


class GerbilizerSimpleNetwork(torch.nn.Module):

    def __init__(self, CONFIG):
        super(GerbilizerSimpleNetwork, self).__init__()

        T = CONFIG["NUM_AUDIO_SAMPLES"]
        N = CONFIG["NUM_MICROPHONES"]

        self.conv_layers = torch.nn.Sequential(
            GerbilizerSimpleLayer(
                CONFIG["NUM_MICROPHONES"],
                CONFIG["NUM_CHANNELS_LAYER_1"],
                CONFIG["FILTER_SIZE_LAYER_1"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_1"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_1"],
                CONFIG["NUM_CHANNELS_LAYER_2"],
                CONFIG["FILTER_SIZE_LAYER_2"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_2"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_2"],
                CONFIG["NUM_CHANNELS_LAYER_3"],
                CONFIG["FILTER_SIZE_LAYER_3"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_3"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_3"],
                CONFIG["NUM_CHANNELS_LAYER_4"],
                CONFIG["FILTER_SIZE_LAYER_4"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_4"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_4"],
                CONFIG["NUM_CHANNELS_LAYER_5"],
                CONFIG["FILTER_SIZE_LAYER_5"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_5"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_5"],
                CONFIG["NUM_CHANNELS_LAYER_6"],
                CONFIG["FILTER_SIZE_LAYER_6"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_6"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_6"],
                CONFIG["NUM_CHANNELS_LAYER_7"],
                CONFIG["FILTER_SIZE_LAYER_7"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_7"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_7"],
                CONFIG["NUM_CHANNELS_LAYER_8"],
                CONFIG["FILTER_SIZE_LAYER_8"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_8"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_8"],
                CONFIG["NUM_CHANNELS_LAYER_9"],
                CONFIG["FILTER_SIZE_LAYER_9"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_9"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_9"],
                CONFIG["NUM_CHANNELS_LAYER_10"],
                CONFIG["FILTER_SIZE_LAYER_10"],
                downsample=False,
                dilation=CONFIG["DILATION_LAYER_10"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_10"],
                CONFIG["NUM_CHANNELS_LAYER_11"],
                CONFIG["FILTER_SIZE_LAYER_11"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_11"]
            ),
            GerbilizerSimpleLayer(
                CONFIG["NUM_CHANNELS_LAYER_11"],
                CONFIG["NUM_CHANNELS_LAYER_12"],
                CONFIG["FILTER_SIZE_LAYER_12"],
                downsample=True,
                dilation=CONFIG["DILATION_LAYER_12"]
            ),
        )

        self.final_pooling = torch.nn.Conv1d(
            CONFIG["NUM_CHANNELS_LAYER_12"],
            CONFIG["NUM_CHANNELS_LAYER_12"],
            kernel_size=ceiling_division(T, 32),
            groups=CONFIG["NUM_CHANNELS_LAYER_12"],
            padding=0
        )

        # Final linear layer to reduce the number of channels.
        self.x_coord_readout = torch.nn.Linear(
            CONFIG["NUM_CHANNELS_LAYER_12"],
            CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        self.y_coord_readout = torch.nn.Linear(
            CONFIG["NUM_CHANNELS_LAYER_12"],
            CONFIG["NUM_SLEAP_KEYPOINTS"]
        )


    def forward(self, x):

        h1 = self.conv_layers(x)
        h2 = torch.squeeze(self.final_pooling(h1), dim=-1)
        px = self.x_coord_readout(h2)
        py = self.y_coord_readout(h2)
        return torch.stack((px, py), dim=-1)


class GerbilizerDenseNet(torch.nn.Module):

    def __init__(self, CONFIG):
        super(GerbilizerDenseNet, self).__init__()

        if CONFIG["POOLING"] == "AVG":
            self.pooling = torch.nn.AvgPool1d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
        elif CONFIG["POOLING"] == "MAX":
            self.pooling = torch.nn.MaxPool1d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )
        else:
            raise ValueError("Did not recognize POOLING config.")

        # Initial number of audio channels.
        N = CONFIG["NUM_MICROPHONES"]

        self.f_convs = torch.nn.ModuleList([])
        self.g_convs = torch.nn.ModuleList([])
        self.norm_layers = torch.nn.ModuleList([])

        for i in range(12):
            n = CONFIG[f"NUM_CHANNELS_LAYER_{i + 1}"]
            fs = CONFIG[f"FILTER_SIZE_LAYER_{i + 1}"]
            d = CONFIG[f"DILATION_LAYER_{i + 1}"]
            self.f_convs.append(torch.nn.Conv1d(
                N, n, fs, stride=2, padding=((fs * d - 1) // 2), dilation=d
            ))
            self.g_convs.append(torch.nn.Conv1d(
                N, n, fs, stride=2, padding=((fs * d - 1) // 2), dilation=d
            ))
            if CONFIG["USE_BATCH_NORM"]:
                self.norm_layers.append(torch.nn.BatchNorm1d(N + n))
            else:
                self.norm_layers.append(torch.nn.Identity())
            N = N + n

        # Final pooling layer, which takes a weighted average
        # over the time axis.
        self.final_pooling = torch.nn.Conv1d(
            N, N, kernel_size=10, groups=N, padding=0
        )

        # Final linear layer to reduce the number of channels.
        self.x_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        self.y_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )

        # Initialize weights
        self.f_convs[0].weight.data.mul_(CONFIG["INPUT_SCALE_FACTOR"])
        self.g_convs[0].weight.data.mul_(CONFIG["INPUT_SCALE_FACTOR"])
        self.x_coord_readout.weight.data.mul_(CONFIG["OUTPUT_SCALE_FACTOR"])
        self.y_coord_readout.weight.data.mul_(CONFIG["OUTPUT_SCALE_FACTOR"])

    def forward(self, x):

        for fc, gc, bnrm in zip(
                self.f_convs, self.g_convs, self.norm_layers
            ):
            h = torch.tanh(fc(x)) * torch.sigmoid(gc(x))
            xp = self.pooling(x)
            x = bnrm(torch.cat((xp, h), dim=1))

        x_final = torch.squeeze(self.final_pooling(x), dim=-1)
        px = self.x_coord_readout(x_final)
        py = self.y_coord_readout(x_final)
        return torch.stack((px, py), dim=-1)


class GerbilizerHourglassNet(nn.Module):
    def __init__(self, config):
        super(GerbilizerHourglassNet, self).__init__()

        self.nonlinearity = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, 2) if config['USE_MAX_POOLING'] else nn.Identity()

        # Create a set of Conv1d layers to reduce input audio to a vector
        n_mics = config['NUM_MICROPHONES']
        n_conv_layers = config['NUM_CONV_LAYERS']
        n_channels = [config[f'NUM_CHANNELS_LAYER_{n + 1}'] for n in range(n_conv_layers)]
        n_channels.insert(0, n_mics)
        # Each entry is a tuple of the form (in_channels, out_channels)
        in_out_channels = zip(n_channels[:-1], n_channels[1:])
        dilations = [config[f'DILATION_LAYER_{n + 1}'] for n in range(n_conv_layers)]
        strides = [config[f'STRIDE_LAYER_{n + 1}'] for n in range(n_conv_layers)]
        kernel_sizes = [config[f'FILTER_SIZE_LAYER_{n + 1}'] for n in range(n_conv_layers)]

        self.conv_layers = nn.ModuleList()
        for in_out, filter_size, stride, dilation in zip(in_out_channels, kernel_sizes, strides, dilations):
            in_channels, out_channels = in_out
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=filter_size,
                    stride=stride,
                    dilation=dilation
                )
            )

        self.b_norms = nn.ModuleList()
        for n_features in n_channels[:-1]:
            self.b_norms.append(
                nn.BatchNorm1d(n_features) if config['USE_BATCH_NORM']
                else nn.Identity()
            )
        
        n_fc_layers = config['NUM_FC_LAYERS']
        fc_hidden_sizes = [config[f'HIDDEN_SIZE_FC_{n+1}'] for n in range(n_fc_layers)]
        fc_hidden_sizes.insert(0, out_channels)

        self.fc_layers = nn.ModuleList()
        for in_channels, out_channels in zip(fc_hidden_sizes[:-1], fc_hidden_sizes[1:]):
            self.fc_layers.append(
                nn.Linear(in_channels, out_channels)
            )

        # Reshape the intermediate vector into an image
        self.resize_channels = config['RESIZE_TO_N_CHANNELS']
        self.resize_width = int(np.sqrt(fc_hidden_sizes[-1] // self.resize_channels))
        self.resize_height = self.resize_width


        # Create a set of TransposeConv2d layers to upsample the reshaped vector
        self.tc_layers = nn.ModuleList()
        n_tc_layers = config['NUM_TCONV_LAYERS']
        n_tc_channels = [config[f'TCONV_CHANNELS_LAYER_{n + 1}'] for n in range(n_tc_layers)]
        n_tc_channels.insert(0, self.resize_channels)
        # Each entry is a tuple of the form (in_channels, out_channels)
        in_out_channels = zip(n_tc_channels[:-1], n_tc_channels[1:])
        tc_dilations = [config[f'TCONV_DILATION_LAYER_{n + 1}'] for n in range(n_tc_layers)]
        tc_strides = [config[f'TCONV_STRIDE_LAYER_{n + 1}'] for n in range(n_tc_layers)]
        tc_kernel_sizes = [config[f'TCONV_FILTER_SIZE_LAYER_{n + 1}'] for n in range(n_tc_layers)]

        for in_out, k_size, dilation, stride in zip(in_out_channels, tc_kernel_sizes, tc_dilations, tc_strides):
            in_channels, out_channels = in_out
            self.tc_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k_size,
                    dilation=dilation,
                    stride=stride
                )
            )

        self.tc_b_norms = nn.ModuleList()
        for n_features in n_tc_channels[:-1]:
            self.tc_b_norms.append(
                nn.BatchNorm2d(n_features) if config['USE_BATCH_NORM']
                else nn.Identity()
            )


    def forward(self, x):
        conv_output = x
        for c_layer, b_norm in zip(self.conv_layers, self.b_norms):
            conv_output = b_norm(conv_output)
            conv_output = c_layer(conv_output)
            conv_output = self.nonlinearity(conv_output)
            conv_output = self.maxpool(conv_output)

        avg = F.adaptive_avg_pool1d(conv_output, 1)
        avg = torch.squeeze(avg, -1)
        for lin_layer in self.fc_layers:
            avg = lin_layer(avg)

        avg = avg.reshape((-1, self.resize_channels, self.resize_height, self.resize_width))
        tc_output = avg
        for tc_layer, b_norm in zip(self.tc_layers, self.tc_b_norms):
            tc_output = b_norm(tc_output)
            tc_output = tc_layer(tc_output)
            tc_output = self.nonlinearity(tc_output)

        squeezed = torch.squeeze(tc_output)
        # Convert to probability distribution
        return F.softmax(squeezed, dim=0)
