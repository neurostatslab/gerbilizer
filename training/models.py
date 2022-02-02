import torch

from convlstm import ConvLSTM


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
    elif CONFIG["ARCHITECTURE"] == "GerbilizerReLUDenseNet":
        model = GerbilizerDenseNet(CONFIG)
    elif CONFIG["ARCHITECTURE"] == "GerbilizerRNNConv":
        model = GerbilizerRNNConv(CONFIG)
    
    if CONFIG["DEVICE"] == "GPU":
        model = model.cuda()
    
    def loss_function(x, y):
        return torch.mean(torch.square(x - y), axis=-1)

    return model, loss_function


class GerbilizerDenseNet(torch.nn.Module):

    def __init__(self, CONFIG):
        super(GerbilizerDenseNet, self).__init__()

        self.pooling = torch.nn.AvgPool1d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

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

        # Final linear layer to reduce the number of channels.
        self.x_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        self.y_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )

    def forward(self, x):

        for fc, gc, bnrm in zip(
                self.f_convs, self.g_convs, self.norm_layers
            ):
            h = torch.tanh(fc(x)) * torch.sigmoid(gc(x))
            xp = self.pooling(x)
            x = bnrm(torch.cat((xp, h), dim=1))

        px = self.x_coord_readout(x.swapaxes(1, 2))
        py = self.y_coord_readout(x.swapaxes(1, 2))
        return torch.stack((px, py), dim=-1)


class GerbilizerReLUDenseNet(torch.nn.Module):

    def __init__(self, CONFIG):
        super(GerbilizerDenseNet, self).__init__()

        self.pooling = torch.nn.AvgPool1d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )

        # Initial number of audio channels.
        N = CONFIG["NUM_MICROPHONES"]

        self.convs = torch.nn.ModuleList([])
        self.norm_layers = torch.nn.ModuleList([])

        for i in range(12):
            n = CONFIG[f"NUM_CHANNELS_LAYER_{i + 1}"]
            fs = CONFIG[f"FILTER_SIZE_LAYER_{i + 1}"]
            d = CONFIG[f"DILATION_LAYER_{i + 1}"]
            self.convs.append(torch.nn.Conv1d(
                N, n, fs, stride=2, padding=((fs * d - 1) // 2), dilation=d
            ))
            if CONFIG["USE_BATCH_NORM"]:
                self.norm_layers.append(torch.nn.BatchNorm1d(N + n))
            else:
                self.norm_layers.append(torch.nn.Identity())
            N = N + n

        # Final linear layer to reduce the number of channels.
        self.x_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )
        self.y_coord_readout = torch.nn.Linear(
            N, CONFIG["NUM_SLEAP_KEYPOINTS"]
        )

    def forward(self, x):

        for conv, bnrm in zip(self.convs, self.norm_layers):
            h = torch.relu(conv(x))
            xp = self.pooling(x)
            x = bnrm(torch.cat((xp, h), dim=1))

        px = self.x_coord_readout(x.swapaxes(1, 2))
        py = self.y_coord_readout(x.swapaxes(1, 2))
        return torch.stack((px, py), dim=-1)


class GerbilizerRNNConv(torch.nn.Module):
    def __init__(self, CONFIG):
        super(GerbilizerRNNConv, self).__init__()

        # Currently, only capable of predicting one point
        if CONFIG['NUM_SLEAP_KEYPOINTS'] != 1:
            raise NotImplementedError('GerbilizerRNNConv does not support inference of multiple keypoints')
        
        n_mics = CONFIG["NUM_MICROPHONES"]
        in_channels = n_mics

        n_layers = CONFIG['RNN_DEPTH']
        hidden_size = CONFIG['RNN_HIDDEN_SIZE']
        
        kernel_width = CONFIG['RNN_CONV_KERNEL_WIDTH']
        kernel_height = CONFIG['RNN_CONV_KERNEL_HEIGHT']
        kernel_shape = (kernel_width, kernel_height)

        self.conv_layer = ConvLSTM(
            n_mics,
            hidden_size,
            kernel_shape,
            n_layers,
            batch_first=True
        )

        # 1x1 conv to act as a fc layer among hidden state dimension
        self.nin_layer = torch.nn.Conv2d(
            hidden_size, 
            1,
            (1, 1)
        )

        # TODO: add config entries for fc layer sizes and count
        self.ff_layers = torch.nn.ModuleList()
        self.ff_layers.append(torch.nn.Linear(129 * 2, 256))
        self.ff_layers.append(torch.nn.Linear(256, 2))

    def forward(self, x, lens):
        """ Split the pack-padded batch into n samples, which are
        computed individually and concatenated to the final output.
        I think a better way to accomplish this would be to rewrite
        the ConvLSTM class to support this functionality, but I don't
        think that's worth doing before determining whether this
        architecture works for us.
        """
        predictions = list()
        for n, sample in enumerate(x):
            # Remove padding and add fake batch dimension
            fake_batch = torch.unsqueeze(sample[:lens[n], ...], 0)
            # The second tuple element contains the hidden and cell states
            conv_out, _ = self.conv_layer(fake_batch)
            # I don't know why this returns a list
            conv_out = conv_out[-1]
            # Grab only the last iteration's hidden state
            conv_out = conv_out[:, -1, ...]

            # Evaluate FC NiN over the hidden state's dimension
            linear_in = self.nin_layer(conv_out)

            # Flatten and pass through FC layers
            # Avoid flattening the batch dimension
            linear_out = torch.flatten(linear_in, start_dim=1)

            for fc_layer in self.ff_layers:
                linear_out = fc_layer(linear_out)
            
            predictions.append(linear_out)
        predictions = torch.cat(predictions, dim=0)
        return predictions