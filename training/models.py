import torch

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
