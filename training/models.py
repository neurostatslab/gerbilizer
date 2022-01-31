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
        
        # Create convolutional layers
        self.conv_layers = torch.nn.ModuleList()
        n_mics = CONFIG["NUM_MICROPHONES"]
        in_channels = n_mics

        n_layers = CONFIG['NUM_CONV_LAYERS']
        input_len = CONFIG['INPUT_AUDIO_LEN']

        for i in range(n_layers):
            out_channels = CONFIG[f'NUM_CHANNELS_LAYER_{i + 1}']
            kernel_size = CONFIG[f'FILTER_SIZE_LAYER_{i + 1}']
            dilation = CONFIG[f'DILATION_LAYER_{i + 1}']
            stride = CONFIG[f'STRIDE_LAYER_{i + 1}']
            conv_layer = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation
            )
            self.conv_layers.append(conv_layer)
            in_channels = out_channels
        
        if CONFIG['USE_POOLING']:
            self.pooling_layer = torch.nn.MaxPool1d(
                kernel_size=2,
                stride=2
            )
        else:
            self.pooling_layer = torch.nn.Identity()

        # The number of 'features' seen by the RNN at each time step
        # is equal to the number of channels we end up with after all
        # convolutions
        # Note that axes 1 and 2 will need to be swapped, as the RNN module
        # expects input to have the shape (batch_size, seq_len, n_features)
        n_rnn_features = in_channels
        rnn_hidden_size = CONFIG['RNN_HIDDEN_SIZE']
        n_rnn_layers = CONFIG['RNN_DEPTH']
        is_bidirectional = CONFIG['RNN_IS_BIDIRECTIONAL']
        dropout = CONFIG['RNN_DROPOUT_PROB']
    
        if CONFIG['RECURRENT_CELL_TYPE'] == 'RNN':
            cell_type = torch.nn.RNN
        elif CONFIG['RECURRENT_CELL_TYPE'] == 'GRU':
            cell_type = torch.nn.GRU
        elif CONFIG['RECURRENT_CELL_TYPE'] == 'LSTM':
            cell_type = torch.nn.LSTM
        else:
            raise NotImplementedError(f'Unknown value {CONFIG["RECURRENT_CELL_TYPE"]} for recurrent cell type.')

        self.recurrent_layer = cell_type(
            input_size=n_rnn_features,
            hidden_size=rnn_hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=is_bidirectional
        )
        # Reduce the hidden state to a coordinate prediction
        self.feedforward = torch.nn.Linear(rnn_hidden_size * n_rnn_layers, 2)

    def forward(self, x):
        conv_result = x
        for conv_layer in self.conv_layers:
            conv_result = self.pooling_layer(conv_layer(conv_result))
        # Swaps the signal_len and num_features dims
        rnn_input = torch.transpose(conv_result, 1, 2)
        # [0] because it returns a tuple of (outputs, hidden states)
        rnn_output = self.recurrent_layer(rnn_input)[0]
        # We only care about the prediction produced at the end of the sequence
        final_output = rnn_output[:, -1, :]
        return self.feedforward(final_output)