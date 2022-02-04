import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


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
        
        # TODO: Add configurable parameters
        conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=1,
                out_channels=32,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=32),
            torch.nn.MaxPool3d(
                kernel_size=(1, 1, 4),
                stride=(1, 1, 4)
            )
        )

        conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=32,
                out_channels=16,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=16),
            torch.nn.MaxPool3d(
                kernel_size=(1, 1, 4),
                stride=(1, 1, 4)
            )
        )

        conv_layer_3 = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=16,
                out_channels=16,
                kernel_size=3
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm3d(num_features=16),
            torch.nn.MaxPool3d(
                kernel_size=(1, 1, 2),
                stride=(1, 1, 2)
            )
        )

        self.conv_layers = torch.nn.ModuleList([conv_layer_1, conv_layer_2, conv_layer_3])

        self.recurrent_layer = torch.nn.LSTM(
            input_size=192,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        self.ff_layer = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )

    def forward(self, packed_x):
        """ Since the Conv2d class doesn't support packed sequences,
        the sequences are unpacked before evaluation there, and repacked
        again before enterring the recurrent layers.
        This will be more effecient as the depth of the recurrent module
        increases.
        """
        conv_outputs = list()
        unpacked_x, lens = pad_packed_sequence(packed_x, batch_first=True)

        conv_outputs = list()
        for n, samp in enumerate(unpacked_x):
            conv_out = samp[:lens[n]]
            # Unsqueeze twice to create fake batch dim and channel dim
            conv_out = torch.unsqueeze(conv_out, 0)
            conv_out = torch.unsqueeze(conv_out, 0)
            
            for c_layer in self.conv_layers:
                conv_out = c_layer(conv_out)
            # Move the time dim to the front
            conv_out = torch.transpose(conv_out, 0, 2)
            # Flatten to create a shape that the LSTM module will accept
            conv_out = torch.flatten(conv_out, start_dim=1)
            conv_outputs.append(conv_out)
        
        # Repack for LSTM efficiency
        pad_conv_outputs = pad_sequence(conv_outputs, batch_first=True)
        conv_output_lens = [tensor.shape[0] for tensor in conv_outputs]
        packed_conv_outputs = pack_padded_sequence(
            pad_conv_outputs, 
            conv_output_lens, 
            batch_first=True, 
            enforce_sorted=False
        )

        rec_out, _ = self.recurrent_layer(packed_conv_outputs)
        # Unpack again and select the final hidden state for each input in the minibatch
        unpacked_data, unpacked_lens = pad_packed_sequence(rec_out, batch_first=True)
        last_hidden_states = unpacked_data[
            torch.arange(unpacked_data.shape[0]),  # The length - 1 element is only relevant to the corresponding input's index
            unpacked_lens - 1,  # Length - 1 to get the last element of the RNN chain
            :  # every element of the hidden state
        ]

        # x-y coords of final prediction
        lin_out = self.ff_layer(last_hidden_states)
        return lin_out