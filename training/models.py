from unicodedata import bidirectional
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
    """ This version is based on the model introduced in the paper
    "Sound Event Localization and Detection of Overlapping Sources
    Using Convolutional Recurrent Neural Networks", Adavanne et al.
    """
    def __init__(self, CONFIG):
        super(GerbilizerRNNConv, self).__init__()
        self.config = CONFIG

        # Currently, only capable of predicting one point
        if CONFIG['NUM_SLEAP_KEYPOINTS'] != 1:
            raise NotImplementedError('GerbilizerRNNConv does not support inference of multiple keypoints')
        
        n_mics = CONFIG["NUM_MICROPHONES"]
        in_channels = n_mics
        
        n_conv_blocks = CONFIG['NUM_CONV_BLOCKS']
        self.conv_blocks = torch.nn.ModuleList()

        for n in range(n_conv_blocks):
            in_channels = CONFIG[f'CONV_BLOCK_{n+1}_IN_CHANNELS']
            out_channels = CONFIG[f'CONV_BLOCK_{n+1}_OUT_CHANNELS']
            kernel_size = CONFIG[f'CONV_BLOCK_{n+1}_KERNEL_SIZE']
            mp_size = CONFIG[f'CONV_BLOCK_{n+1}_MAXPOOL_SIZE']
            
            conv_block = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding='same'
                ),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.MaxPool2d(
                    kernel_size=(mp_size, 1),
                    stride=(mp_size, 1)
                )
            )

            self.conv_blocks.append(conv_block)
        
        rec_input_size = CONFIG['RECURRENT_INPUT_SIZE']
        rec_layer_depth = CONFIG['RECURRENT_LAYER_DEPTH']
        rec_hidden_size = CONFIG['RECURRENT_HIDDEN_SIZE']
        rec_dropout = 0.5 if CONFIG['RECURRENT_USE_DROPOUT'] else 0
        rec_type = torch.nn.GRU if CONFIG['RECURRENT_LAYER_TYPE'] == 'GRU' else torch.nn.LSTM
        self.recurrent_layer = rec_type(
            input_size=rec_input_size,
            hidden_size=rec_hidden_size,
            num_layers=rec_layer_depth,
            dropout=rec_dropout,
            batch_first=True,
            bidirectional=True
        )

        ff_hidden_size = CONFIG['FC_HIDDEN_SIZE']
        ff_use_dropout = CONFIG['FC_USE_DROPOUT']
        self.ff_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * rec_hidden_size, ff_hidden_size),
            (torch.nn.Dropout(0.5) if ff_use_dropout else torch.nn.Identity()),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_hidden_size, 2)
        )
    
    def clip_gradients(self):
        if not self.config['SHOULD_CLIP_RECURRENT_GRAD']:
            return
        max_norm = self.config['RECURRENT_MAX_GRAD_NORM']
        torch.nn.utils.clip_grad_norm_(
            self.recurrent_layer.parameters(),
            max_norm=max_norm,
            error_if_nonfinite=True  # No point in continuing if the grad is inf
        )

    def forward(self, packed_x):
        """ Since the Conv2d class doesn't support packed sequences,
        the sequences are unpacked before evaluation there, and repacked
        again before enterring the recurrent layers.
        This will be more effecient as the depth of the recurrent module
        increases.
        """
        unpacked_x, lens = pad_packed_sequence(packed_x, batch_first=True)
        conv_outputs = list()

        for n, samp in enumerate(unpacked_x):
            conv_out = samp[:lens[n]]
            # Unsqueeze to create fake batch dim
            conv_out = torch.unsqueeze(conv_out, 0)
            # Move the mag/phase axis to the front (to act as a channel dim in the conv. layer)
            conv_out = torch.transpose(conv_out, 1, 3)
            
            for c_block in self.conv_blocks:
                conv_out = c_block(conv_out)
            
            # Move the time dim (variable length) to the front
            conv_out = torch.transpose(conv_out, 0, 3)
            # Flatten to create a shape that the LSTM module will accept
            conv_out = torch.flatten(conv_out, start_dim=1)
            # The shape should now be (L, n_conv_filters * 2) depending on the sizes of the maxpool filters
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