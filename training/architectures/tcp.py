from torch import nn


class TCP(nn.Module):
    def __init__(self, input_dim, n_layers, n_hidden, nonlinearity='relu', dropout=0.0):
        # just gonna make this an MLP
        super(TCP, self).__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = nonlinearity.lower()
            if nonlinearity == 'relu':
                nonlin = nn.ReLU()
            elif nonlinearity == 'gelu':
                nonlin = nn.GELU()
            else:
                raise ValueError(f'Unsupported nonlinearity: {nonlinearity}')
        else:
            nonlin = nonlinearity()
        
        mods = []
        mods.append(nn.Linear(input_dim, n_hidden))
        mods.append(nonlin)
        for _ in range(1, n_layers - 1):
            mods.append(nn.Dropout(dropout))
            mods.append(nn.Linear(n_hidden, n_hidden))
            mods.append(nonlin)
        mods.append(nn.Linear(n_hidden, 1))

        self.network = nn.Sequential(*mods)

    def forward(self, final_layer):
        return self.network(final_layer)
