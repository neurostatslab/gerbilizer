from math import comb

import torch
from torch import nn

from .encodings import LearnedEncoding, FixedEncoding
from .sparse_attn import SparseTransformerEncoder, SparseTransformerEncoderLayer


class GerbilizerReducedAttentionNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        n_mics = config['NUM_MICROPHONES']
        if config['COMPUTE_XCORRS']:
            n_mics += comb(n_mics, 2)
        
        reduction = config['REDUCTION']
        self.reduction = reduction
        d_model = config[f'TRANSFORMER_D_MODEL']
        n_attn_heads = config['N_ATTENTION_HEADS']
        n_transformer_layers = config['N_TRANSFORMER_LAYERS']
        n_global = config['TRANSFORMER_GLOBAL_BLOCKS']
        n_window = config['TRANSFORMER_WINDOW_BLOCKS']
        n_random = config['TRANSFORMER_RANDOM_BLOCKS']
        block_size = config['TRANSFORMER_BLOCK_SIZE']

        self.data_encoding = nn.Linear(n_mics * reduction, d_model)

        encoder_layer = SparseTransformerEncoderLayer(
            d_model,
            n_attn_heads,
            block_size=block_size,
            n_global=n_global,
            n_window=n_window,
            n_random=n_random,
            dim_feedforward=2048,
            dropout=0.1
        )

        self.transformer = SparseTransformerEncoder(encoder_layer, n_transformer_layers)
        
        # No dropout here since the TransformerEncoder implements its own dropout
        # Add 1 to account for the additional cls token appended in forward()
        # self.p_encoding = FixedEncoding(n_features, max_len=config['SAMPLE_LEN'] + 1)
        self.p_encoding = LearnedEncoding(d_model=d_model, max_seq_len=config['SAMPLE_LEN'] // reduction + 1)

        self.linear_dim = 1024

        self.dense = nn.Sequential(
            nn.Linear(d_model, self.linear_dim),
            nn.ReLU()
        )

        self.coord_readout = nn.Sequential(nn.Linear(self.linear_dim, 2), nn.Tanh())
    
    def _clip_grads(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
    
    def embed(self, x):
        """ I'm splitting the forward pass into this 'embedding' segment
        and the final x-y coord output to make inheritance easier for the
        hourglass subclass.
        """
        # X initial shape: (batch, channels, seq)
        x = x.view(x.shape[0], x.shape[1] * self.reduction, -1).transpose(-1, -2)
        reduced_x = self.data_encoding(x)  # Should have shape (batch, seq // reduction, d_model)
        cls_token = torch.zeros((reduced_x.shape[0], 1, reduced_x.shape[-1]), device=reduced_x.device)
        transformer_input = torch.cat([cls_token, reduced_x], dim=1)
        encoded = self.p_encoding(transformer_input)
        transformer_out = self.transformer(encoded)[:, 0, :]
        return self.dense(transformer_out)

    def forward(self, x, *, return_embedding=False):
        linear_out = self.embed(x)
        if return_embedding:
            return self.coord_readout(linear_out), linear_out.detach().clone()
        return self.coord_readout(linear_out)
    
    def trainable_params(self):
        return self.parameters()
