from math import ceil
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..training.augmentations import build_audio_augmentations
from ..training.configs import update_recursively


class Transpose(nn.Module):
    """Convenience module for transposing a tensor within a sequential block"""

    def __init__(self, dim_a: int, dim_b: int):
        super().__init__()
        self.a = dim_a
        self.b = dim_b

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(self.a, self.b)


class Skip(nn.Module):
    """Convenience module for adding an input tensor to the output of a block of operations"""

    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class TokenizerSimpleLayer(torch.nn.Module):
    """A simple convolutional building block for tokenizing audio"""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        filter_size: int,
        stride: int,
    ):
        super(TokenizerSimpleLayer, self).__init__()

        self.fc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            stride=stride,
            groups=channels_in,
        )
        self.gc = torch.nn.Conv1d(
            channels_in,
            channels_out,
            filter_size,
            stride=stride,
            groups=channels_in,
        )
        self.norm = torch.nn.BatchNorm1d(channels_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fcx = self.fc(x)
        return self.norm(
            (0.95 * torch.tanh(fcx) + 0.05 * fcx) * torch.sigmoid(self.gc(x))
        )


class Learned1DEncoding(nn.Module):
    """Positional encoding in the time domain"""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        enc = torch.empty((max_seq_len, d_model))
        nn.init.uniform_(enc, -0.1, 0.1)
        self.encoding = nn.Parameter(enc, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[: x.shape[1], :].unsqueeze(0)


class MicrophoneIdentityEncoding(nn.Module):
    """Encoding of microphone identity"""

    def __init__(self, num_channels):
        pass

    def forward(self, x):
        pass


class Tokenizer(nn.Module):
    """Abstract class defining the tokenizer interface"""

    def __init__(self):
        super().__init__()
        self.is_teacher: bool
        self.positional_encoding: nn.Module
        self.augment: nn.Module
        self.teacher: Optional[nn.Module]

    def forward(self, x):
        raise NotImplementedError("This is an abstract class")

    def step_teacher(self):
        raise NotImplementedError("This is an abstract class")


class SpectrogramTokenizer(Tokenizer):
    pass


class ConvolutionalTokenizer(Tokenizer):
    # Mirrors the structure of the convolutional encoder in w2v 2.0
    default_config = {
        "TYPE": "CONVOLUTIONAL",
        "CHANNELS": [256, 768],
        "KERNEL_SIZES": [15, 15],
        "STRIDES": [5, 5],
        # Constants for the teacher time constant
        "TAU_INITIAL": 0.9997,
        "TAU_FINAL": 0.99999,
        "TAU_NUM_STEPS": 75_000,
    }

    def __init__(self, config: dict, *, create_teacher: bool = True):
        super().__init__()
        tokenizer_config = config["MODEL_PARAMS"].get("TOKENIZER", {})
        update_recursively(tokenizer_config, ConvolutionalTokenizer.default_config)
        config["MODEL_PARAMS"][
            "TOKENIZER"
        ] = tokenizer_config  # Allow changes to propagate to the config file saved to the model directory

        n_mics = config["DATA"]["NUM_MICROPHONES"]
        updated_stride = [s * n_mics for s in tokenizer_config["STRIDES"]]
        channels = [1] + tokenizer_config["CHANNELS"]

        self.convs = []
        for in_channels, out_chonnels, kernel_size, stride in zip(
            channels[:-1],
            channels[1:],
            tokenizer_config["KERNEL_SIZES"],
            updated_stride,
        ):
            self.convs.append(
                TokenizerSimpleLayer(in_channels, out_chonnels, kernel_size, stride)
            )
        self.convs = nn.Sequential(*self.convs)

        self.positional_encoding = Learned1DEncoding(
            channels[-1], config["DATA"]["CROP_LENGTH"]
        )

        self.teacher = None
        if create_teacher:
            self.create_teacher(config)
            self.is_teacher = False
            self.tau_initial = tokenizer_config["TAU_INITIAL"]
            self.tau_final = tokenizer_config["TAU_FINAL"]
            self.tau_num_steps = tokenizer_config["TAU_NUM_STEPS"]
            self.current_num_steps = 0
        else:
            self.requires_grad_(False)
            self.is_teacher = True

        self.augment = build_audio_augmentations(config)

    @property
    def tau(self) -> float:
        """Returns the current value of the teacher time constant
        It is a linear interpolation between tau_initial and tau_final for tau_num_steps steps and
        constant afterwards
        """
        if self.is_teacher:
            raise ValueError("Cannot call tau when teacher is self")
        if self.current_num_steps >= self.tau_num_steps:
            return self.tau_final
        return (
            self.tau_initial
            + (self.tau_final - self.tau_initial)
            * self.current_num_steps
            / self.tau_num_steps
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, n_mics = x.shape
        x = x.reshape(bsz, 1, -1)
        conv_out = self.convs(x).transpose(-1, -2)  # Returns (bsz, seq_len, d_model)
        return self.apply_positional_encoding(conv_out)

    def apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(x)  # Doesn't need any reshaping

    def create_teacher(self, config: dict) -> None:
        self.teacher = ConvolutionalTokenizer(config, create_teacher=False)
        # Copy initial weights over
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.parameters()
        ):
            teacher_param.data = student_param.data.detach().clone()

    def teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the teacher tokenizer"""
        if self.is_teacher:
            raise ValueError("Cannot call teacher_forward when teacher is self")
        return self.teacher.forward(x)

    def step_teacher(self) -> None:
        """Updates the teacher parameters with an exponential moving average of the student parameters"""
        if self.is_teacher:
            raise ValueError("Cannot step teacher when teacher is self")
        tau = self.tau
        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.parameters()
        ):
            teacher_param.data = (
                1 - tau
            ) * teacher_param.data + tau * student_param.data.detach().clone()
        self.current_num_steps += 1


class GerbilizerAttentionNet(nn.Module):
    default_config = {
        "MASK_PROB": 0.5,
        "MASK_TOKEN_LEARNED": True,
        "MASK_BLOCK_LENGTH": 5,  # In tokens, exact length depends on stride
        "MASKS_PER_SEQUENCE": 16,
        "ENCODER": {
            "D_MODEL": 768,
            "NUM_LAYERS": 6,
            "NUM_HEADS": 8,
            "D_FF": 2048,
        },
        "DECODER": {
            "NUM_LAYERS": 2,
            "NUM_HEADS": 8,
            "D_FF": 2048,
        },
        "TOKENIZER": {
            "TYPE": "CONVOLUTIONAL",
        },
    }

    def __init__(self, config: dict):
        super().__init__()
        model_config = config.get("MODEL_PARAMS", {})
        update_recursively(model_config, GerbilizerAttentionNet.default_config)
        config[
            "MODEL_PARAMS"
        ] = model_config  # Allow changes to propagate to the config file saved to the model directory

        encoder_config = model_config["ENCODER"]
        decoder_config = model_config["DECODER"]
        tokenizer_config = model_config["TOKENIZER"]

        d_model = encoder_config["D_MODEL"]
        self.mask_prob = model_config["MASK_PROB"]
        self.mask_block_length = model_config["MASK_BLOCK_LENGTH"]
        self.masks_per_sequence = model_config["MASKS_PER_SEQUENCE"]
        if model_config["MASK_TOKEN_LEARNED"]:
            self.mask_token = nn.Parameter(torch.empty(d_model), requires_grad=True)
            nn.init.uniform_(self.mask_token, -0.1, 0.1)
        else:
            self.mask_token = nn.Parameter(torch.zeros(d_model), requires_grad=False)

        # A little something for the IDE
        self.tokenizer: Tokenizer
        if tokenizer_config["TYPE"] == "CONVOLUTIONAL":
            self.tokenizer = ConvolutionalTokenizer(config)
        elif tokenizer_config["TYPE"] == "SPECTROGRAM":
            self.tokenizer = SpectrogramTokenizer(config)
        else:
            raise ValueError("Tokenizer type not recognized")

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=encoder_config["NUM_HEADS"],
                dim_feedforward=encoder_config["D_FF"],
                dropout=0.1,
            ),
            num_layers=encoder_config["NUM_LAYERS"],
        )

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=decoder_config["NUM_HEADS"],
                dim_feedforward=decoder_config["D_FF"],
                dropout=0.1,
            ),
            num_layers=decoder_config["NUM_LAYERS"],
        )

    @staticmethod
    def __block_mask(
        length: int, num_unmasked: int, block_length: int
    ) -> torch.BoolTensor:
        """Creates a mask with contiguous blocks of unmasked tokens
        Mask has value 1 for unmasked tokens and 0 for masked tokens
        Returns a mask of shape (length,)
        """
        mask = torch.zeros((length,), dtype=torch.bool)

        num_start_points = ceil(
            (num_unmasked + 0.05 * length) / block_length
        )  # Add some extra just in case overlaps result in undermasking
        start_points = torch.randint(0, length - block_length, (num_start_points,))
        for start_point in start_points:
            mask[start_point : start_point + block_length] = 1

        # Ensure the number of masked tokens is correct
        unchecked_num_unmasked = mask.sum()
        while unchecked_num_unmasked < num_unmasked:
            idx = torch.randint(0, length, (1,))
            if not mask[idx]:
                mask[idx] = 1
                unchecked_num_unmasked += 1
        while unchecked_num_unmasked > num_unmasked:
            idx = torch.randint(0, length, (1,))
            if mask[idx]:
                mask[idx] = 0
                unchecked_num_unmasked -= 1

        return mask

    @staticmethod
    def __shuffling_for_mask(mask: torch.BoolTensor) -> torch.IntTensor:
        """Creates a shuffling permutation for a given mask
        The permutation is such that the unmasked tokens are at the beginning of the sequence
        """
        return mask.int().argsort(dim=-1, descending=True)

    def mask_sequence(
        self, tokens: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Prepare masked tokens for training. Returns a tuple of the masked tokens and a mask.

        Params:
            tokens: a tensor of shape (batch_size, masks_per_sequence, seq_len, d_model) containing the token indices
        """
        bsz, num_tokens, d_model = tokens.shape

        num_masked = int(self.mask_prob * num_tokens)
        num_unmasked = num_tokens - num_masked

        num_masks = bsz * self.masks_per_sequence
        mask = (
            torch.stack(
                [
                    GerbilizerAttentionNet.__block_mask(
                        num_tokens, num_unmasked, self.mask_block_length
                    )
                    for _ in range(num_masks)
                ]
            )
            .reshape(bsz, self.masks_per_sequence, num_tokens)
            .to(tokens.device)
        )

        shuffling = GerbilizerAttentionNet.__shuffling_for_mask(mask)
        unshuffling = shuffling.argsort(dim=-1)

        expanded_tokens = tokens.unsqueeze(1).expand(
            -1, self.masks_per_sequence, -1, -1
        )
        unmasked_indices = (
            shuffling[..., :num_unmasked].unsqueeze(-1).expand(-1, -1, -1, d_model)
        )
        unmasked_tokens = torch.gather(expanded_tokens, -2, unmasked_indices)

        # Returns unmasked tokens of shape (batch_size, masks_per_sequence, num_unmasked, d_model) and unshuffling of shape (batch_size, masks_per_sequence, seq_len)
        # unshuffling tensor assumes unmasked tokens are at the beginning of the sequence
        return unmasked_tokens, unshuffling

    def construct_decoder_input(
        self, encoded_tokens: torch.FloatTensor, unshuffling: torch.IntTensor
    ) -> torch.FloatTensor:
        """Reconstructs the original sequence with masked tokens inserted"""
        # Encoded tokens shape: (bsz * masks_per_sequence, num_unmasked, d_model)
        # Unshuffling shape: (bsz, masks_per_sequence, seq_len)
        bsz, _, seq_len = unshuffling.shape
        d_model = encoded_tokens.shape[-1]
        num_unmasked = encoded_tokens.shape[-2]
        num_masked = unshuffling.shape[2] - num_unmasked

        expanded_mask_token = self.mask_token[None, None, :].expand(
            bsz * self.masks_per_sequence, num_masked, -1
        )
        shuffled_tensor = torch.cat(
            [encoded_tokens, expanded_mask_token], dim=-2
        )  # (bsz * masks_per_sequence, seq_len, d_model)

        # reshaping and expansion in preparation for gather
        unshuffling = unshuffling.reshape(
            bsz * self.masks_per_sequence, seq_len, 1
        ).expand(-1, -1, d_model)
        return shuffled_tensor.gather(
            -2, unshuffling
        )  # returns (bsz * masks_per_sequence, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Storting with the original data x, first augment it
        # This ensures both the teacher and student work with the same augmentation of the data
        x = self.tokenizer.augment(x)
        # Then tokenize it
        teacher_tokens = self.tokenizer.teacher_forward(x)
        student_tokens = self.tokenizer(x)  # (bsz, seq_len, d_model)
        # Tokenizer handles initial application of positional encoding

        # Then mask the tokens
        unmasked_tokens, unshuffling = self.mask_sequence(student_tokens)
        # Encode the masked tokens
        bsz, masks_per_sequence, num_unmasked, d_model = unmasked_tokens.shape
        # Flatten the input for convenience
        encoded_tokens = self.encoder(
            unmasked_tokens.reshape(-1, num_unmasked, d_model)
        )

        # Reconstruct the original sequence with mask tokens inserted
        # Since this operates on tokens, it behaves the same for each input modality and can be done
        # outside the tokenizer class
        decoder_input = self.construct_decoder_input(encoded_tokens, unshuffling)
        decoder_input = self.tokenizer.apply_positional_encoding(decoder_input)

        # Undo the earlier flattening to disambiguate the output ordering
        decode_output = self.decoder(decoder_input).reshape(
            bsz, masks_per_sequence, -1, d_model
        )
        return decode_output, teacher_tokens
