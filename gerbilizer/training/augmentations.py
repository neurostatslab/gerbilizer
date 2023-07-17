import torch
from torch import nn
from torch_audiomentations import AddColoredNoise, PolarityInversion


class Identity(nn.Module):
    """Implements the identity function as a placeholder augmentation"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class TimeMask(nn.Module):
    """Time masking data augmentation. Masks a random part of the audio across all channels.
    Mutates the input tensor in-place. A bit slow because it has to do a lot of indexing.
    """

    def __init__(
        self, min_mask_length: int = 0, max_mask_length: int = 0, prob: float = 0.5
    ):
        super().__init__()
        self.min_mask_length = min_mask_length
        self.max_mask_length = max_mask_length
        self.prob = prob

    def forward(self, x):
        if x.dim() <= 2:
            # unbatched input
            if torch.rand() > self.prob:
                return x
            num_samples, num_channels = x.shape
            mask_start = torch.randint(0, num_samples - self.max_mask_length, (1,))
            mask_end = mask_start + torch.randint(
                self.min_mask_length, self.max_mask_length, (1,)
            )
            x[mask_start:mask_end, :] = 0
            return x
        # Batched input
        bshape = x.shape[:-2]
        num_samples, num_channels = x.shape[-2:]
        x = x.reshape(-1, num_samples, num_channels)
        bsz = x.shape[0]

        prob_mask = torch.rand(bsz) < self.prob  # True where we should apply the mask
        prob_idx = torch.nonzero(prob_mask).squeeze(1)
        num_true = prob_idx.shape[0]
        mask_start = torch.randint(0, num_samples - self.max_mask_length, (num_true,))
        mask_lengths = torch.randint(
            self.min_mask_length, self.max_mask_length, (num_true,)
        )
        mask_end = mask_start + mask_lengths
        for i, start, end in zip(prob_idx, mask_start, mask_end):
            x[i, start:end, :] = 0
        return x.reshape(*bshape, num_samples, num_channels)


def build_augmentations(CONFIG: dict) -> nn.Module:
    """Builds an augmentation module using the config parameters
    The augmentation module is a subclass of nn.Module which takes in a batched data tensor
    and returns a batched data tensor with the augmentations applied.

    Args:
        CONFIG (dict): Whole config dictionary

    Returns:
        nn.Module: Augmemtation module
    """
    augmentations = []
    aug_config = CONFIG["AUGMENTATIONS"]
    sample_rate = CONFIG["DATA"]["SAMPLE_RATE"]

    if not CONFIG["AUGMENTATIONS"]["AUGMENT_DATA"]:
        return Identity()

    if inversion_config := aug_config.get("INVERSION", False):
        # Inverts the polarity of the audio
        inversion = PolarityInversion(
            p=inversion_config.get("PROB", 0.5),
            mode="per_example",
            sample_rate=sample_rate,
        )
        augmentations.append(inversion)

    if noise_config := aug_config.get("NOISE", False):
        # Adds white background noise to the audio
        noise = AddColoredNoise(
            min_snr_in_db=noise_config.get("MIN_SNR", 0),
            max_snr_in_db=noise_config.get("MAX_SNR", 10),
            p=noise_config.get("PROB", 0.5),
            min_f_decay=0,
            max_f_decay=2,
            mode="per_example",
            sample_rate=sample_rate,
        )
        augmentations.append(noise)

    if mask_config := aug_config.get("MASK", False):
        # Masks a random part of the audio
        mask = TimeMask(
            min_mask_length=mask_config.get("MIN_LENGTH", 75),
            max_mask_length=mask_config.get("MAX_LENGTH", 250),
            prob=mask_config.get("PROB", 0.5),
        )
        augmentations.append(mask)

    return nn.Sequential(*augmentations)
