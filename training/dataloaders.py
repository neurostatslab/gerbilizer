"""
Functions to construct torch Dataset, DataLoader
objects and specify data augmentation.
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self, datapath, *,
        flip_vert=False, flip_horiz=False
    ):
        """
        Args:
            datapath (str):
              Path to directory containing the 'snippet{idx}' subdirectories
        """
        self.datapath = datapath
        self.dataset = h5py.File(datapath, 'r')
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz

    def __del__(self):
        self.dataset.close()

    def __len__(self):
        return self.dataset['vocalizations'].shape[0]

    def __getitem__(self, idx):
        
        # Load audio waveforms in time domain. Each sample is held
        # in a matrix with dimensions (10, num_audio_samples)
        #
        # The four microphones are arranged like this:
        #
        #           1-------------0
        #           |             |
        #           |             |
        #           |             |
        #           |             |
        #           2-------------3
        #
        # The 
        # 0 - mic 0 trace
        # 1 - mic 1 trace
        # 2 - mic 2 trace
        # 3 - mic 3 trace
        # 4 - (0, 1) - cross-correlation of mic 0 and mic 1
        # 5 - (0, 2) - cross-correlation of mic 0 and mic 2
        # 6 - (0, 3) - cross-correlation of mic 0 and mic 3
        # 7 - (1, 2) - cross-correlation of mic 1 and mic 2
        # 8 - (1, 3) - cross-correlation of mic 1 and mic 3
        # 9 - (2, 3) - cross-correlation of mic 2 and mic 3
        #
        sound = self.dataset['vocalizations'][idx][:]
        midpoint = len(sound) // 2
        snip_len = 125 * 45  # 45ms
        # sound = sound[midpoint - snip_len // 2 : midpoint + snip_len // 2, ...]
        # TODO: Fold cross correlations into hourglass model

        # Load animal location in the environment.
        #
        # shape: (num_keypoints, 2 (x/y coordinates))
        location_map = self.dataset['locations'][idx][:]

        # With p = 0.5, flip vertically
        if self.flip_vert and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            location_map = location_map[::-1, :]
            # mic 0 -> mic 3
            # mic 1 -> mic 2
            # mic 2 -> mic 1
            # mic 3 -> mic 0
            # (0, 1) -> (3, 2)  so  4 -> 9
            # (0, 2) -> (3, 1)  so  5 -> 8
            # (0, 3) -> (3, 0)  so  6 -> 6
            # (1, 2) -> (2, 1)  so  7 -> 7
            # (1, 3) -> (2, 0)  so  8 -> 5
            # (2, 3) -> (1, 0)  so  9 -> 4
            # sound = sound[[3, 2, 1, 0, 9, 8, 6, 7, 5, 4]]
            sound = sound[[3, 2, 1, 0]]

        # With p = 0.5, flip horizontally
        if self.flip_horiz and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            location_map = location_map[:, ::-1]
            # mic 0 -> mic 1
            # mic 1 -> mic 0
            # mic 2 -> mic 3
            # mic 3 -> mic 2
            # (0, 1) -> (1, 0)  so  4 -> 4
            # (0, 2) -> (1, 3)  so  5 -> 8
            # (0, 3) -> (1, 2)  so  6 -> 7
            # (1, 2) -> (0, 3)  so  7 -> 6
            # (1, 3) -> (0, 2)  so  8 -> 5
            # (2, 3) -> (3, 2)  so  9 -> 9
            # sound = sound[[1, 0, 3, 2, 4, 8, 7, 6, 5, 9]]
            sound = sound[[1, 0, 3, 2]]

        return sound.astype("float32"), location_map.astype("float32")


def build_dataloaders(path_to_data, CONFIG):

    # Construct Dataset objects.
    traindata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "train_set.h5"),
        flip_vert=(CONFIG["AUGMENT_DATA"] and CONFIG["AUGMENT_FLIP_VERT"]),
        flip_horiz=(CONFIG["AUGMENT_DATA"] and CONFIG["AUGMENT_FLIP_HORIZ"])
    )
    # TODO -- make new validation and test set files!
    valdata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "val_set.h5"),
        flip_vert=False, flip_horiz=False
    )
    testdata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "test_set.h5"),
        flip_vert=False, flip_horiz=False
    )

    # Construct DataLoader objects.
    train_dataloader = DataLoader(
        traindata,
        batch_size=CONFIG["TRAIN_BATCH_SIZE"],
        shuffle=True
    )
    val_dataloader = DataLoader(
        valdata,
        batch_size=CONFIG["VAL_BATCH_SIZE"],
        shuffle=True
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=CONFIG["TEST_BATCH_SIZE"],
        shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader
