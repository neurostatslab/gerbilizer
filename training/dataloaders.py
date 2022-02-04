"""
Functions to construct torch Dataset, DataLoader
objects and specify data augmentation.
"""

import os
import h5py
import joblib
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self, datapath, *,
        flip_vert=False, flip_horiz=False,
        audio_pad_len=125000
    ):
        """
        Args:
            datapath (str):
              Path to directory containing the 'snippet{idx}' subdirectories
            audio_pad_len (int):
              Standardized length of all audio samples. Shorter vocalizations
              will be zero-pad to this length. Longer vocalizations will be
              truncated equally at both ends to match this length
        """
        self.datapath = datapath
        self.dataset = h5py.File(datapath, 'r')
        self.audio_pad_len = audio_pad_len  # Standardized audio sample length, int
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz

    def __del__(self):
        self.dataset.close()

    def __len__(self):
        return self.dataset['vocalizations'].shape[0]

    def __getitem__(self, idx):
        
        # Load audio waveforms in time domain. Each sample is held
        # in a matrix with dimensions (4, num_audio_samples)
        #
        # The four microphones are arranged like this:
        #
        #           1-------------0
        #           |             |
        #           |             |
        #           |             |
        #           |             |
        #           2-------------3
        sound = self.dataset['vocalizations'][idx][:]

        # Load animal location in the environment.
        #
        # shape: (num_keypoints, 2 (x/y coordinates), num_video_frames)
        locations = self.dataset['locations'][idx]

        # With p = 0.5, flip vertically
        if self.flip_vert and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            locations[:, 1] *= -1
            sound = sound[[3, 2, 1, 0]]

        # With p = 0.5, flip horizontally
        if self.flip_horiz and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            locations[:, 0] *= -1
            sound = sound[[1, 0, 3, 2]]

        return sound, locations

class VarLenDataset(Dataset):
    def __init__(
        self, datapath, output_scaler_path, *,
        flip_vert=False, flip_horiz=False
    ):
        """
        Args:
            datapath (str):
              Path to directory containing the 'snippet{idx}' subdirectories
            audio_pad_len (int):
              Standardized length of all audio samples. Shorter vocalizations
              will be zero-pad to this length. Longer vocalizations will be
              truncated equally at both ends to match this length
        """
        self.datapath = datapath
        self.dataset = h5py.File(datapath, 'r')
        self.output_scaler = joblib.load(output_scaler_path)
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz

    def __del__(self):
        self.dataset.close()

    def __len__(self):
        # The last listed index is the ending idx of the last vocalization
        # and doesn't represent the start of a new audio snippet
        return len(self.dataset['vocalization_indices']) - 1

    def __getitem__(self, idx):
        # Load audio waveforms in time domain. Each sample is held
        # in a matrix with dimensions (4, num_audio_samples)
        #
        # The four microphones are arranged like this:
        #
        #           1-------------0
        #           |             |
        #           |             |
        #           |             |
        #           |             |
        #           2-------------3
        idx_start, idx_end = self.dataset['vocalization_indices'][idx:idx+2]
        stft_sound = self.dataset['vocalizations'][idx_start:idx_end, ...]

        # Load animal location in the environment.
        #
        # shape: (num_keypoints, 2 (x/y coordinates), num_video_frames)
        locations = self.dataset['locations'][idx]
        locations = self.output_scaler.inverse_transform(locations.reshape((-1, 2))).flatten()

        # With p = 0.5, flip vertically
        if self.flip_vert and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            locations[1] *= -1
            stft_sound = stft_sound[..., [3, 2, 1, 0, 7, 6, 5, 4]]

        # With p = 0.5, flip horizontally
        if self.flip_horiz and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            locations[0] *= -1
            stft_sound = stft_sound[..., [1, 0, 3, 2, 5, 4, 7, 6]]

        return stft_sound, self.output_scaler.transform(locations.reshape((-1, 2))).flatten()
    
    @classmethod
    def collate_batch(cls, batch):
        sounds, locations = zip(*batch)
        lens = [len(sound) for sound in sounds]
        sound_tensors = [torch.from_numpy(sound) for sound in sounds]

        padded_seq = pad_sequence(sound_tensors, batch_first=True)
        packed_seq = pack_padded_sequence(
            padded_seq,
            lens,
            batch_first=True,
            enforce_sorted=False
        )
        locations = torch.from_numpy(np.stack(locations, axis=0))
        return packed_seq, locations


def build_dataloaders(path_to_data, CONFIG):

    # Construct Dataset objects.
    if CONFIG['ARCHITECTURE'] == 'GerbilizerRNNConv':
        traindata = VarLenDataset(
            os.path.join(path_to_data, "train_set.h5"),
            output_scaler_path='data/2022-02-03-12-58-33_locationscaler.joblib',
            flip_vert=(CONFIG["AUGMENT_DATA"] and CONFIG["AUGMENT_FLIP_VERT"]),
            flip_horiz=(CONFIG["AUGMENT_DATA"] and CONFIG["AUGMENT_FLIP_HORIZ"])
        )
        # TODO -- make new validation and test set files!
        valdata = VarLenDataset(
            os.path.join(path_to_data, "val_set.h5"),
            output_scaler_path='data/2022-02-03-12-58-33_locationscaler.joblib',
            flip_vert=False, flip_horiz=False
        )
        testdata = VarLenDataset(
            os.path.join(path_to_data, "test_set.h5"),
            output_scaler_path='data/2022-02-03-12-58-33_locationscaler.joblib',
            flip_vert=False, flip_horiz=False
        )
    else:
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

    collate_fn = VarLenDataset.collate_batch if CONFIG['ARCHITECTURE'] == 'GerbilizerRNNConv' else None

    # Construct DataLoader objects.
    train_dataloader = DataLoader(
        traindata,
        batch_size=CONFIG["TRAIN_BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        valdata,
        batch_size=CONFIG["VAL_BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=CONFIG["TEST_BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader, test_dataloader






