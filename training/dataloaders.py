"""
Functions to construct torch Dataset, DataLoader
objects and specify data augmentation.
"""

from itertools import combinations
from math import comb
import os
from typing import Optional, Tuple

import h5py
import numpy as np
from scipy.signal import correlate
from torch import randint
from torch.utils.data import Dataset, DataLoader


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self,
        datapath, *,
        flip_vert: bool=False,
        flip_horiz: bool=False,
        segment_len: int=256,
        map_size: Optional[int]=None,
        arena_dims: Optional[Tuple[float, float]]=None,
        make_xcorrs: bool=False,
    ):
        """
        Args:
            datapath (str):
                Path to directory containing the 'snippet{idx}' subdirectories
            flip_vert (bool):
                When true, mirroring augmentation will be applied to data and labels
            flip_horiz (bool):
                When true, mirroring augmentation will be applied to data and labels
            segment_len (int):
                Length of the segments sampled from individual vocalizations within
                the dataset
            map_size (int):
                When provided, informs the dataloader of the size of the confidence maps
                generated by the model. When none, x and y coordinates will be provided 
                as labels.
            arena_dims (float tuple):
                The width and length of the arena in which sounds are localized. Used to
                scale labels (in millimeters)
            make_xcorrs (bool):
                When true, the dataset will compute pairwise cross correlations between
                the traces of all provided microphones.
        """
        self.datapath = datapath
        self.dataset = h5py.File(datapath, 'r')
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz
        self.segment_len = segment_len
        self.samp_size = 1
        self.map_dim = map_size
        self.arena_dims = arena_dims
        self.make_xcorrs = make_xcorrs
        self.n_channels = None

    def __del__(self):
        self.dataset.close()

    def __len__(self):
        if 'len_idx' in self.dataset:
            return len(self.dataset['len_idx']) - 1
        return len(self.dataset['vocalizations'])

    def _crop_audio(self, audio):
        # TODO: Delete this fn?
        pad_len = self.crop_audio
        n_samples = audio.shape[0]
        n_channels = audio.shape[1]

        new_len = min(pad_len, n_samples)
        clipped_audio = audio[:new_len, ...].T  # Change shape to (n_mics, <=pad_len)

        zeros = np.zeros((n_channels, pad_len), dtype=audio.dtype)
        zeros[:, :clipped_audio.shape[1]] = clipped_audio
        return zeros
    
    @classmethod
    def sample_segment(cls, audio, section_len):
        """ Samples a contiguous segment of length `section_len` from audio sample `audio` randomly
        within margins extending 10% of the total audio length from either end of the audio sample.

        Returns: audio segment with shape (n_channels, section_len)
        """
        n_samp = len(audio)
        margin = int(n_samp * 0.1)
        idx_range = margin, n_samp-margin-section_len
        if n_samp - 2*margin <= section_len:
            # If section_len is longer than the audio we're sampling from, randomly place the entire
            # audio sample within a block of zeros of length section_len
            padding = np.zeros((audio.shape[1], section_len))
            offset = randint(-margin, margin, (1,)).item()
            end = min(audio.shape[0] + offset, section_len)
            if offset < 0:
                padding[:, :end] = audio[-offset:end-offset, :].T
            else:
                padding[:, offset:end] = audio[:end-offset, :].T
            return padding
        start = randint(*idx_range, (1,)).item()
        end = start + section_len
        return audio[start:end, ...].T

    @classmethod
    def _append_xcorr(cls, audio, *, is_batch=False):
        # Assumes the audio has shape (n_channels, n_samples), which is true
        # after sample_segment has been called
        # Assumes unbatched input
        if is_batch:
            n_channels = audio.shape[1]
            audio_with_corr = np.empty((audio.shape[0], n_channels + comb(n_channels, 2), audio.shape[2]), audio.dtype)
        else:
            n_channels = audio.shape[0]
            audio_with_corr = np.empty((n_channels + comb(n_channels, 2), audio.shape[1]), audio.dtype)
        self.n_channels = n_channels
        audio_with_corr[..., :n_channels, :] = audio

        if is_batch:
            for batch in range(audio.shape[0]):
                for n, (a, b) in enumerate(combinations(audio[batch], 2)):
                    # a and b are mic traces
                    corr = correlate(a, b, 'same')
                    audio_with_corr[batch, n+n_channels, :] = corr
        else:
            for n, (a, b) in enumerate(combinations(audio, 2)):
                # a and b are mic traces
                corr = correlate(a, b, 'same')
                audio_with_corr[n+n_channels, :] = corr
        
        return audio_with_corr
        
    
    def _audio_for_index(self, dataset, idx):
        """ Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        if 'len_idx' in dataset:
            start, end = dataset['len_idx'][idx:idx+2]
            audio = dataset['vocalizations'][start:end, ...]
            if self.n_channels is None:
                self.n_channels = audio.shape[1]
            return audio
        else:
            return dataset['vocalizations'][idx]

    def _label_for_index(self, dataset, idx):
        location_map = dataset['locations'][idx]
        if self.map_dim is not None:
            # location_map = GerbilVocalizationDataset.gaussian_location_map(self.map_dim, location_map, 5)
            location_map = GerbilVocalizationDataset.wass_location_map(self.map_dim, location_map, arena_dims=self.arena_dims)
        return location_map
    
    @classmethod
    def scale_features(cls, inputs, labels, arena_dims, is_batch=False, n_mics=4):
        """ Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [0, 1].
        """

        if labels is not None:
            scaled_labels = np.empty_like(labels)

            is_map = len(labels.shape) == (3 if is_batch else 2)
            if not is_map:
                # Shift range to [-1, 1]
                x_scale = arena_dims[0] / 2  # Arena half-width (mm)
                y_scale = arena_dims[1] / 2
                scaled_labels[..., 0] = labels[..., 0] / x_scale
                scaled_labels[..., 1] = labels[..., 1] / y_scale
            else:
                lmin = labels.min(axis=(-1, -2), keepdims=True)
                lmax = labels.max(axis=(-1, -2), keepdims=True)
                scaled_labels = (labels - lmin) / (lmax - lmin)
        else:
            scaled_labels = None

        scaled_audio = np.empty_like(inputs)
        # std scaling: I think it's ok to use sample statistics instead of population statistics
        # because we treat each vocalization independantly of the others, their scale w.r.t other
        # vocalizations shouldn't affect our task
        raw_audio_mean = inputs[..., :n_mics, :].mean()
        raw_audio_std = inputs[..., :n_mics, :].std()
        scaled_audio[..., :n_mics, :] = (inputs[..., :n_mics, :] - raw_audio_mean) / raw_audio_std
        if n_mics < inputs.shape[-2]:
            xcorr_mean = inputs[..., n_mics:, :].mean()
            xcorr_std = inputs[..., n_mics:, :].std()
            scaled_audio[..., n_mics:, :] = (inputs[..., 4:, :] - xcorr_mean) / xcorr_std
        
        return scaled_audio, scaled_labels

    @classmethod
    def unscale_features(cls, labels, arena_dims):
        """ Changes the units of `labels` from arb. scaled unit (in range [0, 1]) to
        centimeters.
        """
        x_scale = arena_dims[0] / 2
        y_scale = arena_dims[1] / 2
        scaled_labels = np.empty_like(labels)
        scaled_labels[..., 0] = labels[..., 0] * x_scale / 10
        scaled_labels[..., 1] = labels[..., 1] * y_scale / 10
        return scaled_labels
    
    @classmethod
    def gaussian_location_map(cls, map_dim, location, sigma, arena_dims):
        """ Converts a single location to a confidence map.
        Params:
        loc (ndarray): A location. Should have shape (2,). Expected to have units of mm.
        sigma (float): Bandwidth of the gaussian kernel to use. Unit: pixels
        target_dims (tuple): A tuple of the form (width, height) for the output shape
        Returns: An ndarray of shape (height, width)
        """
        target_dims = (map_dim, map_dim)
        # Dimensions of the enclosure in mm
        # Also assumed to be the range of the input data
        half_width, half_height = arena_dims[0] / 2, arena_dims[1] / 2

        min_xy = np.array([-half_width, -half_height], dtype=np.float32)
        max_xy = np.array([half_width, half_height], dtype=np.float32)
        # The target reference frame starts at (0, 0)
        # scaled_loc should be between 0 and 1
        scaled_loc = (location - min_xy) / (max_xy - min_xy)

        x_vals = np.linspace(0, 1, target_dims[0])
        y_vals = np.linspace(0, 1, target_dims[1])
        # A 1x2 row vector broadcasted across n points
        coords = np.stack(np.meshgrid(x_vals, y_vals, indexing='xy'), axis=-1).reshape(-1, 1, 2)
        # Mean subtraction -> (x - \mu)^T vector
        coords -= scaled_loc.reshape((1, 1, 2))

        # Inverse covariance matrix
        sigma_x = sigma / (target_dims[1] ** 2)
        sigma_y = sigma / (target_dims[0] ** 2)
        inv_cov = np.diag([1 / sigma_y, 1 / sigma_x])

        # Everything in the exponent of the (multivariate) Normal PDF
        exponent = -0.5 * np.matmul(np.matmul(coords, inv_cov), coords.transpose(0, 2, 1))
        exponent = np.squeeze(exponent)

        gaussian = np.exp(exponent).astype(np.float32)
        return gaussian.reshape(target_dims[::-1])

    @classmethod
    def wass_location_map(cls, map_dim, loc, arena_dims, *, use_squared_dist=False):
        """ Creates a confidence map in which every pixel holds its distance (L2)
        from the pixel containing the true location.
        Params:
        loc (ndarray): A location. Should have shape (2,). Expected to have units of mm.
        target_dims (tuple): A tuple of the form (width, height) for the output shape
        use_squared_dist (bool): If true, the pixels are populated with the squared distance
        from the provided location
        Returns: An ndarray of shape (height, width)
        """
        target_dims = map_dim, map_dim
        # Dimensions of the enclosure in mm
        # Also assumed to be the range of the input data
        half_width, half_height = arena_dims[0] / 2, arena_dims[1] / 2
        float_dims = np.array(target_dims, dtype=np.float32)

        min_xy = np.array([-half_width, -half_height], dtype=np.float32)
        max_xy = np.array([half_width, half_height], dtype=np.float32)
        # The target reference frame starts at (0, 0)
        # scaled_loc should be between 0 and 1
        scaled_loc = (loc - min_xy) * float_dims / (max_xy - min_xy)
        scaled_loc = scaled_loc.astype(int)

        coord_grid = np.stack(np.meshgrid(
            np.arange(target_dims[0]), 
            np.arange(target_dims[1]), 
            indexing='xy'
        ), axis=-1).astype(np.float32)

        coord_grid -= scaled_loc.reshape((1, 1, 2))
        coord_grid = (coord_grid ** 2).sum(axis=-1)
        if not use_squared_dist:
            coord_grid = np.sqrt(coord_grid)
        return coord_grid

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
        sound = self._audio_for_index(self.dataset, idx)
    
        if self.samp_size > 1:
            sound = np.stack([self.sample_segment(sound, self.segment_len) for _ in range(self.samp_size)], axis=0)
        else:
            sound = self.sample_segment(sound, self.segment_len)

        if self.make_xcorrs:
            sound = GerbilVocalizationDataset._append_xcorr(sound, is_batch=self.samp_size>1)

        # Load animal location in the environment.
        # shape: (2 (x/y coordinates), )
        location_map = self._label_for_index(self.dataset, idx)
        sound, location_map = GerbilVocalizationDataset.scale_features(sound, location_map, is_batch=self.samp_size>1, arena_dims=self.arena_dims, n_mics=self.n_channels)

        is_map = self.map_dim is not None

        # With p = 0.5, flip vertically
        if self.flip_vert and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            if is_map:
                location_map = location_map[::-1, :]
            else:
                location_map[1] *= -1
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
            if sound.shape[0] == 10:
                sound = sound[[3, 2, 1, 0, 9, 8, 6, 7, 5, 4]]
            else:
                sound = sound[[3, 2, 1, 0]]

        # With p = 0.5, flip horizontally
        if self.flip_horiz and np.random.binomial(1, 0.5):
            # Assumes the center of the enclosure is (0, 0)
            if is_map:
                location_map = location_map[:, ::-1]
            else:
                location_map[0] *= -1
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
            if sound.shape[0] == 10:
                sound = sound[[1, 0, 3, 2, 4, 8, 7, 6, 5, 9]]
            else:
                sound = sound[[1, 0, 3, 2]]

        return sound.astype("float32"), location_map.astype("float32")


def build_dataloaders(path_to_data, CONFIG):

    # Construct Dataset objects.
    traindata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "train_set.h5"),
        flip_vert=(CONFIG["AUGMENT_LABELS"] and CONFIG["AUGMENT_FLIP_VERT"]),
        flip_horiz=(CONFIG["AUGMENT_LABELS"] and CONFIG["AUGMENT_FLIP_HORIZ"]),
        segment_len=CONFIG['SAMPLE_LEN'],
        arena_dims=(CONFIG['ARENA_WIDTH'], CONFIG['ARENA_LENGTH']),
        map_size=CONFIG['LOCATION_MAP_DIM'] if ('USE_LOCATION_MAP' in CONFIG) and CONFIG['USE_LOCATION_MAP'] else None,
        make_xcorrs=CONFIG['COMPUTE_XCORRS']
    )
    # TODO -- make new validation and test set files!
    valdata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "val_set.h5"),
        flip_vert=False, flip_horiz=False,
        segment_len=CONFIG['SAMPLE_LEN'],
        arena_dims=(CONFIG['ARENA_WIDTH'], CONFIG['ARENA_LENGTH']),
        map_size=CONFIG['LOCATION_MAP_DIM'] if ('USE_LOCATION_MAP' in CONFIG) and CONFIG['USE_LOCATION_MAP'] else None,
        make_xcorrs=CONFIG['COMPUTE_XCORRS']
    )
    testdata = GerbilVocalizationDataset(
        os.path.join(path_to_data, "test_set.h5"),
        flip_vert=False, flip_horiz=False,
        segment_len=CONFIG['SAMPLE_LEN'],
        arena_dims=(CONFIG['ARENA_WIDTH'], CONFIG['ARENA_LENGTH']),
        map_size=CONFIG['LOCATION_MAP_DIM'] if ('USE_LOCATION_MAP' in CONFIG) and CONFIG['USE_LOCATION_MAP'] else None,
        make_xcorrs=CONFIG['COMPUTE_XCORRS']
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
        shuffle=False
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=CONFIG["TEST_BATCH_SIZE"],
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader
