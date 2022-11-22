"""
Functions to construct torch Dataset, DataLoader
objects and specify data augmentation.
"""

from itertools import combinations
from math import comb
import os
from typing import Optional, Tuple, Union

import h5py
import numpy as np
from scipy.signal import correlate
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader


class GerbilVocalizationDataset(IterableDataset):
    def __init__(
        self,
        datapath: str, *,
        make_xcorrs: bool = False,
        inference: bool = False,
        sequential: bool = False,
        arena_dims: Optional[Tuple[float, float]] = None,
        max_padding: int = 64,
        max_batch_size: int = 125 * 60 * 32,
        augmentation_params: Optional[dict] = None,
    ):
        """A dataloader designed to return batches of vocalizations with similar lengths

        Args:
            datapath (str): Path to HDF5 dataset
            make_xcorrs (bool, optional): Triggers computation of pairwise correlations between input channels. Defaults to False.
            inference (bool, optional): When true, labels will be returned in addition to data. Defaults to False.
            sequential (bool, optional): When true, data will be yielded one-by-one in the order it appears in the dataset. Defaults to False.
            max_padding (int, optional): Maximum amount of padding that can be added to a vocalization
            max_batch_size (int, optional): Maximim number of samples returned (aggregate across batch)
        """
        if isinstance(datapath, str):
            self.dataset = h5py.File(datapath, "r")
        else:
            self.dataset = datapath

        if 'len_idx' not in self.dataset:
            raise ValueError("Improperly formatted dataset")
        
        self.make_xcorrs = make_xcorrs
        self.inference = inference
        self.sequential = sequential
        self.arena_dims = arena_dims
        self.n_channels = None

        self.lengths = np.diff(self.dataset['len_idx'][:])
        self.len_idx = np.argsort(self.lengths)
        self.max_padding = max_padding
        self.max_batch_size = max_batch_size

        self.returned_samples = 0
        self.max_returned_samples = self.dataset['len_idx'][-1]

        self.augmentation_params = augmentation_params if augmentation_params is not None else dict()
    
    def __len__(self):
        return self.max_returned_samples

    def __iter__(self):
        if self.inference or self.sequential:
            for idx in range(len(self.lengths)):
                yield self.__processed_data_for_index__(idx)
            return
        
        while self.returned_samples < self.max_returned_samples:
            start_idx = np.random.choice(self.len_idx)  # randomly sample the shortest vocalization in the batch
            end_idx = start_idx + 1
            cur_batch_size = self.lengths[self.len_idx[start_idx]]

            # Returns True if the requested audio sample isn't too long
            valid_len = lambda i: (self.lengths[self.len_idx[i]] - self.lengths[self.len_idx[start_idx]]) < self.max_padding

            while (cur_batch_size < self.max_batch_size) and (end_idx < len(self.lengths)) and (valid_len(end_idx)):
                cur_batch_size += self.lengths[self.len_idx[end_idx]]
                end_idx += 1  # sequentially append longer vocalizations to the batch
            
            batch = []
            labels = []
            for i in range(start_idx, end_idx):
                real_idx = self.len_idx[i]
                if self.inference:
                    audio = self.__processed_data_for_index__(real_idx)
                    batch.append(audio)
                else:
                    audio, label = self.__processed_data_for_index__(real_idx)
                    batch.append(audio)
                    labels.append(label)
            
            # Requires individual elements to have shape (seq, ...)
            batch = pad_sequence(batch, batch_first=True)  # Should return tensor of shape (batch, seq, num_channels)
            self.returned_samples += cur_batch_size
            yield batch, torch.stack(labels)
        self.returned_samples = 0  # Reset the count so epochs 2+ don't complete instantaneously

    @property
    def max_vocalization_length(self):
        return self.lengths.max()

    def __del__(self):
        self.dataset.close()

    def __append_xcorr(self, audio: np.ndarray):
        is_batch = len(audio.shape) == 3
        n_channels = audio.shape[-1]
        
        audio_with_corr = np.empty(
            (*audio.shape[:-1], n_channels + comb(n_channels, 2)),
            audio.dtype,
        )

        audio_with_corr[..., :n_channels] = audio

        if is_batch:
            for batch in range(audio.shape[0]):
                for n, (a, b) in enumerate(combinations(audio[batch].T, 2)):
                    # a and b are mic traces
                    corr = correlate(a, b, "same")
                    audio_with_corr[batch, :, n + n_channels] = corr
        else:
            for n, (a, b) in enumerate(combinations(audio.T, 2)):
                # a and b are mic traces
                corr = correlate(a, b, "same")
                audio_with_corr[:, n + n_channels] = corr

        return audio_with_corr

    def __audio_for_index(self, dataset: h5py.File, idx: int):
        """Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = dataset["len_idx"][idx : idx + 2]
        audio = dataset["vocalizations"][start:end, ...]
        if self.n_channels is None:
            self.n_channels = audio.shape[1]
        return audio

    def __label_for_index(self, dataset: h5py.File, idx: int):
        return dataset["locations"][idx]

    @staticmethod
    def scale_features(inputs: np.ndarray, labels: np.ndarray, arena_dims: Tuple[int, int], *,  n_mics: int=4):
        """Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [-1, 1].
        """

        scaled_labels = None
        if labels is not None and arena_dims is not None:
            scaled_labels = np.empty_like(labels)

            # Shift range to [-1, 1]
            x_scale = arena_dims[0] / 2  # Arena half-width (mm)
            y_scale = arena_dims[1] / 2
            scaled_labels = labels / np.array([x_scale, y_scale])

        scaled_audio = np.empty_like(inputs)
        # std scaling: I think it's ok to use sample statistics instead of population statistics
        # because we treat each vocalization independantly of the others, their scale w.r.t other
        # vocalizations shouldn't affect our task
        raw_audio_mean = inputs[..., :n_mics].mean(axis=(-2, -1), keepdims=True)
        raw_audio_std = inputs[..., :n_mics].std(axis=(-2, -1), keepdims=True)
        scaled_audio[..., :n_mics] = (
            inputs[..., :n_mics] - raw_audio_mean
        ) / raw_audio_std
        if n_mics < inputs.shape[-1]:
            xcorr_mean = inputs[..., n_mics:].mean(axis=(-2, -1), keepdims=True)
            xcorr_std = inputs[..., n_mics:].std(axis=(-2, -1))
            scaled_audio[..., n_mics:] = (
                inputs[..., n_mics:] - xcorr_mean
            ) / xcorr_std

        return scaled_audio, scaled_labels

    @staticmethod
    def unscale_features(labels: Union[np.ndarray, torch.Tensor], arena_dims: Union[Tuple[int, int], np.ndarray, torch.Tensor]):
        """ Changes the units of `labels` from arb. scaled unit (in range [-1, 1]) to
        centimeters.
        """
        if not any([isinstance(arena_dims, torch.Tensor), isinstance(arena_dims, np.ndarray)]):
            scale = np.array(arena_dims) / 2
        else:
            scale = arena_dims / 2
        scaled_labels = labels * scale
        return scaled_labels
    
    @staticmethod
    def add_noise(audio, snr_db):
        # Expects audio to have shape (L, C)
        noise = torch.randn_like(audio)

        # Choose one microphone as a reference, so the strength of the noise is the same across all channels
        audio_norm = torch.linalg.vector_norm(audio, dim=0)[0]
        noise_norm = torch.linalg.vector_norm(noise, dim=0, keepdim=True)
        
        snr = 10 ** (snr_db / 20)
        scale_factor = snr * noise_norm / audio_norm
        return audio * scale_factor + noise

    def __processed_data_for_index__(self, idx: int):
        sound = self.__audio_for_index(self.dataset, idx)

        if self.augmentation_params and self.augmentation_params["AUGMENT_DATA"] and np.random.rand() < self.augmentation_params["AUGMENT_SNR_PROB"]:
            sound = self.add_noise(
                sound, 
                np.random.uniform(
                    self.augmentation_params["AUGMENT_SNR_MIN"],
                    self.augmentation_params["AUGMENT_SNR_MAX"]
                )
            )
        
        if self.make_xcorrs:
            sound = self.__append_xcorr(sound)

        # Load animal location in the environment.
        # shape: (2 (x/y coordinates), )
        location_map = (
            None if self.inference else self.__label_for_index(self.dataset, idx)
        )

        arena_dims = np.array(self.arena_dims)
        sound, location_map = GerbilVocalizationDataset.scale_features(
            sound,
            location_map,
            arena_dims=arena_dims,
            n_mics=self.n_channels,
        )

        if self.inference:
            return torch.from_numpy(sound.astype("float32"))

        return torch.from_numpy(sound.astype("float32")), torch.from_numpy(location_map.astype("float32"))


def build_dataloaders(path_to_data, CONFIG):
    # Construct Dataset objects.
    train_path = os.path.join(path_to_data, "train_set.h5")
    val_path = os.path.join(path_to_data, "val_set.h5")
    test_path = os.path.join(path_to_data, "test_set.h5")

    collate_fn = lambda batch: batch[0]  # Prevent the dataloader from unsqueezing in a batch dimension of size 1
    augment_params = {k: v for k,v in CONFIG.items() if k.startswith('AUGMENT')}

    if os.path.exists(train_path):
        traindata = GerbilVocalizationDataset(
            train_path,
            arena_dims=(CONFIG["ARENA_WIDTH"], CONFIG["ARENA_LENGTH"]),
            make_xcorrs=CONFIG["COMPUTE_XCORRS"],
            max_batch_size=CONFIG["TRAIN_BATCH_MAX_SAMPLES"],
            augmentation_params=augment_params,
        )
        train_dataloader = DataLoader(traindata, collate_fn=collate_fn)
    else:
        train_dataloader = None

    if os.path.exists(val_path):
        valdata = GerbilVocalizationDataset(
            val_path,
            arena_dims=(CONFIG["ARENA_WIDTH"], CONFIG["ARENA_LENGTH"]),
            make_xcorrs=CONFIG["COMPUTE_XCORRS"],
            sequential=True,
        )
        val_dataloader = DataLoader(valdata, collate_fn=collate_fn)
    else:
        val_dataloader = None

    if os.path.exists(test_path):
        testdata = GerbilVocalizationDataset(
            test_path,
            arena_dims=(CONFIG["ARENA_WIDTH"], CONFIG["ARENA_LENGTH"]),
            make_xcorrs=CONFIG["COMPUTE_XCORRS"],
            inference=True,
        )
        test_dataloader = DataLoader(testdata, collate_fn=collate_fn)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader
