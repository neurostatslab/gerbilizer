"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from itertools import combinations
from math import comb
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.signal import correlate
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        *,
        make_xcorrs: bool = False,
        inference: bool = False,
        crop_length: int = None,
        arena_dims: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
    ):
        """A dataloader designed to return batches of vocalizations with similar lengths

        Args:
            datapath (str): Path to HDF5 dataset
            make_xcorrs (bool, optional): Triggers computation of pairwise correlations between input channels. Defaults to False.
            inference (bool, optional): When true, labels will be returned in addition to data. Defaults to False.
            crop_length (int): When provided, will serve random crops of fixed length instead of full vocalizations
        """
        if isinstance(datapath, str):
            self.dataset = h5py.File(datapath, "r")
        else:
            self.dataset = datapath

        if "len_idx" not in self.dataset:
            raise ValueError("Improperly formatted dataset")

        self.make_xcorrs = make_xcorrs
        self.inference = inference
        self.arena_dims = arena_dims
        self.crop_length = crop_length
        self.n_channels = None
    
    def __len__(self):
        return len(self.dataset['len_idx']) - 1

    def __getitem__(self, idx):
        return self.__processed_data_for_index__(idx)

    @property
    def n_vocalizations(self):
        """
        The number of vocalizations contained in this Dataset object.
        """
        return len(self)

    def __del__(self):
        self.dataset.close()

    @staticmethod
    def __make_crop(audio: torch.Tensor, crop_length: int):
        """Given an audio sample of shape (n_samples, n_channels), return a random crop
        of shape (crop_length, n_channels)
        """
        audio_len, _ = audio.shape
        valid_range = audio_len - crop_length
        if valid_range <= 0:  # Audio is shorter than desired crop length, pad right
            pad_size = crop_length - audio_len
            # will fail if input is numpy array
            return F.pad(audio, (0, 0, 0, pad_size))
        range_start = np.random.randint(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    @staticmethod
    def __append_xcorr(audio: Union[torch.Tensor, np.ndarray]):
        is_batch = len(audio.shape) == 3
        n_channels = audio.shape[-1]

        audio_with_corr = torch.empty(
            (*audio.shape[:-1], n_channels + comb(n_channels, 2)),
            dtype=audio.dtype,
        )

        audio_with_corr[..., :n_channels] = audio

        if is_batch:
            for batch in range(audio.shape[0]):
                for n, (a, b) in enumerate(combinations(audio[batch].T, 2)):
                    # a and b are mic traces
                    corr = torch.from_numpy(correlate(a, b, "same"))
                    audio_with_corr[batch, :, n + n_channels] = corr
        else:
            for n, (a, b) in enumerate(combinations(audio.T, 2)):
                # a and b are mic traces
                corr = torch.from_numpy(correlate(a, b, "same"))
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
    def scale_features(
        inputs: np.ndarray,
        labels: np.ndarray,
        arena_dims: Tuple[int, int],
        *,
        n_mics: int = 4,
    ):
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
            # ensure cross correlations are scaled independently of the raw audio
            xcorr_mean = inputs[..., n_mics:].mean(axis=(-2, -1), keepdims=True)
            xcorr_std = inputs[..., n_mics:].std(axis=(-2, -1))
            scaled_audio[..., n_mics:] = (inputs[..., n_mics:] - xcorr_mean) / xcorr_std

        return scaled_audio, scaled_labels

    def __processed_data_for_index__(self, idx: int):
        sound = self.__audio_for_index(self.dataset, idx).astype(np.float32)
        sound = torch.from_numpy(sound)  # Padding numpy arrays yields an error
        sound = self.__make_crop(sound, self.crop_length)

        if self.make_xcorrs:
            sound = self.__append_xcorr(
                sound
            )  # I verified that scipy.signal.correlate accepts torch.Tensor
            # this function always returns a torch Tensor

        # Load animal location in the environment.
        # shape: (2 (x/y coordinates), )
        location = None if self.inference else self.__label_for_index(self.dataset, idx)

        arena_dims = np.array(self.arena_dims)
        sound, location = GerbilVocalizationDataset.scale_features(
            sound,
            location,
            arena_dims=arena_dims,
            n_mics=self.n_channels,
        )

        if self.inference:
            return sound
        return sound, location


def build_dataloaders(path_to_data: str, config: dict):
    # Construct Dataset objects.
    train_path = os.path.join(path_to_data, "train_set.h5")
    val_path = os.path.join(path_to_data, "val_set.h5")
    test_path = os.path.join(path_to_data, "test_set.h5")

    arena_dims = config["DATA"]["ARENA_DIMS"]
    make_xcorrs = config["DATA"]["COMPUTE_XCORRS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    crop_length = config["DATA"]["CROP_LENGTH"]

    avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)

    if os.path.exists(train_path):
        traindata = GerbilVocalizationDataset(
            train_path,
            arena_dims=arena_dims,
            make_xcorrs=make_xcorrs,
            crop_length=crop_length,
        )
        train_dataloader = DataLoader(
            traindata, batch_size=batch_size, shuffle=True, num_workers=avail_cpus
        )
    else:
        train_dataloader = None

    if os.path.exists(val_path):
        valdata = GerbilVocalizationDataset(
            val_path,
            arena_dims=arena_dims,
            make_xcorrs=make_xcorrs,
            crop_length=crop_length,
        )
        val_dataloader = DataLoader(valdata, batch_size=batch_size, num_workers=avail_cpus, shuffle=False)
    else:
        val_dataloader = None

    if os.path.exists(test_path):
        testdata = GerbilVocalizationDataset(
            test_path,
            arena_dims=arena_dims,
            crop_length=crop_length,
            make_xcorrs=make_xcorrs,
            inference=True,
        )
        test_dataloader = DataLoader(testdata, shuffle=False, batch_size=batch_size, num_workers=avail_cpus)
    else:
        test_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader
