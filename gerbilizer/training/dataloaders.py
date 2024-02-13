"""
Functions to construct Datasets and DataLoaders for training and inference
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class GerbilVocalizationDataset(Dataset):
    def __init__(
        self,
        datapath: Union[Path, str],
        crop_length: int,
        *,
        inference: bool = False,
        arena_dims: Optional[Union[np.ndarray, Tuple[float, float]]] = None,
        index: Optional[np.ndarray] = None,
    ):
        """
        Args:
            datapath (Path): Path to HDF5 dataset
            inference (bool, optional): When true, data will be cropped deterministically. Defaults to False.
            crop_length (int): Length of audio samples to return.
            arena_dims (Optional[Union[np.ndarray, Tuple[float, float]]], optional): Dimensions of the arena in mm. Used to scale labels.
            index (Optional[np.ndarray], optional): An array of indices to use for this dataset. Defaults to None, which will use the full dataset
        """
        if isinstance(datapath, str):
            datapath = Path(datapath)
        if isinstance(datapath, Path):
            self.dataset = h5py.File(datapath, "r")
        else:
            self.dataset = datapath

        if not isinstance(arena_dims, np.ndarray):
            arena_dims = np.array(arena_dims).astype(np.float32)

        if "length_idx" not in self.dataset:
            raise ValueError("Improperly formatted dataset")

        self.inference = inference
        self.arena_dims = arena_dims
        self.crop_length = crop_length
        self.index = index

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        return len(self.dataset["length_idx"]) - 1

    def __getitem__(self, idx):
        return self.__processed_data_for_index__(idx)

    def __del__(self):
        self.dataset.close()

    def __make_crop(self, audio: torch.Tensor, crop_length: int):
        """Given an audio sample of shape (n_samples, n_channels), return a random crop
        of shape (crop_length, n_channels)
        """
        audio_len, _ = audio.shape
        valid_range = audio_len - crop_length
        if valid_range <= 0:  # Audio is shorter than desired crop length, pad right
            pad_size = crop_length - audio_len
            # will fail if input is numpy array
            return F.pad(audio, (0, 0, 0, pad_size))
        if self.inference:
            range_start = 0
        else:
            range_start = np.random.randint(0, valid_range)
        range_end = range_start + crop_length
        return audio[range_start:range_end, :]

    def __audio_for_index(self, dataset: h5py.File, idx: int):
        """Gets an audio sample from the dataset. Will determine the format
        of the dataset and handle it appropriately.
        """
        start, end = dataset["length_idx"][idx : idx + 2]
        audio = dataset["audio"][start:end, ...]
        audio = (audio - audio.mean()) / audio.std()
        return torch.from_numpy(audio.astype(np.float32))

    def __label_for_index(self, dataset: h5py.File, idx: int):
        if "locations" not in dataset:
            return None
        return torch.from_numpy(dataset["locations"][idx].astype(np.float32))

    def scale_features(
        self,
        audio: np.ndarray,
        labels: np.ndarray,
    ):
        """Scales the inputs to have zero mean and unit variance. Labels are scaled
        from millimeter units to an arbitrary unit with range [-1, 1].
        """

        scaled_labels = None
        if labels is not None and self.arena_dims is not None:
            # Shift range to [-1, 1]
            scaled_labels = labels / torch.from_numpy(self.arena_dims) * 2

        scaled_audio = (audio - audio.mean()) / audio.std()

        return scaled_audio, scaled_labels

    def __processed_data_for_index__(self, idx: int):
        sound = self.__audio_for_index(self.dataset, idx)
        sound = self.__make_crop(sound, self.crop_length)

        location = self.__label_for_index(self.dataset, idx)

        sound, location = self.scale_features(sound, location)

        return sound, location

    def collate(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate function for the dataloader. Takes a list of (audio, label) tuples and returns
        a batch of audio and labels.
        """
        audio, labels = [x[0] for x in batch], [x[1] for x in batch]
        audio = torch.stack(audio)
        if labels[0] is not None:
            labels = torch.stack(labels)
        else:
            labels = [None] * len(audio)
        return audio, labels


def build_dataloaders(
    path_to_data: Union[Path, str], config: dict, index_dir: Optional[Path]
):
    # Construct Dataset objects.
    if not isinstance(path_to_data, Path):
        path_to_data = Path(path_to_data)

    arena_dims = config["DATA"]["ARENA_DIMS"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    crop_length = config["DATA"]["CROP_LENGTH"]

    index_arrays = {"train": None, "val": None}
    if index_dir is not None:
        index_arrays["train"] = np.load(index_dir / "train_set.npy")
        index_arrays["val"] = np.load(index_dir / "val_set.npy")

    avail_cpus = max(1, len(os.sched_getaffinity(0)) - 1)

    if path_to_data.is_dir():
        train_path = path_to_data / "train_set.h5"
        val_path = path_to_data / "val_set.h5"
    else:
        train_path = path_to_data
        val_path = path_to_data
        if index_dir is None:
            # manually create train/val split
            with h5py.File(train_path, "r") as f:
                dset_size = len(f["length_idx"]) - 1
            full_index = np.arange(dset_size)
            rng = np.random.default_rng(0)
            rng.shuffle(full_index)
            index_arrays["train"] = full_index[: int(0.8 * dset_size)]
            index_arrays["val"] = full_index[int(0.8 * dset_size) :]

    traindata = GerbilVocalizationDataset(
        train_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        index=index_arrays["train"],
    )

    valdata = GerbilVocalizationDataset(
        val_path,
        arena_dims=arena_dims,
        crop_length=crop_length,
        inference=True,
        index=index_arrays["val"],
    )
    train_dataloader = DataLoader(
        traindata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=avail_cpus,
        collate_fn=traindata.collate,
    )

    val_dataloader = DataLoader(
        valdata,
        batch_size=batch_size,
        num_workers=avail_cpus,
        shuffle=False,
        collate_fn=valdata.collate,
    )

    return train_dataloader, val_dataloader
