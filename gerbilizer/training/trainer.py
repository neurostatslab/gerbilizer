import json
import logging
import os
from os import path
from typing import Generator, NewType, Tuple, Union

import h5py
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# from augmentations import build_augmentations
from ..training.dataloaders import build_dataloaders, GerbilVocalizationDataset
from ..training.logger import ProgressLogger
from ..training.models import build_model


JSON = NewType("JSON", dict)


def make_logger(filepath: str) -> logging.Logger:
    logger = logging.getLogger("train_log")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(filepath))
    return logger


def l2_distance(preds: np.ndarray, labels: np.ndarray, arena_dims) -> np.ndarray:
    """
    Unscale predictions and locations, then return the l2 distances
    across the batch in centimeters.
    """
    pred_cm = GerbilVocalizationDataset.unscale_features(preds, arena_dims)
    label_cm = GerbilVocalizationDataset.unscale_features(labels, arena_dims)
    return np.linalg.norm(pred_cm - label_cm, axis=-1)


class Trainer:
    """A helper class for training and performing inference with Gerbilizer models"""

    @staticmethod
    def __query_mem_usage():
        if not torch.cuda.is_available():
            return ""
        used_gb = torch.cuda.max_memory_allocated() / (2**30)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (2**30)
        torch.cuda.reset_peak_memory_stats()
        return "Max mem. usage: {:.2f}/{:.2f}GiB".format(used_gb, total_gb)

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        config_data: JSON,
        *,
        eval_mode: bool = False,
    ):
        """Parameters:
        - data_dir:
            Path to directory containing train/test/val files named train_set.h5, etc...
        - model_dir:
            Path to the directory that will hold model weights and logs
        - config_data:
            Contents of model config as a JSON object (python dictionary-like)
        - eval_mode:
            Should be false when training and true when performing inference.
        """
        self.__eval = eval_mode
        self.__datafile = data_dir
        self.__model_dir = model_dir
        self.__config = config_data

        if not self.__eval:
            self.__init_output_dir()
            self.__init_dataloaders()

        self.__init_model()

        if not self.__eval:
            # self.__augment = build_augmentations(self.__config) if self.__config['AUGMENT_DATA'] else None
            self.__logger.info(f" ==== STARTING TRAINING ====\n")
            self.__logger.info(
                f">> SAVING INITIAL MODEL WEIGHTS TO {self.__init_weights_file}"
            )
            self.save_weights(self.__init_weights_file)

            self.__best_loss = float("inf")

    def load_weights(self, weights: Union[dict, str]):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if isinstance(weights, str):
            weights = torch.load(weights, map_location=device)
        self.model.load_state_dict(weights, strict=False)

    def save_weights(self, weight_path: str):
        torch.save(self.model.state_dict(), weight_path)

    def finalize(self):
        self.__logger.info(
            ">> FINISHED ALL EPOCHS. TOTAL TIME ELAPSED: "
            + self.__progress_log.print_time_since_initialization()
        )
        self.__logger.info(
            f">> SAVING FINAL MODEL WEIGHTS TO {self.__final_weights_file}"
        )
        self.save_weights(self.__final_weights_file)

    def __init_logger(self):
        log_filepath = os.path.join(self.__model_dir, "train_log.txt")
        self.__logger = make_logger(log_filepath)
        print(f"Saving logs to file: `{log_filepath}`")

    def __init_output_dir(self):
        if path.exists(self.__model_dir):
            raise ValueError(
                f"Model directory {self.__model_dir} already exists. Perhaps this job id is taken?"
            )
        os.makedirs(self.__model_dir)

        self.__best_weights_file = os.path.join(self.__model_dir, "best_weights.pt")
        self.__init_weights_file = os.path.join(self.__model_dir, "init_weights.pt")
        self.__final_weights_file = os.path.join(self.__model_dir, "final_weights.pt")

        # Write the active configuration to disk
        self.__config["WEIGHTS_PATH"] = self.__best_weights_file
        self.__config[
            "DATAFILE_PATH"
        ] = self.__datafile  # Found that it's helpful to keep track of this
        with open(os.path.join(self.__model_dir, "config.json"), "w") as ctx:
            json.dump(self.__config, ctx, indent=4)

        self.__init_logger()

        self.__traindata, self.__valdata, self.__testdata = self.__init_dataloaders()
        if self.__traindata is not None:
            self.__logger.info(f"Training set:\t{self.__traindata.dataset}")
        if self.__valdata is not None:
            self.__logger.info(f"Validation set:\t{self.__valdata.dataset}")
        if self.__testdata is not None:
            self.__logger.info(f"Test set:\t{self.__testdata.dataset}")

        self.__progress_log = ProgressLogger(
            self.__config["NUM_EPOCHS"],
            self.__traindata,
            self.__valdata,
            self.__config["LOG_INTERVAL"],
            self.__model_dir,
            self.__logger,
        )

    def __init_dataloaders(self):
        # Load training set, validation set, test set.
        return build_dataloaders(self.__datafile, self.__config)

    def __init_model(self):
        """Creates the model, optimizer, and loss function."""
        # Set random seeds. Note that numpy random seed will affect
        # the data augmentation under the current implementation.
        torch.manual_seed(self.__config["TORCH_SEED"])
        np.random.seed(self.__config["NUMPY_SEED"])

        # Specify network architecture and loss function.
        self.model, self.__loss_fn = build_model(self.__config)
        # In inference mode, there is no logger
        if not self.__eval:
            self.__logger.info(self.model.__repr__())

            if self.__config["OPTIMIZER"] == "SGD":
                base_optim = torch.optim.SGD
                optim_args = {
                    "momentum": self.__config["MOMENTUM"],
                    "weight_decay": self.__config["WEIGHT_DECAY"],
                }
            elif self.__config["OPTIMIZER"] == "ADAM":
                base_optim = torch.optim.Adam
                optim_args = {
                    "betas": (self.__config["ADAM_BETA1"], self.__config["ADAM_BETA2"])
                }
            else:
                raise NotImplementedError(
                    f'Unrecognized optimizer "{self.__config["OPTIMIZER"]}"'
                )

            params = (
                self.model.trainable_params()
                if hasattr(self.model, "trainable_params")
                else self.model.parameters()
            )
            self.__optim = base_optim(
                params, lr=self.__config["MAX_LEARNING_RATE"], **optim_args
            )
            self.__scheduler = CosineAnnealingLR(
                self.__optim,
                T_max=self.__config["NUM_EPOCHS"],
                eta_min=self.__config["MIN_LEARNING_RATE"],
            )

    def train_epoch(self):
        # Set the learning rate using cosine annealing.
        self.__progress_log.start_training()
        self.model.train()
        for sounds, locations in self.__traindata:
            # Move data to gpu, if desired.
            if self.__config["DEVICE"] == "GPU":
                sounds = sounds.cuda()
                locations = locations.cuda()
            # Prepare optimizer.
            self.__optim.zero_grad()

            # Forward pass, including data augmentation.
            # aug_input = self.__augment(sounds, sample_rate=125000) if self.__config['AUGMENT_DATA'] else sounds
            aug_input = sounds
            outputs = self.model(aug_input)

            # Compute loss.
            losses = self.__loss_fn(outputs, locations)
            mean_loss = torch.mean(losses)

            # Backwards pass.
            mean_loss.backward()
            if self.__config["CLIP_GRADIENTS"]:
                self.model.clip_grads()
            self.__optim.step()

            # Count batch as completed.
            self.__progress_log.log_train_batch(
                mean_loss.item(), np.nan, sounds.shape[0] * sounds.shape[1]
            )
        self.__scheduler.step()

    def eval_validation(self):
        self.__progress_log.start_testing()
        self.model.eval()
        arena_dims = (self.__config["ARENA_WIDTH"], self.__config["ARENA_LENGTH"])
        if self.__valdata is not None:
            with torch.no_grad():
                for sounds, locations in self.__valdata:
                    # Move data to gpu
                    if self.__config["DEVICE"] == "GPU":
                        sounds = sounds.cuda()
                    locations = locations.numpy()

                    # Forward pass.
                    outputs = self.model(sounds).cpu().numpy()

                    # Convert outputs and labels to centimeters from arb. unit
                    # But only if the outputs are x,y coordinate pairs
                    if outputs.ndim == 2 and outputs.shape[1] == 2:
                        mean_loss = l2_distance(outputs, locations, arena_dims).mean()
                    elif outputs.ndim == 3 and outputs.shape[1:] == (3, 2):
                        predicted_locations = outputs[:, 0]
                        mean_loss = l2_distance(
                            predicted_locations, locations, arena_dims
                        ).mean()
                    else:
                        losses = self.__loss_fn(outputs, locations)
                        mean_loss = torch.mean(losses).item()

                    # Log progress
                    self.__progress_log.log_val_batch(
                        mean_loss / 10.0, np.nan, sounds.shape[0] * sounds.shape[1]
                    )

            # Done with epoch.
            val_loss = self.__progress_log.finish_epoch()
        else:
            val_loss = 0

        # Save best set of weights.
        if val_loss < self.__best_loss or self.__valdata is None:
            self.__best_loss = val_loss
            fmt_val_loss = "{:.3f}cm".format(val_loss)
            self.__logger.info(
                f">> MEAN VALIDATION LOSS, {fmt_val_loss}, IS BEST SO FAR, SAVING WEIGHTS TO {self.__best_weights_file}"
            )
            self.save_weights(self.__best_weights_file)
        self.__logger.info(Trainer.__query_mem_usage())

    def eval_on_dataset(
        self,
        dataset: Union[str, h5py.File],
        arena_dims: Tuple[float, float],
    ) -> Generator[np.ndarray, None, None]:
        """Creates an iterator to perform inference on a given dataset
        Parameters:
        - dataset: Either an h5py File or path to an h5 file
        - arena_dims: dimensions of the arena, used to rescale data to real-world
            units. If not provided, outputs will remain unscaled.
        - samples_per_vocalization: Number of independant samples evaluated from each vocalization
        """
        should_close_file = False
        if isinstance(dataset, str):
            dataset = h5py.File(dataset, "r")
            should_close_file = True

        dset = GerbilVocalizationDataset(
            datapath=dataset,
            make_xcorrs=self.__config["COMPUTE_XCORRS"],
            arena_dims=arena_dims,
            crop_length=self.__config.get("CROP_LENGTH", None),
            inference=True,
        )

        dloader = DataLoader(dset)

        self.model.eval()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        with torch.no_grad():
            for batch in dloader:
                data = batch  # (1, channels, seq)
                output = self.model(data.to(device)).cpu().numpy()
                yield output

        if should_close_file:
            dataset.close()
