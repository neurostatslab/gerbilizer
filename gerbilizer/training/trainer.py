import logging
import os
from os import path
from typing import Generator, NewType, Tuple, Union

import h5py
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch_optimizer as optim

from ..outputs.base import ModelOutput, ProbabilisticOutput, Unit
from ..training.augmentations import build_augmentations
from ..training.dataloaders import build_dataloaders, GerbilVocalizationDataset
from ..training.logger import ProgressLogger
from ..training.models import build_model
from ..calibration import CalibrationAccumulator

try:
    # Attempt to use json5 if available
    import pyjson5 as json

    using_json5 = True
except ImportError:
    logging.warn("Warning: json5 not available, falling back to json.")
    import json

    using_json5 = False

JSON = NewType("JSON", dict)


def make_logger(filepath: str) -> logging.Logger:
    logger = logging.getLogger("train_log")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(filepath))
    return logger


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

        # disable torch 2.0 SDP optimizations
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_flash_sdp(False)
        # torch.backends.cuda.enable_math_sdp(False)

        self.__eval = eval_mode
        self.__datafile = data_dir
        self.__model_dir = model_dir
        self.__config = config_data

        if torch.cuda.is_available() and self.__config["GENERAL"]["DEVICE"] == "GPU":
            self.device = torch.device("cuda")
        elif (
            torch.backends.mps.is_available()
            and self.__config["GENERAL"]["DEVICE"] == "GPU"
        ):
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        if not self.__eval:
            self.__init_output_dir()
            self.__init_dataloaders()
        self.__init_model()

        self.augment = build_augmentations(self.__config)

        if not self.__eval:
            self.__logger.info(f" ==== STARTING TRAINING ====\n")
            self.__logger.info(
                f">> SAVING INITIAL MODEL WEIGHTS TO {self.__init_weights_file}"
            )
            self.save_weights(self.__init_weights_file)

            self.__best_loss = float("inf")

    def load_weights(self, weights: Union[dict, str]):
        device = self.device
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
        # if path.exists(self.__model_dir):
        #     raise ValueError(
        #         f"Model directory {self.__model_dir} already exists. Perhaps this job id is taken?"
        #     )
        # os.makedirs(self.__model_dir)

        self.__best_weights_file = os.path.join(self.__model_dir, "best_weights.pt")
        self.__init_weights_file = os.path.join(self.__model_dir, "init_weights.pt")
        self.__final_weights_file = os.path.join(self.__model_dir, "final_weights.pt")

        # Write the active configuration to disk
        self.__config["WEIGHTS_PATH"] = self.__best_weights_file
        # Found that it's helpful to keep track of this
        self.__config["DATA"]["DATAFILE_PATH"] = self.__datafile

        self.__init_logger()

        self.__traindata, self.__valdata, self.__testdata = self.__init_dataloaders()
        if self.__traindata is not None:
            self.__logger.info(f"Training set:\t{self.__traindata.dataset}")
        if self.__valdata is not None:
            self.__logger.info(f"Validation set:\t{self.__valdata.dataset}")
        if self.__testdata is not None:
            self.__logger.info(f"Test set:\t{self.__testdata.dataset}")

        self.__progress_log = ProgressLogger(
            self.__config["OPTIMIZATION"]["NUM_EPOCHS"],
            self.__traindata,
            self.__valdata,
            self.__config["GENERAL"]["LOG_INTERVAL"],
            self.__model_dir,
            self.__logger,
        )

    def __init_dataloaders(self):
        # Load training set, validation set, test set.
        return build_dataloaders(self.__datafile, self.__config)

    def __init_model(self):
        """Creates the model, optimizer, and loss function."""
        # Set random seeds.
        torch.manual_seed(self.__config["GENERAL"]["TORCH_SEED"])
        np.random.seed(self.__config["GENERAL"]["NUMPY_SEED"])

        # Specify network architecture and loss function.
        self.model, self.__loss_fn = build_model(self.__config)
        self.model = self.model.to(self.device)
        print(f"Moving model to device: {self.device}")
        print("SDP enabled: ")
        print(torch.backends.cuda.flash_sdp_enabled())
        print(torch.backends.cuda.mem_efficient_sdp_enabled())
        print(torch.backends.cuda.math_sdp_enabled())

        # In inference mode, there is no logger
        if not self.__eval:
            # The JSON5 library only supports writing in binary mode, but the built-in json library does not
            # Ensure this is written after the model has had the chance to update the config
            filemode = "wb" if using_json5 else "w"
            with open(os.path.join(self.__model_dir, "config.json"), filemode) as ctx:
                json.dump(self.__config, ctx, indent=4)

            self.__logger.info(self.model.__repr__())

            if self.__config["OPTIMIZATION"]["OPTIMIZER"] == "SGD":
                base_optim = torch.optim.SGD
                optim_args = {
                    "momentum": self.__config["OPTIMIZATION"]["MOMENTUM"],
                    "weight_decay": self.__config["OPTIMIZATION"]["WEIGHT_DECAY"],
                }
            elif self.__config["OPTIMIZATION"]["OPTIMIZER"] == "ADAM":
                base_optim = torch.optim.Adam
                optim_args = {
                    "betas": self.__config["OPTIMIZATION"]["ADAM_BETAS"],
                }
            elif self.__config["OPTIMIZATION"]["OPTIMIZER"] == "LAMB":
                base_optim = optim.Lamb
                optim_args = {
                    "betas": self.__config["OPTIMIZATION"]["LAMB_BETAS"],
                    "weight_decay": self.__config["OPTIMIZATION"]["WEIGHT_DECAY"],
                }
            else:
                raise NotImplementedError(
                    f'Unrecognized optimizer "{self.__config["OPTIMIZATION"]["OPTIMIZER"]}"'
                )

            params = (
                self.model.trainable_params()
                if hasattr(self.model, "trainable_params")
                else self.model.parameters()
            )
            self.__optim = base_optim(
                params,
                lr=self.__config["OPTIMIZATION"]["MAX_LEARNING_RATE"],
                **optim_args,
            )

            self.__scheduler = CosineAnnealingLR(
                self.__optim,
                T_max=self.__config["OPTIMIZATION"]["NUM_EPOCHS"],
                eta_min=self.__config["OPTIMIZATION"]["MIN_LEARNING_RATE"],
            )

    def train_epoch(self):
        # Set the learning rate using cosine annealing.
        self.__progress_log.start_training()
        self.model.train()

        iter = 0
        for sounds, locations in self.__traindata:
            # Move data to device
            sounds = sounds.to(self.device)
            locations = locations.to(self.device)

            # This should always exist, but might be the identity function
            sounds = self.augment(sounds)

            # Prepare optimizer.
            self.__optim.zero_grad()

            # Forward pass
            outputs = self.model(sounds)

            # Compute loss.
            losses = self.__loss_fn(outputs, locations)
            mean_loss = torch.mean(losses)

            # Backwards pass.
            mean_loss.backward()
            if self.__config["OPTIMIZATION"]["CLIP_GRADIENTS"]:
                self.model.clip_grads()
            self.__optim.step()

            # Count batch as completed.
            self.__progress_log.log_train_batch(
                mean_loss.item(), np.nan, sounds.shape[0]
            )
            iter += 1
            # if iter > 10: break
        self.__scheduler.step()

    def eval_validation(self):
        self.__progress_log.start_testing()
        self.model.eval()
        if self.__valdata is not None:
            with torch.no_grad():
                idx = 0
                compute_calibration = False
                for sounds, locations in self.__valdata:
                    if self.__config["GENERAL"]["DEVICE"] == "GPU":
                        sounds = sounds.to(self.device)

                    batch_err = 0

                    outputs: list[ModelOutput] = self.model(sounds, unbatched=True)

                    for (output, location) in zip(outputs, locations):
                        # Calculate error in cm
                        point_estimate = output.point_estimate(units=Unit.CM).cpu().numpy()
                        location = output._convert(
                            location, Unit.ARBITRARY, Unit.CM
                            ).cpu().numpy()

                        err = np.linalg.norm(point_estimate - location, axis=-1).item()
                        batch_err += err

                        if isinstance(output, ProbabilisticOutput):
                            compute_calibration = True
                            if idx == 0:
                                ca = CalibrationAccumulator(output.arena_dims[Unit.MM].cpu().numpy())
                            location_mm = location * 10
                            ca.calculate_step(output, location_mm)

                        idx += 1

                    # Log progress
                    self.__progress_log.log_val_batch(
                        batch_err / sounds.shape[0], np.nan, sounds.shape[0] * sounds.shape[1]
                    )

            # Done with epoch.
            cal_curve = None
            if compute_calibration:
                cal_curve = ca.results()['calibration_curve']
            val_loss = self.__progress_log.finish_epoch(calibration_curve=cal_curve)

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
    ) -> Generator[ModelOutput, None, None]:
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
            make_xcorrs=self.__config["DATA"]["COMPUTE_XCORRS"],
            arena_dims=arena_dims,
            crop_length=self.__config["DATA"].get("CROP_LENGTH", None),
            inference=True,
        )

        dloader = DataLoader(dset, collate_fn=dset.collate_fn)

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for batch in dloader:
                data = batch  # (1, channels, seq)
                output = self.model(data.to(self.device))
                yield output

        if should_close_file:
            dataset.close()
