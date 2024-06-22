import json
import logging
import os
from pathlib import Path
from typing import NewType, Optional

import numpy as np
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    SequentialLR,
)

from ..calibration import CalibrationAccumulator
from ..outputs.base import ModelOutput, ProbabilisticOutput, Unit
from ..training.augmentations import build_augmentations
from ..training.dataloaders import build_dataloaders
from ..training.logger import ProgressLogger
from ..training.models import build_model

JSON = NewType("JSON", dict)


def make_logger(filepath: str) -> logging.Logger:
    logger = logging.getLogger("train_log")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(filepath))
    logger.addHandler(logging.StreamHandler())
    return logger


class Trainer:
    """A helper class for training and performing inference with Vocalocator models"""

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
        index_dir: Optional[Path] = None,
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

        if self.__config["GENERAL"]["DEVICE"] == "CPU":
            self.device = torch.device("cpu")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            # fall back
            torch.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        if not self.__eval:
            self.__init_dataloaders(index_dir)
            self.__init_output_dir()
        self.__init_model()

        self.augment = build_augmentations(self.__config)

        if not self.__eval:
            self.__logger.info(f" ==== STARTING TRAINING ====\n")
            self.__logger.info(
                f">> SAVING INITIAL MODEL WEIGHTS TO {self.__init_weights_file}"
            )
            self.save_weights(self.__init_weights_file)

            self.__best_loss = float("inf")

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
        self.__best_weights_file = os.path.join(self.__model_dir, "best_weights.pt")
        self.__init_weights_file = os.path.join(self.__model_dir, "init_weights.pt")
        self.__final_weights_file = os.path.join(self.__model_dir, "final_weights.pt")

        # Found that it's helpful to keep track of this
        self.__config["DATA"]["DATAFILE_PATH"] = self.__datafile

        self.__init_logger()

        if self.__traindata is not None:
            self.__logger.info(f"Training set:\t{self.__traindata.dataset}")
        if self.__valdata is not None:
            self.__logger.info(f"Validation set:\t{self.__valdata.dataset}")

        self.num_epochs: int = self.__config["OPTIMIZATION"]["NUM_EPOCHS"]
        self.__progress_log = ProgressLogger(
            self.num_epochs,
            self.__traindata,
            self.__valdata,
            self.__config["GENERAL"]["LOG_INTERVAL"],
            self.__model_dir,
            self.__logger,
        )

    def __init_dataloaders(self, index_dir: Optional[Path]):
        # Load training set, validation set, test set.
        self.__traindata, self.__valdata, testdata = build_dataloaders(
            self.__datafile, self.__config, index_dir
        )

        # save indices for reproducibility
        train_idx = self.__traindata.dataset.index
        val_idx = self.__valdata.dataset.index

        if train_idx is not None:
            index_dir = Path(self.__model_dir) / "indices"
            index_dir.mkdir(parents=True, exist_ok=True)
            np.save(index_dir / "train_set.npy", train_idx)
            np.save(index_dir / "val_set.npy", val_idx)

        if testdata is not None:
            test_idx = testdata.dataset.index
            np.save(index_dir / "test_set.npy", test_idx)

    def __init_model(self):
        """Creates the model, optimizer, and loss function."""
        # Set random seeds.
        torch.manual_seed(self.__config["GENERAL"]["TORCH_SEED"])
        np.random.seed(self.__config["GENERAL"]["NUMPY_SEED"])

        # Specify network architecture and loss function.
        self.model, self.__loss_fn = build_model(self.__config)
        if not self.__eval:
            self.__config["WEIGHTS_PATH"] = self.__best_weights_file
        self.model.to(self.device)

        # In inference mode, there is no logger
        if not self.__eval:
            with open(os.path.join(self.__model_dir, "config.json"), "w") as ctx:
                json.dump(self.__config, ctx, indent=4)

            self.__logger.info(self.model.__repr__())

            optim_config = self.__config["OPTIMIZATION"]
            if optim_config["OPTIMIZER"] == "SGD":
                base_optim = torch.optim.SGD
                optim_args = {
                    "momentum": optim_config["MOMENTUM"],
                    "weight_decay": optim_config["WEIGHT_DECAY"],
                }
            elif optim_config["OPTIMIZER"] == "ADAM":
                base_optim = torch.optim.Adam
                optim_args = {"betas": optim_config["ADAM_BETAS"]}
            else:
                raise NotImplementedError(
                    f'Unrecognized optimizer "{optim_config["OPTIMIZER"]}"'
                )

            params = (
                self.model.trainable_params()
                if hasattr(self.model, "trainable_params")
                else self.model.parameters()
            )

            self.__optim = base_optim(
                params,
                lr=optim_config["INITIAL_LEARNING_RATE"],
                **optim_args,
            )

            scheduler_configs = optim_config["SCHEDULERS"]

            schedulers = []
            epochs_active_per_scheduler = []

            for scheduler_config in scheduler_configs:
                scheduler_type = scheduler_config["SCHEDULER_TYPE"]
                # by default, if number of active epochs is not specified, default to
                # running for the remaining duration.
                epochs_active = scheduler_config.get("NUM_EPOCHS_ACTIVE")
                if epochs_active is None:
                    total_specified_already = sum(epochs_active_per_scheduler)
                    remaining_dur = self.num_epochs - total_specified_already
                    self.__logger.info(
                        "No `NUM_EPOCHS_ACTIVE` parameter passed to scheduler "
                        f"{scheduler_type}! Defaulting to remaining train duration, "
                        f"{remaining_dur}."
                    )
                    epochs_active = remaining_dur
                epochs_active_per_scheduler.append(epochs_active)

                # parse lr scheduler
                if scheduler_type == "COSINE_ANNEALING":
                    base_scheduler = CosineAnnealingLR
                    scheduler_args = {
                        "T_max": epochs_active,
                        "eta_min": scheduler_config.get("MIN_LEARNING_RATE", 0),
                    }
                elif scheduler_type == "EXPONENTIAL_DECAY":
                    base_scheduler = ExponentialLR
                    scheduler_args = {
                        "gamma": scheduler_config["MULTIPLICATIVE_DECAY_FACTOR"]
                    }
                elif scheduler_type == "REDUCE_ON_PLATEAU":
                    base_scheduler = ReduceLROnPlateau
                    scheduler_args = {
                        "factor": scheduler_config.get(
                            "MULTIPLICATIVE_DECAY_FACTOR", 0.1
                        ),
                        "patience": scheduler_config.get("PLATEAU_DECAY_PATIENCE", 10),
                        "threshold_mode": scheduler_config.get(
                            "PLATEAU_THRESHOLD_MODE", "rel"
                        ),
                        "threshold": scheduler_config.get("PLATEAU_THRESHOLD", 1e-4),
                        "min_lr": scheduler_config.get("MIN_LEARNING_RATE", 0),
                    }
                else:
                    raise NotImplementedError(
                        f'Unrecognized scheduler "{scheduler_config["SCHEDULER_TYPE"]}"'
                    )
                schedulers.append(base_scheduler(self.__optim, **scheduler_args))

            # sequential lr expects the points at which it should switch,
            # so take the cumulative sum and throw out the endpoint
            milestones = list(np.cumsum(epochs_active_per_scheduler))[:-1]
            self.__scheduler = SequentialLR(
                self.__optim, schedulers=schedulers, milestones=milestones
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

    def eval_validation(self):
        self.__progress_log.start_testing()
        self.model.eval()
        if self.__valdata is not None:
            with torch.no_grad():
                idx = 0
                compute_calibration = False
                for sounds, locations in self.__valdata:
                    sounds = sounds.to(self.device)

                    batch_err = 0

                    outputs: list[ModelOutput] = self.model(sounds, unbatched=True)

                    for output, location in zip(outputs, locations):
                        # Calculate error in cm
                        point_estimate = (
                            output.point_estimate(units=Unit.CM).cpu().numpy()
                        )
                        location = (
                            output._convert(location, Unit.ARBITRARY, Unit.CM)
                            .float()
                            .cpu()
                            .numpy()
                        )

                        # Ensure the same number of dimensions are being used
                        location = location[: point_estimate.shape[-1]]

                        err = np.linalg.norm(point_estimate - location, axis=-1).item()
                        batch_err += err

                        if (
                            isinstance(output, ProbabilisticOutput)
                            and location.shape[-1] == 2
                        ):
                            compute_calibration = True
                            if idx == 0:
                                ca = CalibrationAccumulator(
                                    output.arena_dims[Unit.MM].cpu().numpy()
                                )
                            location_mm = location * 10
                            ca.calculate_step(output, location_mm)

                        idx += 1

                    # Log progress
                    self.__progress_log.log_val_batch(
                        batch_err / sounds.shape[0],
                        np.nan,
                        sounds.shape[0],
                    )

            # Done with epoch.
            cal_curve = None
            if compute_calibration:
                cal_curve = ca.results()["calibration_curve"]
            val_loss = self.__progress_log.finish_epoch(calibration_curve=cal_curve)

        else:
            val_loss = 0.0

        # Save best set of weights.
        if val_loss < self.__best_loss or self.__valdata is None:
            self.__best_loss = val_loss
            fmt_val_loss = "{:.3f}cm".format(val_loss)
            self.__logger.info(
                f">> MEAN VALIDATION LOSS, {fmt_val_loss}, IS BEST SO FAR, SAVING WEIGHTS TO {self.__best_weights_file}"
            )
            self.save_weights(self.__best_weights_file)
        self.__logger.info(Trainer.__query_mem_usage())

        return val_loss

    def update_learning_rate(self, val_loss: float):
        """
        Update the learning rate after epoch completion.
        """
        if isinstance(self.__scheduler, ReduceLROnPlateau):
            self.__scheduler.step(val_loss)
        else:
            self.__scheduler.step()

    def train(self):
        """
        Train the model for self.num_epochs passes through the dataset.
        """
        for _ in range(self.num_epochs):
            self.train_epoch()
            val_loss = self.eval_validation()
            self.update_learning_rate(val_loss)
        self.finalize()
