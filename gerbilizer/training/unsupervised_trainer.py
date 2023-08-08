import json as normal_json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

from ..architectures.attentionnet import GerbilizerAttentionNet
from .logger import ProgressLogger
from .losses import unsupervised_cosine_loss, unsupervised_mse_loss
from .trainer import make_logger

try:
    # Attempt to use json5 if available
    import pyjson5 as json

    using_json5 = True
except ImportError:
    logging.warn("Warning: json5 not available, falling back to json.")
    import json

    using_json5 = False
from .dataloaders import build_unsupervised_dataloader
from .trainer import JSON, Trainer


class UnsupervisedTrainer:
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        config_data: JSON,
    ):
        """Parameters:
        - data_dir:
            Path to directory containing train/test/val files named train_set.h5, etc...
        - model_dir:
            Path to the directory that will hold model weights and logs
        - config_data:
            Contents of model config as a JSON object (python dictionary-like)
        """
        self.__datafile = data_dir
        self.__model_dir = model_dir
        self.__config = config_data

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.__init_dataloaders()
        self.__init_output_dir()
        self.__init_model()

        self.__logger.info(f" ==== STARTING TRAINING ====\n")
        self.__logger.info(
            f">> SAVING INITIAL MODEL WEIGHTS TO {self.__final_weights_file}"
        )

        self.save_weights(self.__final_weights_file)
        self.last_weight_update = time.time()

        self.recent_batches_processed = 0
        self.num_steps = 0
        self.num_train_steps = self.__config["OPTIMIZATION"]["NUM_TRAIN_STEPS"]
        self.steps_per_scheduler_update = self.__config["OPTIMIZATION"][
            "STEPS_PER_SCHEDULER_UPDATE"
        ]

    def __init_dataloaders(self):
        self.train_loader = build_unsupervised_dataloader(
            self.__datafile, self.__config
        )
        self.__train_iter = iter(self.train_loader)

    def __init_output_dir(self):
        self.__final_weights_file = os.path.join(self.__model_dir, "final_weights.pt")

        self.__config["WEIGHTS_PATH"] = self.__final_weights_file
        self.__config["DATA"]["DATAFILE_PATH"] = self.__datafile

        self.__init_logger()

        self.num_weight_updates = (
            self.__config["OPTIMIZATION"]["NUM_TRAIN_STEPS"]
            // self.__config["OPTIMIZATION"]["STEPS_PER_SCHEDULER_UPDATE"]
        )

        self.__progress_log = ProgressLogger(
            self.num_weight_updates,
            self.train_loader,
            self.train_loader,
            self.__config["GENERAL"]["LOG_INTERVAL"],
            self.__model_dir,
            self.__logger,
        )
        self.__progress_log.num_train_images = (
            self.__config["OPTIMIZATION"]["NUM_TRAIN_STEPS"]
            * self.__config["DATA"]["BATCH_SIZE"]
        )

    def __init_logger(self):
        log_filepath = Path(self.__model_dir) / "train.log"
        self.__logger = make_logger(log_filepath)

    def __init_model(self):
        torch.manual_seed(self.__config["GENERAL"]["TORCH_SEED"])
        np.random.seed(self.__config["GENERAL"]["NUMPY_SEED"])

        self.model = GerbilizerAttentionNet(self.__config)
        self.model.to(self.device)

        if self.__config["OPTIMIZATION"]["LOSS"] == "L2":
            self.loss_fn = unsupervised_mse_loss
        elif self.__config["OPTIMIZATION"]["LOSS"] == "COSINE":
            self.loss_fn = unsupervised_cosine_loss
        else:
            raise NotImplementedError(
                f'Unrecognized loss function {self.__config["OPTIMIZATION"]["LOSS"]}'
            )

        with open(os.path.join(self.__model_dir, "config.json"), "w") as ctx:
            normal_json.dump(self.__config, ctx, indent=4)

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
                remaining_dur = self.num_weight_updates - total_specified_already
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
            elif scheduler_type == "LINEAR":
                base_scheduler = LinearLR
                scheduler_args = {
                    "total_iters": epochs_active,
                    "start_factor": scheduler_config.get("START_SCALING_FACTOR", 1 / 5),
                    "end_factor": scheduler_config.get("END_SCALING_FACTOR", 1),
                }
            elif scheduler_type == "EXPONENTIAL_DECAY":
                base_scheduler = ExponentialLR
                scheduler_args = {
                    "gamma": scheduler_config["MULTIPLICATIVE_DECAY_FACTOR"]
                }
            elif scheduler_type == "REDUCE_ON_PLATEAU":
                base_scheduler = ReduceLROnPlateau
                scheduler_args = {
                    "factor": scheduler_config.get("MULTIPLICATIVE_DECAY_FACTOR", 0.1),
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

    def fetch_data(self):
        try:
            batch = next(self.__train_iter)
            return batch
        except StopIteration:
            self.__train_iter = iter(self.train_loader)
            return self.fetch_data()

    def train_step(self):
        self.model.train()
        self.__optim.zero_grad()

        batch = self.fetch_data()
        batch = batch.to(self.device)
        reconstructed_tokens, teacher_tokens = self.model(batch)
        loss = self.loss_fn(reconstructed_tokens, teacher_tokens)
        loss.backward()

        self.__optim.step()
        # Don't forget to step the teacher weights too
        self.model.tokenizer.step_teacher()
        self.num_steps += 1
        self.recent_batches_processed += 1

        return loss

    def save_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def train(self):
        self.__progress_log.start_training()
        while self.num_steps < self.num_train_steps:
            for _ in range(self.steps_per_scheduler_update):
                last_loss = self.train_step()
            self.__scheduler.step()
            self.__progress_log.log_train_batch(
                last_loss.item(),
                np.nan,
                self.steps_per_scheduler_update * self.__config["DATA"]["BATCH_SIZE"],
            )
            if (
                time.time() - self.last_weight_update
                > self.__config["OPTIMIZATION"]["SAVE_WEIGHTS_INTERVAL"]
            ):
                self.save_weights(self.__final_weights_file)
                self.last_weight_update = time.time()
