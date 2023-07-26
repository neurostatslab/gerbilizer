import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      ReduceLROnPlateau, SequentialLR)

from ..training.augmentations import build_audio_augmentations
from ..training.logger import ProgressLogger
from .losses import unsupervised_loss
from .trainer import make_logger

try:
    # Attempt to use json5 if available
    import pyjson5 as json

    using_json5 = True
except ImportError:
    logging.warn("Warning: json5 not available, falling back to json.")
    import json

    using_json5 = False
from .augmentations import (build_audio_augmentations,
                            build_spectrogram_augmentations)
from .dataloaders import build_spectrogram_dataloader
from .trainer import JSON, Trainer


class UnsupervisedTrainer(Trainer):
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        config_data: JSON,
    ):
        super().__init__(data_dir, model_dir, config_data, eval_mode=False)
        self.augment = None
        self.audio_augmentations = build_audio_augmentations(config_data)
        self.spec_augmentations = build_spectrogram_augmentations(config_data)
        self.recent_batches_processed = 0

    def __init_dataloaders(self):
        self.train_loader = build_spectrogram_dataloader(self.__datafile, self.__config)
        self.__train_iter = iter(self.train_loader)
    
    def __init_output_dir(self):
        self.__final_weights_file = os.path.join(self.__model_dir, 'final_weights.pt')

        self.__config['WEIGHTS_PATH'] = self.__final_weights_file
        self.__config['DATA']['DATAFILE_PATH'] = self.__datafile

        self.__init_logger()

        self.num_epochs: int = self.__config['OPTIMIZATION']['NUM_WEIGHT_UPDATES']

        self.__progress_log = ProgressLogger(
            self.num_epochs,
            self.__traindata,
            self.__traindata,
            self.__config['GENERAL']['LOG_INTERVAL'],
            self.__model_dir,
            self.__logger,
        )

    def __init_logger(self):
        log_filepath = Path(self.__model_dir) / 'train.log'
        self.__logger = make_logger(log_filepath)

    def __init_model(self):
        torch.manual_seed(self.__config["GENERAL"]["TORCH_SEED"])
        np.random.seed(self.__config["GENERAL"]["NUMPY_SEED"])

        self.loss_fn = unsupervised_loss

        filemode = "wb" if using_json5 else "w"
        with open(os.path.join(self.__model_dir, "config.json"), filemode) as ctx:
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
        batch = self.spec_augmentations(batch)
        output, mask = self.model(batch)
        loss = self.loss_fn(output, batch, mask)
        loss.backward()

        self.__optim.step()
        self.__scheduler.step()
        self.num_steps += 1
        self.recent_batches_processed += 1

        if self.__progress_log.last_log_time - time.time() > 5:
            self.__progress_log.log_train_batch(loss.item(), np.nan, self.recent_batches_processed)
            self.recent_batches_processed = 0

    def train(self):
        for _ in range(self.num_epochs):
            self.train_step()
    