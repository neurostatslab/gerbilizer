import argparse
import json
import logging
import os
from typing import NewType, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch

from architectures.tcp import TCP
from augmentations import build_augmentations
from configs import build_config_from_file, build_config_from_name
from dataloaders import build_dataloaders, GerbilVocalizationDataset
from logger import ProgressLogger
from models import build_model


logger = logging.getLogger('no_spam')
logger.setLevel(logging.DEBUG)


JSON = NewType('JSON', dict)
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

# =============================== #
# === Parse Command Line Args === #
# =============================== #

def get_args():
    parser = argparse.ArgumentParser()

    # Configs can be provided as either a name, which then references an entry in the dictionary
    # located in configs.py, or as a path to a JSON file, when then uses the entries in that file
    # to override the default configuration entries.

    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Used to specify model configuration via a hard-coded named configuration.",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Used to specify model configuration via a JSON file."
    )

    parser.add_argument(
        "--datafile",
        type=str,
        required=False,
        # default=os.path.join(base_dir, "data"),
        help="Path to vocalization data.",
    )

    parser.add_argument(
        '--pretrained',
        required=False,
        action='store_true',
        help='Flag indicating that the model is pretrained and will be finetuned.'
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def find_next_job_id(model_name):
    job_dir = os.path.join(
        base_dir,
        "trained_models",
        model_name
    )
    if not os.path.exists(job_dir):
        return 1
    else:
        dirs = [f for f in os.listdir(job_dir) if os.path.isdir(os.path.join(job_dir, f))]
        # Finds the highest job id and adds 1, defualts to 1 if there are no existing jobs
        return 1 if not dirs else 1 + max(int(d) for d in dirs)


def validate_args(args):
    if args.config_file is not None and not os.path.exists(args.config_file):
        raise ValueError(f"Requested config JSON file could not be found: {args.config_file}")
    
    if args.config_file is None and args.config is None:
        args.config = "default"

    # Although it's somewhat inappropriate, I've elected to load config JSON here because a
    # thorough validation of the user input involves validating the presently unloaded JSON
    if args.config_file is not None:
        args.config_data = build_config_from_file(args.config_file)
    else:
        # Only runs when config_file is not provided
        # Although the intention is for the user to only provide one of the two, this
        # prioritizes config_file for the sake of eliminating undefined behavior
        args.config_data = build_config_from_name(args.config)
    
    if args.datafile is None:
        if 'DATAFILE_PATH' not in args.config_data:
            raise ValueError(f"Error: no data files provided")
        else:
            args.datafile = args.config_data['DATAFILE_PATH']
    
    args.job_id = find_next_job_id(args.config_data['CONFIG_NAME'])
    if 'JOB_ID' in args.config_data:
        args.job_id = args.config_data['JOB_ID']
    else:
        args.config_data['JOB_ID'] = args.job_id


class Trainer():
    def __init__(self,
        datafile: str,
        config_data: JSON,
        job_id: int, *,
        eval_mode: bool=False
    ):
        self._eval = eval_mode
        self._datafile = datafile
        self._config = config_data
        self._job_id = job_id

        self.tcp = None
        self.tcp_optim = None
        
        if not self._eval:
            self._setup_dataloaders()
            self._setup_output_dirs()
            self._augment = build_augmentations(self._config)

        # setup_model depends on logger setup in setup_output_dirA
        # thus, the if statement on _eval is split in two
        self._setup_model()
        self.model.logger = logger

        if not self._eval:
            logger.info(f" ==== STARTING TRAINING ====\n")
            logger.info(f">> SAVING INITIAL MODEL WEIGHTS TO {self.init_weights_file}")
            self.save_weights(self.init_weights_file)

            self._best_loss = float('inf')
            self.last_completed_epoch = 0
            self.loss_history = list()

    def train_epoch(self, epoch_num):
        # Set the learning rate using cosine annealing.
        new_lr = self._get_lr(epoch_num)
        logger.info(f">> SETTING NEW LEARNING RATE: {new_lr}")
        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr
        if self.tcp_optim is not None:
            self.tcp_optim.param_groups[0]['lr'] = new_lr

        arena_dims = (self._config['ARENA_WIDTH'], self._config['ARENA_LENGTH'])
        self.progress_log.start_training()
        self.model.train()
        self.tcp.train()
        for sounds, locations in self.traindata:
            # Don't process partial batches.
            if len(sounds) != self._config["TRAIN_BATCH_SIZE"]:
                break

            # Move data to gpu, if desired.
            if self._config["DEVICE"] == "GPU":
                sounds = sounds.cuda()
                locations = locations.cuda()

            # Prepare optimizer.
            self.optim.zero_grad()
            if self.tcp_optim is not None:
                self.tcp_optim.zero_grad()

            # Forward pass, including data augmentation.
            aug_input = self._augment(sounds, sample_rate=125000) if self._config['AUGMENT_DATA'] else sounds
            if self.tcp is not None:
                outputs, embedding = self.model(aug_input, return_embedding=True)
                tcp_output = self.tcp(embedding)
            else:
                outputs = self.model(aug_input)

            # Compute loss.
            losses = self._loss_fn(outputs, locations)
            if self.tcp is not None:
                pred_cm = GerbilVocalizationDataset.unscale_features(outputs.detach().clone(), arena_dims)
                label_cm = GerbilVocalizationDataset.unscale_features(locations, arena_dims)
                # Bypass the mse loss in favor of mean error
                tcp_target = torch.sqrt(((label_cm - pred_cm) ** 2).sum(axis=-1))
                tcp_loss = torch.mean((tcp_target - tcp_output) ** 2)
                tcp_loss.backward()
                self.tcp_optim.step()

            mean_loss = torch.mean(losses)

            # Backwards pass.
            mean_loss.backward()
            if self._config['CLIP_GRADIENTS']:
                self.model._clip_grads()
            self.optim.step()

            # Count batch as completed.
            self.progress_log.log_train_batch(
                mean_loss.item(), np.nan, len(sounds)
            )

            if self._config['SAVE_LOSS_PLOT']:
                self.loss_history.append(mean_loss.detach().cpu().item())
            
        self.last_completed_epoch = epoch_num + 1

    def eval_validation(self):
        self.progress_log.start_testing()
        self.model.eval()
        self.tcp.eval()
        arena_dims = (self._config['ARENA_WIDTH'], self._config['ARENA_LENGTH'])
        with torch.no_grad():
            for sounds, locations in self.valdata:
                # Move data to gpu
                if self._config["DEVICE"] == "GPU":
                    sounds = sounds.cuda()
                    locations = locations.cuda()

                # Forward pass.
                if self.tcp is not None:
                    outputs, embedding = self.model(sounds, return_embedding=True)
                    tcp_output = self.tcp(embedding).cpu().numpy()
                else:
                    outputs = self.model(sounds)


                # Convert outputs and labels to centimeters from arb. unit
                # But only if the outputs are x,y coordinate pairs
                if outputs.dim() == 2 and outputs.shape[1] == 2:
                    pred_cm = GerbilVocalizationDataset.unscale_features(outputs, arena_dims)
                    label_cm = GerbilVocalizationDataset.unscale_features(locations, arena_dims)

                    if self.tcp is not None:
                        # Bypass the mse loss in favor of mean error
                        pred_cm = GerbilVocalizationDataset.unscale_features(outputs, arena_dims)
                        label_cm = GerbilVocalizationDataset.unscale_features(locations, arena_dims)

                        mean_loss = torch.sqrt(((label_cm - pred_cm) ** 2).sum(axis=-1)).cpu().numpy()
                        for a, b in zip(mean_loss.ravel(), tcp_output.ravel()):
                            logger.info('Minibatch mean loss: {:.1f}. TCP predicted: {:.1f}'.format(a, b))
                        mean_loss = mean_loss.mean()
                else:
                    losses = self._loss_fn(outputs, locations)
                    mean_loss = torch.mean(losses).cpu().item()

                # Log progress
                self.progress_log.log_val_batch(
                    mean_loss, np.nan, len(sounds)
                )
        
        if self._config['SAVE_SAMPLE_OUTPUT']:
            maps_np = outputs.detach().cpu().numpy()
            labels_np = locations.detach().cpu().numpy()
            np.save(
                os.path.join(self.sample_output_dir, 'epoch_{:0>4d}_pred.npy'.format(self.last_completed_epoch)),
                maps_np
            )
            np.save(
                os.path.join(self.sample_output_dir, 'epoch_{:0>4d}_true.npy'.format(self.last_completed_epoch)),
                labels_np
            )
        
        if self._config['SAVE_LOSS_PLOT']:
            try:
                plt.plot(self.loss_history, 'r-')
                plt.xlabel('Minibatch #')
                plt.ylabel('Loss (arb. unit)')
                plt.title("Training loss evolution")
                plt.yscale('log')
                plt_path = os.path.join(self.output_dir, 'loss_fig.png')
                plt.savefig(plt_path)
            except:
                pass  # If the loss is negative, things can go wrong here

        # Done with epoch.
        val_loss = self.progress_log.finish_epoch()

        # Save best set of weights.
        if val_loss < self._best_loss:
            self._best_loss = val_loss
            fmt_val_loss = '{:.3f}cm'.format(val_loss)
            logger.info(
                f">> MEAN VALIDATION LOSS, {fmt_val_loss}, IS BEST SO FAR, SAVING WEIGHTS TO {self.best_weights_file}"
            )
            self.save_weights(self.best_weights_file)
        logger.info(self._query_mem_usage())
    
    def _setup_output_dirs(self):
        # Save results to `../../trained_models/{config}/{job_id}` directory.
        self.output_dir = os.path.join(
            base_dir,
            "trained_models",
            self._config['CONFIG_NAME'],
            "{0:05g}".format(self._job_id)
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.sample_output_dir = os.path.join(
            self.output_dir,
            'val_predictions'
        )
        os.makedirs(self.sample_output_dir, exist_ok=True)

        # Write the active configuration to disk
        self._config['DATAFILE_PATH'] = self._datafile  # Found that it's helpful to keep track of this
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as ctx:
            json.dump(self._config, ctx, indent=4)

        # Configure log file.
        log_filepath = os.path.join(self.output_dir, 'train_log.txt')
        ch = logging.FileHandler(filename=log_filepath, mode='w')
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.progress_log = ProgressLogger(
            self._config["NUM_EPOCHS"],
            self.traindata,
            self.valdata,
            self._config["LOG_INTERVAL"],
            self.output_dir,
            logger
        )
        print(f"Saving logs to file: `{log_filepath}`")

        # Create `./results/{job id}/ckpts/{ckpt id}` folders for model weights.
        self.best_weights_file = os.path.join(self.output_dir, "best_weights.pt")
        self.init_weights_file = os.path.join(self.output_dir, "init_weights.pt")
        self.final_weights_file = os.path.join(self.output_dir, "final_weights.pt")
    
    def save_weights(self, weight_path):
        torch.save(self.model.state_dict(), weight_path)

    def _setup_model(self):
        """ Creates the model, optimizer, and loss function.
        """
        # Set random seeds. Note that numpy random seed will affect
        # the data augmentation under the current implementation.
        torch.manual_seed(self._config["TORCH_SEED"])
        np.random.seed(self._config["NUMPY_SEED"])

        # Specify network architecture and loss function.
        self.model, self._loss_fn = build_model(self._config)
        if not self._eval:
            logger.info(self.model.__repr__())

            if self._config['OPTIMIZER'] == 'SGD':
                base_optim = torch.optim.SGD
                optim_args = {
                    'momentum': self._config['MOMENTUM'],
                    'weight_decay': self._config['WEIGHT_DECAY']
                }
            elif self._config['OPTIMIZER'] == 'ADAM':
                base_optim = torch.optim.Adam
                optim_args = {
                    'betas': (self._config['ADAM_BETA1'], self._config['ADAM_BETA2'])
                }
            else:
                raise NotImplementedError(f'Unrecognized optimizer "{self._config["OPTIMIZER"]}"')

            params = self.model.trainable_params() if hasattr(self.model, 'trainable_params') else self.model.parameters()
            self.optim = base_optim(
                params,
                lr=0.0, # this is set manually at the start of each epoch.
                **optim_args
            )

    def _query_mem_usage(self):
        if not torch.cuda.is_available():
            return ''
        used_gb = torch.cuda.max_memory_allocated() / (2**30)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (2**30)
        torch.cuda.reset_max_memory_allocated()
        return 'Max mem. usage: {:.2f}/{:.2f}GiB'.format(used_gb, total_gb)

    def _setup_dataloaders(self):
        # Load training set, validation set, test set.
        self.traindata, self.valdata, self.testdata = build_dataloaders(
            self._datafile, self._config
        )
        self.traindata.dataset.segment_len = self._config['SAMPLE_LEN'] if 'SAMPLE_LEN' in self._config else 256
        self.valdata.dataset.segment_len = self._config['SAMPLE_LEN'] if 'SAMPLE_LEN' in self._config else 256
        logger.info(f"Training set:\t{self.traindata.dataset}")
        logger.info(f"Validation set:\t{self.valdata.dataset}")
        logger.info(f"Test set:\t{self.testdata.dataset}")
    
    def _setup_tcp(self):
        inp_dimension = self._config['TCP_IN_DIM']
        num_layers = self._config['TCP_N_LAYERS']
        num_hidden = self._config['TCP_N_HIDDEN']
        self.tcp = TCP(inp_dimension, num_layers, num_hidden, dropout=0.5)
        if torch.cuda.is_available():
            self.tcp.cuda()

        if self._config['OPTIMIZER'] == 'SGD':
            base_optim = torch.optim.SGD
            optim_args = {
                'momentum': self._config['MOMENTUM'],
                'weight_decay': self._config['WEIGHT_DECAY']
            }
        elif self._config['OPTIMIZER'] == 'ADAM':
            base_optim = torch.optim.Adam
            optim_args = {
                'betas': (self._config['ADAM_BETA1'], self._config['ADAM_BETA2'])
            }
        else:
            raise NotImplementedError(f'Unrecognized optimizer "{self._config["OPTIMIZER"]}"')

        params = self.tcp.parameters()
        self.tcp_optim = base_optim(
            params,
            lr=0.0, # this is set manually at the start of each epoch.
            **optim_args
        )


    def _get_lr(self, epoch_num: int) -> float:
        """
        Given integer `epoch` specifying the current training
        epoch, return learning rate.
        """
        lm0 = self._config["MIN_LEARNING_RATE"]
        lm1 = self._config["MAX_LEARNING_RATE"]
        f = epoch_num / self._config["NUM_EPOCHS"]
        # f = (epoch_num % 50) / 50
        return (
            lm0 + 0.5 * (lm1 - lm0) * (1 + np.cos(f * np.pi))
        )
    
    @classmethod
    def from_trained_model(cls,
        config: JSON, *,
        model_name: Optional[str]=None,
        job_id: Optional[int]=None,
        device_override: Optional[str]=None,
        finetune_data: Optional[str]=None
    ):
        """ Initializes the model with weights initialized from the best weights of a
        previously trained instance.
        Parameters:
        config (JSON):
            - config dictionary for the model
        model_name (str):
            - Model name under which the weights will be located. If not provided,
            will be sourced from config dict (assuming it's present)
        job_id (int):
            - Job ID in which the weights will be located. If not provided, will be
            sourced from config dict (assuming it's present)
        device_override (str):
            - When specified, overrides the device (cpu or gpu) present in the config
        finetune_data (str):
            - When specified, indicates that a trained model is being finetuned on provided data.
        """
        if job_id is None:
            if config['JOB_ID'] is None:
                raise ValueError('No job id provided. Cannot locate weights.')
            job_id = config['JOB_ID']
        if model_name is None:
            if config['CONFIG_NAME'] is None:
                raise ValueError('No model name provided. Cannot locate weights')
            model_name = config['CONFIG_NAME']
        
        if device_override is not None:
            device = device_override
            # This needs to be changed so the model is loaded into the right device
            config['DEVICE'] = device.upper()
        else:
            device = config['DEVICE'].lower()
        
        if finetune_data is not None:
            job_id = config['JOB_ID']  # Use the pretrained model's ID to load weights
            # Update config, which will be written to disk, with a fresh ID
            config['JOB_ID'] = find_next_job_id(config['CONFIG_NAME'])
        
        device = 'cuda:0' if device == 'gpu' else 'cpu'

        # Use new ID to create model folder
        trainer = cls(finetune_data, config, config['JOB_ID'], eval_mode=(finetune_data is None))
        if finetune_data is not None:
            trainer._setup_tcp()

        if 'WEIGHTS_PATH' in config:
            weight_file = config['WEIGHTS_PATH']
        else:
            weight_file = os.path.join(
                base_dir,
                "trained_models",
                model_name,
                "{0:05g}".format(job_id),  # Use the old ID to load weghhts
                "best_weights.pt"
            )

        weights = torch.load(weight_file, map_location=device)
        trainer.model.load_state_dict(weights, strict=False)

        return trainer


def run():
    args = get_args()
    if args.pretrained:
        trainer = Trainer.from_trained_model(
            args.config_data,
            job_id=args.job_id,
            finetune_data=args.datafile
        )
    else:
        trainer = Trainer(args.datafile, args.config_data, args.job_id)

    num_epochs = args.config_data['NUM_EPOCHS']
    for epoch in range(num_epochs):
        trainer.train_epoch(epoch)
        trainer.eval_validation()

    logger.info(
        ">> FINISHED ALL EPOCHS. TOTAL TIME ELAPSED: " +
        trainer.progress_log.print_time_since_initialization()
    )
    logger.info(f">> SAVING FINAL MODEL WEIGHTS TO {trainer.final_weights_file}")
    trainer.save_weights(trainer.final_weights_file)


if __name__ == '__main__':
    run()

# TODO: Reintroduce test loss after rewriting non-mse loss functions to produce losses
# in real world units
"""
# === EVAL TEST ACCURACY === #
logger.info(f">> EVALUATING ACCURACY ON TEST SET...")

total_testloss = 0.0
model.eval()
with torch.no_grad():
    for sounds, locations in testdata:

        # Move data to gpu
        if CONFIG["DEVICE"] == "GPU":
            sounds = sounds.cuda()
            locations = locations.cuda()

        # Forward pass.
        outputs = model(sounds)

        # Compute loss.
        losses = loss_function(outputs, locations)
        total_testloss += torch.sum(losses).item()

# Write test loss to file
with open(os.path.join(output_dir, "test_set_rmse.txt"), "w") as f:
    f.write(f"{1e3 * np.sqrt(total_testloss / len(testdata))}\n")
"""