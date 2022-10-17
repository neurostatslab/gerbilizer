"""
Defines a helpful ProgressLogger class for displaying
training progress in a deep learning setting.
"""
import os
from time import time
import numpy as np


def format_seconds(seconds):
    """
    Converts decimal number of seconds into a nice string.
    """
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    mins, secs = divmod(remainder, 60)

    outstr = f"{secs:02.0f} secs"
    if mins > 0:
        outstr = f"{mins:02g} mins, " + outstr
    if hours > 0:
        outstr = f"{hours:02g} hours, " + outstr
    if days > 0:
        outstr = f"{days:02g} days, " + outstr

    return outstr


class LossAccumulator:
    def log(self, loss, num_correct, batch_size, elapsed):
        self.imagecount += batch_size
        self.total_loss += loss * batch_size
        self.total_correct += num_correct
        self.total_elapsed += elapsed

    def reset(self):
        self.total_correct = 0
        self.total_loss = 0.0
        self.total_elapsed = 0.0
        self.imagecount = 0

    @property
    def secs_per_image(self):
        return self.total_elapsed / self.imagecount

    @property
    def accuracy(self):
        return self.total_correct / self.imagecount

    @property
    def mean_loss(self):
        return self.total_loss / self.imagecount


class ProgressLogger:
    def __init__(
        self, num_epochs, traindata, valdata, log_interval, output_dir, logger
    ):

        self.num_epochs = num_epochs
        self.num_train_images = traindata.dataset.max_returned_samples
        self.num_val_images = valdata.dataset.max_returned_samples
        self.log_interval = log_interval
        self.epochcount = 0
        self.epoch_start_time = None
        self.time_ckpt = time()
        self.elapsed_training = 0.0
        self.elapsed_testing = 0.0
        self.last_log_time = -np.inf
        self.time_initialized = time()

        self.train_accumulator = LossAccumulator()
        self.val_accumulator = LossAccumulator()

        self.train_loss_filepath = os.path.join(output_dir, "train_loss.txt")
        self.val_loss_filepath = os.path.join(output_dir, "val_loss.txt")

        self.logger = logger

    def require_log(self):
        """Returns True if we need to log progress."""
        if (time() - self.last_log_time) > self.log_interval:
            self.last_log_time = time()
            return True
        else:
            return False

    def timer_split(self):
        """Returns time in seconds between each function call."""
        split = time() - self.time_ckpt
        self.time_ckpt = time()
        return split

    def start_training(self):

        # Increment epochs.
        self.epochcount += 1
        self.logger.info(">> STARTING EPOCH {}".format(self.epochcount))

        # Print estimated overall time remaining.
        if self.epochcount > 1:
            epochs_left = 1 + self.num_epochs - self.epochcount
            time_remaining = epochs_left * (
                self.train_accumulator.secs_per_image * self.num_train_images
                + self.val_accumulator.secs_per_image * self.num_val_images
            )
            self.logger.info(
                ">> TIME ELAPSED SO FAR:\t" + self.print_time_since_initialization()
            )
            self.logger.info(
                ">> EST. TIME REMAINING:\t" + format_seconds(time_remaining)
            )

        # Reset training statistics.
        self.train_accumulator.reset()

        # Used to print total time elapsed per epoch.
        self.epoch_start_time = time()

        # Reset timer before commencing training.
        self.timer_split()

    def log_train_batch(self, loss, num_correct, batch_size):

        # Log training statistics.
        self.train_accumulator.log(loss, num_correct, batch_size, self.timer_split())

        # Output training progress.
        if self.require_log():
            self.logger.info(
                "TRAINING. \t Epoch "
                + "{} / {} ".format(self.epochcount, self.num_epochs)
                + "[{}/{}]".format(
                    self.train_accumulator.imagecount, self.num_train_images
                )
                + f"   minibatch loss: {loss:.5f}"
            )

    def start_testing(self):

        # Log progress.
        self.logger.info(">> DONE TRAINING, STARTING TESTING.")

        # Reset testing statistics.
        self.val_accumulator.reset()

        # Reset time before commencing testing.
        self.timer_split()

    def log_val_batch(self, loss, num_correct, batch_size):

        # Log testing statistics.
        self.val_accumulator.log(loss, num_correct, batch_size, self.timer_split())
        # Output training progress.
        if self.require_log():
            self.logger.info(
                "TESTING VALIDATION SET. Epoch {} ".format(self.epochcount)
                + "[{}/{}]".format(self.val_accumulator.imagecount, self.num_val_images)
            )

    def finish_epoch(self):
        """Log statistics on the test set."""
        self.logger.info(
            ">> FINISHED EPOCH IN: " + format_seconds(time() - self.epoch_start_time)
        )

        train_loss = self.train_accumulator.mean_loss
        val_loss = self.val_accumulator.mean_loss

        with open(self.train_loss_filepath, "a") as f:
            f.write(f"{train_loss}\n")

        with open(self.val_loss_filepath, "a") as f:
            f.write(f"{val_loss}\n")

        return val_loss

    def print_time_since_initialization(self):
        """Display time since ProgressLogger was initialized."""
        return format_seconds(time() - self.time_initialized)
