import os
import argparse
import torch
import logging
import warnings
import numpy as np
import json

from configs import build_config
from models import build_model
from dataloaders import build_dataloaders
from augmentations import build_augmentations
from logger import ProgressLogger

# =============================== #
# === Parse Command Line Args === #
# =============================== #

parser = argparse.ArgumentParser()
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__),"..")
)

parser.add_argument(
    "--config",
    type=str,
    required=False,
    default="default",
    help="Used to specify model configuration.",
)

parser.add_argument(
    "--job_id",
    type=int,
    required=False,
    default=1,
    help="Used to seed random hyperparameters and save output.",
)

parser.add_argument(
    "--datafile",
    type=str,
    required=False,
    default=os.path.join(base_dir, "data"),
    help="Path to vocalization data.",
)

args = parser.parse_args()

# ============================================== #
# === Set up file structure to save results. === #
# ============================================== #

# Save results to `../../trained_models/{config}/{job_id}` directory.
output_dir = os.path.join(
    base_dir,
    "trained_models",
    args.config,
    "{0:05g}".format(args.job_id)
)
os.makedirs(output_dir, exist_ok=True)

# Configure log file.
log_filepath = os.path.join(output_dir, "train_log.txt")
logging.basicConfig(
    level=logging.DEBUG,
    filename=log_filepath,
    format="%(asctime)-15s %(levelname)-8s %(message)s"
)
print(f"Saving logs to file: `{log_filepath}`")

# Create `./results/{job id}/ckpts/{ckpt id}` folders for model weights.
best_weights_file = os.path.join(output_dir, "best_weights.pt")
init_weights_file = os.path.join(output_dir, "init_weights.pt")
final_weights_file = os.path.join(output_dir, "final_weights.pt")

# ========================== #
# === Construct Network. === #
# ========================== #

# Specify hyperparameters
CONFIG = build_config(args.config, args.job_id)

# Print out simple human-readable summary of job parameters.
with open(os.path.join(output_dir, "config.json"), "w") as f:
    f.write(json.dumps(CONFIG, indent=4))

logging.info("Wrote config.txt file".format(args.job_id))

# Set random seeds. Note that numpy random seed will affect
# the data augmentation under the current implementation.
torch.manual_seed(CONFIG["TORCH_SEED"])
np.random.seed(CONFIG["NUMPY_SEED"])

# Specify network architecture and loss function.
model, loss_function = build_model(CONFIG)
logging.info(model.__repr__())
logging.info(f"Model device: {next(model.parameters()).device.__repr__()}")

# Specify optimizer.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.0, # this is set manually at the start of each epoch.
    momentum=CONFIG["MOMENTUM"],
    weight_decay=CONFIG["WEIGHT_DECAY"]
)

# Cosine annealing learning rate decay.
def lr_schedule(epoch):
    """
    Given integer `epoch` specifying the current training
    epoch, return learning rate.
    """
    lm0 = CONFIG["MIN_LEARNING_RATE"]
    lm1 = CONFIG["MAX_LEARNING_RATE"]
    f = epoch / CONFIG["NUM_EPOCHS"]
    return (
        lm0 + 0.5 * (lm1 - lm0) * (1 + np.cos(f * np.pi))
    )

# Load training set, validation set, test set.
traindata, valdata, testdata = build_dataloaders(
    args.datafile, CONFIG
)
logging.info(
    "Training set:   " + traindata.dataset.__repr__()
)
logging.info(
    "Validation set:   " + valdata.dataset.__repr__()
)
logging.info(
    "Test set:   " + testdata.dataset.__repr__()
)

# Specifying data augmentations.
augment = build_augmentations(CONFIG)

# ========================== #
# === MAIN TRAINING LOOP === #
# ========================== #

# Logs progress during training.
progress = ProgressLogger(
    CONFIG["NUM_EPOCHS"],
    traindata,
    valdata,
    CONFIG["LOG_INTERVAL"],
    output_dir
)
best_loss = np.inf

# Save initial weights.
#  -> Note
logging.info(f" ==== STARTING TRAINING ====\n")
logging.info(f">> SAVING INITIAL MODEL WEIGHTS TO {init_weights_file}")
torch.save(model.state_dict(), init_weights_file)

for epochcount in range(CONFIG["NUM_EPOCHS"]):

    # Set the learning rate using cosine annealing.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_schedule(epochcount)

    # === TRAIN FOR ONE EPOCH === #
    progress.start_training()
    model.train()
    for minibatch in traindata:
        if len(minibatch) == 3:
                sounds, lens, locations = minibatch
        else:
            sounds, locations = minibatch
            lens = None

        # Don't process partial batches.
        if len(sounds) != CONFIG["TRAIN_BATCH_SIZE"]:
            break

        # Move data to gpu, if desired.
        if CONFIG["DEVICE"] == "GPU":
            sounds = sounds.cuda()
            locations = locations.cuda()

        # Prepare optimizer.
        optimizer.zero_grad()

        # Forward pass, including data augmentation.
        if lens is None:
            outputs = model(
                augment(sounds, sample_rate=125000)
            )
        else:
            outputs = model(
               sounds,
               lens
            )

        # Compute loss.
        losses = loss_function(outputs, locations)
        mean_loss = torch.mean(losses)

        # Backwards pass.
        mean_loss.backward()
        optimizer.step()

        # Count batch as completed.
        progress.log_train_batch(
            mean_loss.item(), np.nan, len(sounds)
        )

    # === EVAL VALIDATION ACCURACY === #
    progress.start_testing()
    model.eval()
    with torch.no_grad():
        for minibatch in valdata:
            if len(minibatch) == 3:
                sounds, lens, locations = minibatch
            else:
                sounds, locations = minibatch
                lens = None

            # Move data to gpu
            if CONFIG["DEVICE"] == "GPU":
                sounds = sounds.cuda()
                locations = locations.cuda()

            # Forward pass.
            if lens is None:
                outputs = model(sounds)
            else:
                outputs = model(sounds, lens)

            # Compute loss.
            losses = loss_function(outputs, locations)
            mean_loss = torch.mean(losses)

            # Log progress
            progress.log_val_batch(
                mean_loss.item(), np.nan, len(sounds)
            )

    # Done with epoch.
    val_loss = progress.finish_epoch()

    # Save best set of weights.
    if val_loss < best_loss:
        best_loss = val_loss
        logging.info(
            f">> VALIDATION LOSS IS BEST SO FAR, SAVING WEIGHTS TO {best_weights_file}"
        )
        torch.save(model.state_dict(), best_weights_file)


# Done training.
logging.info(
    ">> FINISHED ALL EPOCHS. TOTAL TIME ELAPSED: " +
    progress.print_time_since_initialization()
)
logging.info(f">> SAVING FINAL MODEL WEIGHTS TO {final_weights_file}")
torch.save(model.state_dict(), final_weights_file)
