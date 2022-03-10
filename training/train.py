import argparse
import json
import logging
import os

import numpy as np
import torch

from configs import build_config
from models import build_model
from dataloaders import build_dataloaders
# from augmentations import build_augmentations
from logger import ProgressLogger
from sam import SAM


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
os.makedirs(os.path.join(output_dir, 'val_sample'), exist_ok=True)

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
cuda_avail = torch.cuda.is_available()
logging.info(f"Torch cuda availability: {cuda_avail}")
if cuda_avail:
    logging.info(f"CUDA device name: {torch.cuda.get_device_name()}")

# Set random seeds. Note that numpy random seed will affect
# the data augmentation under the current implementation.
torch.manual_seed(CONFIG["TORCH_SEED"])
np.random.seed(CONFIG["NUMPY_SEED"])

# Specify network architecture and loss function.
model, loss_function = build_model(CONFIG)
logging.info(model.__repr__())

# Specify optimizer.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.0, # this is set manually at the start of each epoch.
    momentum=CONFIG["MOMENTUM"],
    weight_decay=CONFIG["WEIGHT_DECAY"]
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0
)

if CONFIG['USE_SAM_OPTIMIZATION']:
    sam_optimizer = SAM(
        model.parameters(),
        base_optimizer=type(optimizer),
        rho=0.05,
        adaptive=False
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
# augment = build_augmentations(CONFIG)
def augment(*args, **kwargs):
    return args[0]


# Move model to GPU, if desired.
if CONFIG["DEVICE"] == "GPU":
    model.cuda()

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

# Save initial weights.
#  -> Note
logging.info(" ==== STARTING TRAINING ====\n")
logging.info(f">> SAVING INITIAL MODEL WEIGHTS TO {init_weights_file}")
torch.save(model.state_dict(), init_weights_file)


def query_gpu_memory():
    total_mem = '{:.2f}'.format( torch.cuda.get_device_properties(0).total_memory / 2**30 )
    reserved = '{:.2f}'.format( torch.cuda.memory_reserved(0) / 2**30 )
    allocated = '{:.2f}'.format( torch.cuda.memory_allocated(0) / 2**30 )
    return f"Reserved GPU memory: {reserved}/{total_mem}GiB\tAllocated memory: {allocated}/{reserved}GiB"


def run_training(epoch_count, optimizer, model, loss_function, traindata):
    # Set the learning rate using cosine annealing.
    logging.info(f">> SETTING NEW LEARNING RATE: {lr_schedule(epoch_count)}")
    if not isinstance(optimizer, SAM):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_schedule(epoch_count)
    else:
        for param_group in optimizer.base_optimizer.param_groups:
            param_group['lr'] = lr_schedule(epoch_count)


    # === TRAIN FOR ONE EPOCH === #
    progress.start_training()
    model.train()
    if CONFIG["DEVICE"] == "GPU":
        logging.info(query_gpu_memory())
    for sounds, locations in traindata:

        # Don't process partial batches.
        if len(sounds) != CONFIG["TRAIN_BATCH_SIZE"]:
            break

        # Move data to gpu, if desired.
        if CONFIG["DEVICE"] == "GPU":
            sounds = sounds.cuda()
            locations = locations.cuda()

        augmented_inputs = augment(sounds, sample_rate=16000)

        # Normal training procedure (without SAM)
        if not isinstance(optimizer, SAM):
            # Prepare optimizer.
            optimizer.zero_grad()

            # Forward pass, including data augmentation.
            outputs = model(augmented_inputs)

            # Compute loss.
            losses = loss_function(outputs, locations)
            mean_loss = torch.mean(losses)
            reported_loss = mean_loss.item()

            # Backwards pass.
            mean_loss.backward()
            optimizer.step()
        else:
            # First pass through data:
            outputs = model(augmented_inputs)
            losses = loss_function(outputs, locations)
            mean_loss = torch.mean(losses)
            # Report the loss from the first pass
            reported_loss = mean_loss.item()
            mean_loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second pass through data:
            outputs = model(augmented_inputs)
            losses = loss_function(outputs, locations)
            mean_loss = torch.mean(losses)
            mean_loss.backward()
            optimizer.second_step(zero_grad=True)

        # Count batch as completed.
        progress.log_train_batch(
            reported_loss, np.nan, len(sounds)
        )


def run_validation(epoch_count, model, loss_function, valdata, best_loss):
    # === EVAL VALIDATION ACCURACY === #
    progress.start_testing()
    model.eval()
    with torch.no_grad():
        for sounds, locations in valdata:

            # Move data to gpu
            if CONFIG["DEVICE"] == "GPU":
                sounds = sounds.cuda()
                locations = locations.cuda()

            # Forward pass.
            outputs = model(sounds)

            # Compute loss.
            losses = loss_function(outputs, locations)
            mean_loss = torch.mean(losses)

            # Save one output for visualization:
            locs = locations.detach().cpu().numpy()
            # sample_idx = np.random.choice(len(locs), 1)[0]
            sample_pred = outputs.detach().cpu().numpy()
            np.save(os.path.join(output_dir, 'val_sample', 'sample_pred_{:0>3d}.npy'.format(epoch_count)), sample_pred)
            np.save(os.path.join(output_dir, 'val_sample', 'sample_true_{:0>3d}.npy'.format(epoch_count)), locs)

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
    return best_loss


best_loss = np.inf
for epochcount in range(CONFIG["NUM_EPOCHS"]):
    run_training(epochcount, optimizer, model, loss_function, traindata)
    best_loss = run_validation(epochcount, model, loss_function, valdata, best_loss)

# Done training.
logging.info(
    ">> FINISHED ALL EPOCHS. TOTAL TIME ELAPSED: " +
    progress.print_time_since_initialization()
)
logging.info(f">> SAVING FINAL MODEL WEIGHTS TO {final_weights_file}")
torch.save(model.state_dict(), final_weights_file)

# === EVAL TEST ACCURACY === #
logging.info(f">> EVALUATING ACCURACY ON TEST SET...")

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
