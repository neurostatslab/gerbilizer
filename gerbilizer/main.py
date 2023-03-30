import argparse
import os
from os import path
import time

import h5py
import numpy as np

from gerbilizer.training.configs import build_config
from gerbilizer.training.trainer import Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # Configs can be provided as either a name, which then references an entry in the dictionary
    # located in configs.py, or as a path to a JSON file, when then uses the entries in that file
    # to override the default configuration entries.

    parser.add_argument(
        "--config",
        type=str,
        help="Used to specify model configuration via a JSON file.",
    )

    parser.add_argument(
        "--data",
        type=str,
        help="Path to directory containing train, test, and validation datasets or single h5 file for inference",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default=path.abspath("."),
        help="Directory for trained models' weights",
    )

    parser.add_argument(
        "--eval", action="store_true", help="Flag for running inference on the model."
    )

    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        help="When performing inference, location to store model output.",
    )

    parser.add_argument(
        "--bare",
        action="store_true",
        required=False,
        help=(
            "By default, this script creates a nested directory structure in "
            "which to store model output. This flag overrides this behavior, placing "
            "output like saved weights directly in the directory provided."
        ),
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def next_available_job_id(model_name, save_dir):
    job_dir = path.join(save_dir, "trained_models", model_name)
    if not path.exists(job_dir):
        return 1
    else:
        dirs = [f for f in os.listdir(job_dir) if path.isdir(path.join(job_dir, f))]
        # Finds the highest job id and adds 1, defualts to 1 if there are no existing jobs
        # TODO: is it worth removing the assumption that all of these names will be parsable as ints?
        return 1 if not dirs else 1 + max(int(d) for d in dirs)


def validate_args(args):
    if args.config is None:
        raise ValueError("No config file provided.")

    if not path.exists(args.config):
        raise ValueError(
            f"Requested config JSON file could not be found: {args.config}"
        )

    if args.save_path is None:
        raise ValueError("No save path (trained model storage location) provided.")

    # Although it's somewhat inappropriate, I've elected to load config JSON here because a
    # thorough validation of the user input involves validating the presently unloaded JSON
    args.config_data = build_config(args.config)

    if args.data is None:
        if "DATAFILE_PATH" not in args.config_data:
            raise ValueError(f"Error: no data files provided")
        else:
            args.data = args.config_data["DATAFILE_PATH"]

    args.job_id = next_available_job_id(args.config_data["CONFIG_NAME"], args.save_path)
    if "JOB_ID" in args.config_data:
        args.job_id = args.config_data["JOB_ID"]

    # place output directly into directory user provides if bare flag is enabled
    if args.bare:
        args.model_dir = args.save_path
    else:
        if not path.exists(args.save_path):
            os.makedirs(args.save_path)
        args.model_dir = path.join(
            args.save_path,
            "trained_models",
            args.config_data["CONFIG_NAME"],
            f"{args.job_id:0>5d}",
        )


def run_eval(args: argparse.Namespace, trainer: Trainer):
    # expects args.data to point toward a file rather than a directory
    # In this case, all three h5py.File objects held by the Trainer are None
    data_path = args.data
    arena_dims = args.config_data["ARENA_WIDTH"], args.config_data["ARENA_LENGTH"]
    if not (data_path.endswith(".h5") or data_path.endswith(".hdf5")):
        raise ValueError(
            "--data argument should point to an HDF5 file with .h5 or .hdf5 file extension"
        )
    if args.output_path is not None:
        dest_path = args.output_path
    else:
        dest_path = data_path.split(".h5")[0] + "_preds.h5"

    with h5py.File(dest_path, "w") as dest:
        # Copy true locations, if available
        with h5py.File(data_path, "r") as source:
            n_vox = (
                len(source["len_idx"]) - 1
                if "len_idx" in source
                else len(source["vocalizations"])
            )
            if "locations" in source:
                source.copy(source["locations"], dest["/"], "locations")
            if "room_dims" in source:
                arena_dims = None
                source.copy(source["room_dims"], dest["/"], "room_dims")

        shape = (n_vox, 3, 2)
        preds = dest.create_dataset("predictions", shape=shape, dtype=np.float32)

        start_time = time.time()
        for n, result in enumerate(
            trainer.eval_on_dataset(data_path, arena_dims=arena_dims)
        ):
            preds[n] = result.squeeze()
            if (n + 1) % 100 == 0:
                est_speed = (n + 1) / (time.time() - start_time)
                remaining_items = n_vox - n
                remaining_time = remaining_items / est_speed
                print(
                    f"Evaluation progress: {n+1}/{n_vox}. Est. remaining time: {int(remaining_time):d}s"
                )
        print("Done")


def run(args):
    weights = (
        args.config_data["WEIGHTS_PATH"] if "WEIGHTS_PATH" in args.config_data else None
    )

    # This modifies args.config_data['WEIGHTS_PATH']
    trainer = Trainer(
        data_dir=args.data,
        model_dir=args.model_dir,
        config_data=args.config_data,
        eval_mode=args.eval,
    )

    if weights is not None:
        trainer.load_weights(weights)

    if args.eval:
        run_eval(args, trainer)
    else:
        num_epochs = args.config_data["NUM_EPOCHS"]
        for _ in range(num_epochs):
            trainer.train_epoch()
            trainer.eval_validation()
        trainer.finalize()


if __name__ == "__main__":
    args = get_args()
    run(args)
