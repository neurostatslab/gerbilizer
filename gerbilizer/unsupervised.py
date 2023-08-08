import argparse
from pathlib import Path

from gerbilizer.training.configs import build_config
from gerbilizer.training.unsupervised_trainer import UnsupervisedTrainer


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
        "--save-path",
        type=str,
        required=False,
        default=Path(".").absolute(),
        help="Directory for trained models' weights",
    )

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args):
    if args.config is None:
        raise ValueError("No config file provided.")

    if not Path(args.config).exists():
        raise ValueError(
            f"Requested config JSON file could not be found: {args.config}"
        )

    if args.save_path is None:
        raise ValueError("No save path (trained model storage location) provided.")

    args.config_data = build_config(args.config)

    if args.data is None:
        raise ValueError(f"Error: no data files provided")
    # Data can be a file or a directory containing many .h5 files
    data_path = Path(args.data)
    if data_path.is_dir() and not data_path.glob("*.h5"):
        raise ValueError(f"Error: no data files provided")
    elif data_path.is_file() and not data_path.exists():
        raise ValueError(f"Error: no data files provided")

    args.model_dir = args.save_path
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        model_dir.mkdir(exist_ok=True, parents=True)


def run(args):
    data_dir = args.data
    model_dir = args.model_dir
    config_data = args.config_data
    trainer = UnsupervisedTrainer(data_dir, model_dir, config_data)
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    run(args)
