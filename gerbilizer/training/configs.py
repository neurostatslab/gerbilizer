"""
File to configure all hyperparameters (model architecture and training).
"""
from sys import stderr
from typing import NewType

try:
    # Attempt to use json5 if available
    import pyjson5 as json
except ImportError:
    print("Warning: json5 not available, falling back to json.", file=stderr)
    import json


JSON = NewType("JSON", dict)
DEFAULT_CONFIG = {
    "OPTIMIZATION": {
        "NUM_EPOCHS": 30,
        "OPTIMIZER": "SGD",
        "MOMENTUM": 0.9,
        "CLIP_GRADIENTS": False,
        "INITIAL_LEARNING_RATE": 0.0005,
        "SCHEDULERS": [
            {"SCHEDULER_TYPE": "COSINE_ANNEALING", "MIN_LEARNING_RATE": 0.0}
        ],
    },
    "ARCHITECTURE": "GerbilizerSimpleNetwork",
    "GENERAL": {
        "CONFIG_NAME": "simple_network",
        "DEVICE": "GPU",  # 'GPU' or 'CPU'
        "TORCH_SEED": 888,  # rng seeds for reproducibility
        "NUMPY_SEED": 777,
        "LOG_INTERVAL": 3,  # Amount of time between consecutive log messages
    },
    "DATA": {
        "NUM_MICROPHONES": 4,
        "SAMPLE_RATE": 125000,
        "COMPUTE_XCORRS": False,
        "BATCH_SIZE": 32,
        "CROP_LENGTH": 8192,
        "AUGMENT_DATA": True,
        "ARENA_DIMS": [558.9, 355.6],
    },
    "AUGMENTATIONS": {
        # Data augmentations: involves performing augmentations to the audio to which the model should be invariant
        "INVERSION": {
            "PROB": 0.5,
        },
        "NOISE": {
            "MIN_SNR": 0,
            "MAX_SNR": 10,
            "PROB": 0.5,
        },
        "MASK": {
            "PROB": 0.5,
            "MIN_LENGTH": 75,  # 0.6 ms at 125 kHz
            "MAX_LENGTH": 125,  # 1 ms at 125 kHz
        },
    },
}


def keys_to_uppercase(dictionary: dict) -> dict:
    """Converts all the string keys in a dictionary to uppercase for consistency"""
    new_dict = dict()
    for key, value in dictionary.items():
        if isinstance(value, dict):  # recurse through subdictionaries if present
            value = keys_to_uppercase(value)
        if isinstance(key, str):
            new_dict[key.upper()] = value
        else:
            new_dict[key] = value
    return new_dict


def update_recursively(dictionary: dict, defaults: dict) -> dict:
    """Updates a dictionary with default values, recursing through subdictionaries"""
    for key, default_value in defaults.items():
        if key not in dictionary:
            dictionary[key] = default_value
        elif isinstance(dictionary[key], dict):
            dictionary[key] = update_recursively(dictionary[key], default_value)
    return dictionary


def build_config(filepath: str) -> JSON:
    with open(filepath, "r") as ctx:
        try:
            config = json.load(ctx)
        except:
            raise ValueError(
                f"Could not parse JSON file at {filepath}. Perhaps a JSON5 file was provided without the necessary libraries installed?"
            )

    if "CONFIG_NAME" not in config["GENERAL"]:
        raise ValueError(
            "Configurations provided as JSON files should include a 'CONFIG_NAME' string."
        )

    config = update_recursively(config, DEFAULT_CONFIG)
    return keys_to_uppercase(config)
