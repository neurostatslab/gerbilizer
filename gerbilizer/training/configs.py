"""
File to configure all hyperparameters (model architecture and training).
"""
import json

from typing import NewType


JSON = NewType("JSON", dict)
DEFAULT_CONFIG = {
    "DEVICE": "CPU",
    "TORCH_SEED": 888,  # RNG initial state seeds
    "NUMPY_SEED": 777,
    "LOG_INTERVAL": 3,  # seconds between each addition to the training log
    "NUM_MICROPHONES": 4,  # not including xcorrs
    "COMPUTE_XCORRS": False,  # Append pair-wise cross-correlations of input audio. O(n^2) space requirement
    "ARENA_WIDTH": 558.8,  # Size of enclosure along x axis
    "ARENA_LENGTH": 355.6,  # Size of enclosure along y axis
    # Training hyperparameters.
    "NUM_EPOCHS": 20,
    "TRAIN_BATCH_SIZE": 64,
    "MAX_LEARNING_RATE": 1e-2,
    "MIN_LEARNING_RATE": 1e-6,
    "WEIGHT_DECAY": 1e-5,
    "CLIP_GRADIENTS": False,
    "OPTIMIZER": "SGD",
    "MOMENTUM": 0.5,
    "VAL_BATCH_SIZE": 64,
    "TEST_BATCH_SIZE": 64,
    # Architecture hyperparameters.
    "ARCHITECTURE": "GerbilizerSimpleNet",
    # Data Augmentations present during training.
    "AUGMENT_DATA": True,
    "AUGMENT_STRETCH_MIN": 0.95,
    "AUGMENT_STRETCH_MAX": 1.1,
    "AUGMENT_STRETCH_PROB": 1e-6,
    "AUGMENT_GAUSS_NOISE": 0.005,
    "AUGMENT_PITCH_MIN": -1.0,
    "AUGMENT_PITCH_MAX": 1.0,
    "AUGMENT_PITCH_PROB": 1e-6,
    "AUGMENT_SHIFT_MIN": -0.1,
    "AUGMENT_SHIFT_MAX": 0.1,
    "AUGMENT_SHIFT_PROB": 1.0,
    "AUGMENT_INVERSION_PROB": 0.5,
    # Label augmentations: involve mirroring sounds within the arena
    "AUGMENT_LABELS": True,
    "AUGMENT_FLIP_HORIZ": True,
    "AUGMENT_FLIP_VERT": True,
}


def keys_to_uppercase(dictionary: dict) -> dict:
    """Converts all the string keys in a dictionary to uppercase"""
    new_dict = dict()
    for key, value in dictionary.items():
        if isinstance(key, str):
            new_dict[key.upper()] = value
        else:
            new_dict[key] = value
    return new_dict


def build_config(filepath: str) -> JSON:
    with open(filepath, "r") as ctx:
        config = json.load(ctx)

    if "CONFIG_NAME" not in config:
        raise ValueError(
            "Configurations provided as JSON files should include a 'CONFIG_NAME' string."
        )

    for key, default_value in DEFAULT_CONFIG.items():
        if key not in config.keys():
            config[key] = default_value

    return keys_to_uppercase(config)
