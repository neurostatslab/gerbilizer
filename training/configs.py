"""
File to configure all hyperparameters (model architecture and training).
"""
from collections import defaultdict
import numpy as np


def build_config(config_name, job_id):
	"""
	Returns dictionary of hyperparameters.
	"""
	
	# Use job_id to seed any random hyperparameters.
	rs = np.random.RandomState(job_id)

	# Specify default hyperparameters
	DEFAULT_CONFIG = {
		"JOB_ID": job_id,
		"DEVICE": "CPU",
		"TORCH_SEED": 888,
		"NUMPY_SEED": 777,
		"LOG_INTERVAL": 3, # seconds
		"NUM_MICROPHONES": 4,
		"NUM_SLEAP_KEYPOINTS": 3,
		"AUDIO_SAMPLE_RATE": 125000,

		# Training hyperparameters.
		"NUM_EPOCHS": 20,
		"TRAIN_BATCH_SIZE": 64,
		"MAX_LEARNING_RATE": 1e0,
		"MIN_LEARNING_RATE": 1e-4,
		"WEIGHT_DECAY": 1e-5,
		"MOMENTUM": 0.5,

		"VAL_BATCH_SIZE": 64,
		"TEST_BATCH_SIZE": 64,

		# Architecture hyperparameters.
		"ARCHITECTURE": "GerbilizerDenseNet",
		"USE_BATCH_NORM": False,

		"NUM_CHANNELS_LAYER_1": 11,
		"NUM_CHANNELS_LAYER_2": 10,
		"NUM_CHANNELS_LAYER_3": 10,
		"NUM_CHANNELS_LAYER_4": 10,
		"NUM_CHANNELS_LAYER_5": 10,
		"NUM_CHANNELS_LAYER_6": 10,
		"NUM_CHANNELS_LAYER_7": 10,
		"NUM_CHANNELS_LAYER_8": 10,
		"NUM_CHANNELS_LAYER_9": 10,
		"NUM_CHANNELS_LAYER_10": 10,
		"NUM_CHANNELS_LAYER_11": 10,
		"NUM_CHANNELS_LAYER_12": 10,

		# "FILTER_SIZE_LAYER_1": 49,
		# "FILTER_SIZE_LAYER_2": 45,
		# "FILTER_SIZE_LAYER_3": 41,
		# "FILTER_SIZE_LAYER_4": 37,
		# "FILTER_SIZE_LAYER_5": 33,
		# "FILTER_SIZE_LAYER_6": 29,
		# "FILTER_SIZE_LAYER_7": 25,
		# "FILTER_SIZE_LAYER_8": 21,
		# "FILTER_SIZE_LAYER_9": 17,
		# "FILTER_SIZE_LAYER_10": 13,
		# "FILTER_SIZE_LAYER_11": 9,
		# "FILTER_SIZE_LAYER_12": 5,

		"FILTER_SIZE_LAYER_1": 51,
		"FILTER_SIZE_LAYER_2": 51,
		"FILTER_SIZE_LAYER_3": 51,
		"FILTER_SIZE_LAYER_4": 51,
		"FILTER_SIZE_LAYER_5": 51,
		"FILTER_SIZE_LAYER_6": 51,
		"FILTER_SIZE_LAYER_7": 51,
		"FILTER_SIZE_LAYER_8": 51,
		"FILTER_SIZE_LAYER_9": 49,
		"FILTER_SIZE_LAYER_10": 37,
		"FILTER_SIZE_LAYER_11": 19,
		"FILTER_SIZE_LAYER_12": 9,

		"DILATION_LAYER_1": 1,
		"DILATION_LAYER_2": 1,
		"DILATION_LAYER_3": 1,
		"DILATION_LAYER_4": 1,
		"DILATION_LAYER_5": 1,
		"DILATION_LAYER_6": 1,
		"DILATION_LAYER_7": 1,
		"DILATION_LAYER_8": 1,
		"DILATION_LAYER_9": 1,
		"DILATION_LAYER_10": 1,
		"DILATION_LAYER_11": 1,
		"DILATION_LAYER_12": 1,

		# Data Augmentations present during training.
		"AUGMENT_DATA": True,
		"AUGMENT_FLIP_HORIZ": True,
		"AUGMENT_FLIP_VERT": True,
		"AUGMENT_STRETCH_MIN": 0.95,
		"AUGMENT_STRETCH_MAX": 1.1,
		"AUGMENT_STRETCH_PROB": 1e-6,
		"AUGMENT_GAUSS_MIN": 1e-5,
		"AUGMENT_GAUSS_MAX": 1e-3,
        "AUGMENT_GAUSS_PROB": 1e-6,
        "AUGMENT_PITCH_MIN": -1.0,
        "AUGMENT_PITCH_MAX": 1.0,
        "AUGMENT_PITCH_PROB": 1e-6,
        "AUGMENT_SHIFT_MIN": -0.2,
        "AUGMENT_SHIFT_MAX": 0.2,
        "AUGMENT_SHIFT_PROB": 1e-6,
	}

	# Specify custom CONFIG dicts below. Note that
	# any unspecified hyperparameters will default
	# to the values held in DEFAULT_CONFIG.
	if config_name == "default":
		CONFIG = dict()

	elif config_name == "alex":
		CONFIG = {
			"MAX_LEARNING_RATE": 10.0,
			"AUGMENT_DATA": False,
		}

	elif config_name == "sweep1":
		# Specify random learning rate, for example.
		CONFIG = {
			"MAX_LEARNING_RATE": 10 ** rs.uniform(-1, 0),
		}

	else:
		raise ValueError(
			f"'{config_name}' was not recognized as a "
			"valid 'config_name' parameter."
			)

	# Fill in any default values.
	for key, default_value in DEFAULT_CONFIG.items():
		if key not in CONFIG.keys():
			CONFIG[key] = default_value

	return CONFIG

