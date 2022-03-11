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
        "NUM_MICROPHONES": 10,  # including xcorrs
        "NUM_SLEAP_KEYPOINTS": 3,
        "NUM_AUDIO_SAMPLES": 10000,
        "AUDIO_SAMPLE_RATE": 125000,

        # Training hyperparameters.
        "NUM_EPOCHS": 20,
        "TRAIN_BATCH_SIZE": 64,
        "MAX_LEARNING_RATE": 1e-2,
        "MIN_LEARNING_RATE": 1e-6,
        "WEIGHT_DECAY": 1e-5,
        "MOMENTUM": 0.5,

        "INPUT_SCALE_FACTOR": 1.0,
        "OUTPUT_SCALE_FACTOR": 1.0,

        "VAL_BATCH_SIZE": 64,
        "TEST_BATCH_SIZE": 64,

        # Architecture hyperparameters.
        "ARCHITECTURE": "GerbilizerDenseNet",
        "USE_BATCH_NORM": False,

        "POOLING": "AVG",

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
        "AUGMENT_GAUSS_NOISE": 0.005,
        "AUGMENT_PITCH_MIN": -1.0,
        "AUGMENT_PITCH_MAX": 1.0,
        "AUGMENT_PITCH_PROB": 1e-6,
        "AUGMENT_SHIFT_MIN": -0.1,
        "AUGMENT_SHIFT_MAX": 0.1,
        "AUGMENT_SHIFT_PROB": 1.0,
        "AUGMENT_INVERSION_PROB": 0.5,
    }

    # Specify custom CONFIG dicts below. Note that
    # any unspecified hyperparameters will default
    # to the values held in DEFAULT_CONFIG.
    if config_name == "default":
        CONFIG = dict()

    elif config_name == "alex":
        CONFIG = {
            "NUM_EPOCHS": 100,
            "ARCHITECTURE": "GerbilizerSimpleNetwork",
            "MAX_LEARNING_RATE": 1e-1,
            "MIN_LEARNING_RATE": 1e-4,
            "MOMENTUM": 0.9,
            "DEVICE": "GPU",
            "AUGMENT_DATA": True,

            "NUM_CHANNELS_LAYER_1": 16,
            "NUM_CHANNELS_LAYER_2": 16,
            "NUM_CHANNELS_LAYER_3": 16,
            "NUM_CHANNELS_LAYER_4": 32,
            "NUM_CHANNELS_LAYER_5": 32,
            "NUM_CHANNELS_LAYER_6": 32,
            "NUM_CHANNELS_LAYER_7": 64,
            "NUM_CHANNELS_LAYER_8": 64,
            "NUM_CHANNELS_LAYER_9": 64,
            "NUM_CHANNELS_LAYER_10": 128,
            "NUM_CHANNELS_LAYER_11": 128,
            "NUM_CHANNELS_LAYER_12": 128,

            "FILTER_SIZE_LAYER_1": 11,
            "FILTER_SIZE_LAYER_2": 11,
            "FILTER_SIZE_LAYER_3": 11,
            "FILTER_SIZE_LAYER_4": 7,
            "FILTER_SIZE_LAYER_5": 7,
            "FILTER_SIZE_LAYER_6": 7,
            "FILTER_SIZE_LAYER_7": 7,
            "FILTER_SIZE_LAYER_8": 7,
            "FILTER_SIZE_LAYER_9": 7,
            "FILTER_SIZE_LAYER_10": 7,
            "FILTER_SIZE_LAYER_11": 7,
            "FILTER_SIZE_LAYER_12": 7,

            "DILATION_LAYER_1": 2,
            "DILATION_LAYER_2": 2,
            "DILATION_LAYER_3": 2,
            "DILATION_LAYER_4": 2,
            "DILATION_LAYER_5": 2,
            "DILATION_LAYER_6": 2,
            "DILATION_LAYER_7": 2,
            "DILATION_LAYER_8": 2,
            "DILATION_LAYER_9": 2,
            "DILATION_LAYER_10": 2,
            "DILATION_LAYER_11": 2,
            "DILATION_LAYER_12": 2,
        }

    elif config_name == "sweep1":
        # Specify random learning rate, for example.
        CONFIG = {
            "MAX_LEARNING_RATE": 10 ** rs.uniform(-1, 0),
        }
    elif config_name == "aramis_hourglass":
        CONFIG = {
            'NUM_MICROPHONES': 4,
            'NUM_CONV_LAYERS': 5,
            'USE_BATCH_NORM': True,
            # 'DEVICE': 'GPU',
            'ARCHITECTURE': 'GerbilizerHourglassNet',
            'NUM_EPOCHS': 100,
            'TRAIN_BATCH_SIZE': 8,
            'SINKHORN_EPSILON': 1e-2,
            'SINKHORN_MAX_ITER': 60,
            
            'NUM_CHANNELS_LAYER_1': 4,
            'NUM_CHANNELS_LAYER_2': 16,
            'NUM_CHANNELS_LAYER_3': 64,
            'NUM_CHANNELS_LAYER_4': 256,
            'NUM_CHANNELS_LAYER_5': 1024,
            
            'STRIDE_LAYER_1': 1,
            'STRIDE_LAYER_2': 2,
            'STRIDE_LAYER_3': 2,
            'STRIDE_LAYER_4': 2,
            'STRIDE_LAYER_5': 4,
            
            'FILTER_SIZE_LAYER_1': 256,
            'FILTER_SIZE_LAYER_2': 128,
            'FILTER_SIZE_LAYER_3': 64,
            'FILTER_SIZE_LAYER_4': 32,
            'FILTER_SIZE_LAYER_5': 16,
            
            'DILATION_LAYER_1': 1,
            'DILATION_LAYER_2': 2,
            'DILATION_LAYER_3': 4,
            'DILATION_LAYER_4': 8,
            'DILATION_LAYER_5': 16,
            
            
            'RESIZE_TO_N_CHANNELS': 16,

            
            'NUM_TCONV_LAYERS': 5,
            
            'TCONV_CHANNELS_LAYER_1': 16,
            'TCONV_CHANNELS_LAYER_2': 8,
            'TCONV_CHANNELS_LAYER_3': 4,
            'TCONV_CHANNELS_LAYER_4': 2,
            'TCONV_CHANNELS_LAYER_5': 1,
            'TCONV_CHANNELS_LAYER_6': 0,
            
            'TCONV_FILTER_SIZE_LAYER_1': 4,
            'TCONV_FILTER_SIZE_LAYER_2': 4,
            'TCONV_FILTER_SIZE_LAYER_3': 4,
            'TCONV_FILTER_SIZE_LAYER_4': 4,
            'TCONV_FILTER_SIZE_LAYER_5': 4,
            'TCONV_FILTER_SIZE_LAYER_6': 0,
            
            'TCONV_STRIDE_LAYER_1': 2,
            'TCONV_STRIDE_LAYER_2': 2,
            'TCONV_STRIDE_LAYER_3': 2,
            'TCONV_STRIDE_LAYER_4': 2,
            'TCONV_STRIDE_LAYER_5': 2,
            'TCONV_STRIDE_LAYER_6': 0,
            
            'TCONV_DILATION_LAYER_1': 1,
            'TCONV_DILATION_LAYER_2': 1,
            'TCONV_DILATION_LAYER_3': 1,
            'TCONV_DILATION_LAYER_4': 1,
            'TCONV_DILATION_LAYER_5': 1,
            'TCONV_DILATION_LAYER_6': 0
        }

    elif config_name == "ralph":
        CONFIG = {
            'NUM_MICROPHONES': 10, #hilbert transforms + xcorrs
            'NUM_CONV_LAYERS': 5,
            'USE_BATCH_NORM': True,
            'DEVICE': 'GPU',
            'ARCHITECTURE': 'GerbilizerHourglassNet',
            'NUM_EPOCHS': 100,
            "AUGMENT_DATA": False,
            'TRAIN_BATCH_SIZE': 8,
            'SINKHORN_EPSILON': 1e-2,
            'SINKHORN_MAX_ITER': 60,
            "MAX_LEARNING_RATE": 1e-1,
            "MIN_LEARNING_RATE": 1e-4,
            "MOMENTUM": 0.9,
            
            'NUM_CHANNELS_LAYER_1': 4,
            'NUM_CHANNELS_LAYER_2': 16,
            'NUM_CHANNELS_LAYER_3': 64,
            'NUM_CHANNELS_LAYER_4': 256,
            'NUM_CHANNELS_LAYER_5': 1024,
            
            'STRIDE_LAYER_1': 1,
            'STRIDE_LAYER_2': 2,
            'STRIDE_LAYER_3': 2,
            'STRIDE_LAYER_4': 2,
            'STRIDE_LAYER_5': 4,
            
            'FILTER_SIZE_LAYER_1': 256,
            'FILTER_SIZE_LAYER_2': 128,
            'FILTER_SIZE_LAYER_3': 64,
            'FILTER_SIZE_LAYER_4': 32,
            'FILTER_SIZE_LAYER_5': 16,
            
            'DILATION_LAYER_1': 1,
            'DILATION_LAYER_2': 2,
            'DILATION_LAYER_3': 4,
            'DILATION_LAYER_4': 8,
            'DILATION_LAYER_5': 16,
            
            
            'RESIZE_TO_N_CHANNELS': 16,

            
            'NUM_TCONV_LAYERS': 5,
            
            'TCONV_CHANNELS_LAYER_1': 16,
            'TCONV_CHANNELS_LAYER_2': 8,
            'TCONV_CHANNELS_LAYER_3': 4,
            'TCONV_CHANNELS_LAYER_4': 2,
            'TCONV_CHANNELS_LAYER_5': 1,
            'TCONV_CHANNELS_LAYER_6': 0,
            
            'TCONV_FILTER_SIZE_LAYER_1': 4,
            'TCONV_FILTER_SIZE_LAYER_2': 4,
            'TCONV_FILTER_SIZE_LAYER_3': 4,
            'TCONV_FILTER_SIZE_LAYER_4': 4,
            'TCONV_FILTER_SIZE_LAYER_5': 4,
            'TCONV_FILTER_SIZE_LAYER_6': 0,
            
            'TCONV_STRIDE_LAYER_1': 2,
            'TCONV_STRIDE_LAYER_2': 2,
            'TCONV_STRIDE_LAYER_3': 2,
            'TCONV_STRIDE_LAYER_4': 2,
            'TCONV_STRIDE_LAYER_5': 2,
            'TCONV_STRIDE_LAYER_6': 0,
            
            'TCONV_DILATION_LAYER_1': 1,
            'TCONV_DILATION_LAYER_2': 1,
            'TCONV_DILATION_LAYER_3': 1,
            'TCONV_DILATION_LAYER_4': 1,
            'TCONV_DILATION_LAYER_5': 1,
            'TCONV_DILATION_LAYER_6': 0
        }

    elif config_name == "ralph_simple":
        CONFIG = {
            "NUM_EPOCHS": 100,
            "ARCHITECTURE": "GerbilizerSimpleNetwork",
            "MAX_LEARNING_RATE": 1e-1,
            "MIN_LEARNING_RATE": 1e-4,
            "MOMENTUM": 0.9,
            "DEVICE": "GPU",
            "AUGMENT_DATA": False,

            "NUM_CHANNELS_LAYER_1": 16,
            "NUM_CHANNELS_LAYER_2": 16,
            "NUM_CHANNELS_LAYER_3": 16,
            "NUM_CHANNELS_LAYER_4": 32,
            "NUM_CHANNELS_LAYER_5": 32,
            "NUM_CHANNELS_LAYER_6": 32,
            "NUM_CHANNELS_LAYER_7": 64,
            "NUM_CHANNELS_LAYER_8": 64,
            "NUM_CHANNELS_LAYER_9": 64,
            "NUM_CHANNELS_LAYER_10": 128,
            "NUM_CHANNELS_LAYER_11": 128,
            "NUM_CHANNELS_LAYER_12": 128,

            "FILTER_SIZE_LAYER_1": 11,
            "FILTER_SIZE_LAYER_2": 11,
            "FILTER_SIZE_LAYER_3": 11,
            "FILTER_SIZE_LAYER_4": 7,
            "FILTER_SIZE_LAYER_5": 7,
            "FILTER_SIZE_LAYER_6": 7,
            "FILTER_SIZE_LAYER_7": 7,
            "FILTER_SIZE_LAYER_8": 7,
            "FILTER_SIZE_LAYER_9": 7,
            "FILTER_SIZE_LAYER_10": 7,
            "FILTER_SIZE_LAYER_11": 7,
            "FILTER_SIZE_LAYER_12": 7,

            "DILATION_LAYER_1": 2,
            "DILATION_LAYER_2": 2,
            "DILATION_LAYER_3": 2,
            "DILATION_LAYER_4": 2,
            "DILATION_LAYER_5": 2,
            "DILATION_LAYER_6": 2,
            "DILATION_LAYER_7": 2,
            "DILATION_LAYER_8": 2,
            "DILATION_LAYER_9": 2,
            "DILATION_LAYER_10": 2,
            "DILATION_LAYER_11": 2,
            "DILATION_LAYER_12": 2,
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
