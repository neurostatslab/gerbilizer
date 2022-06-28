"""Utility functions to help set up pyroomacoustics with MUSE."""
from collections import namedtuple
from collections.abc import Sequence
from typing import Iterable, List, Mapping, Tuple

import librosa
import numpy as np
import pyroomacoustics as pra

from pyroomacoustics.directivities import CardioidFamily, DirectionVector, DirectivityPattern

from config import X_DIM, Y_DIM, Z_DIM, MIC_POS

def generate_room(
    x_dim: float,
    y_dim: float,
    z_dim: float,
    wall_materials: Mapping[str, str] = None,
    sampling_rate=125000
) -> pra.Room:
    """
    Return a shoebox room of the provided dimensions using pyroomacoustics.

    This Shoebox is the basis for the example. After generating it, we'll place mics in it
    and play a sound stimulus. Pyroomacoustics will them simulate the signals that the
    mic would recieve based on their locations and the location of the sound stimulus.
    """
    # Materials of the room's walls
    # See different materials at https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html
    if not wall_materials:
        materials = pra.make_materials(
            ceiling = "fibre_absorber_2",
            floor   = "fibre_absorber_2",
            east    = "hard_surface",
            west    = "hard_surface",
            north   = "hard_surface",
            south   = "hard_surface"
        )
    else:
        materials = pra.make_materials(**wall_materials)
    
    shoebox_object = pra.ShoeBox(
        [x_dim, y_dim, z_dim],
        fs=sampling_rate,
        materials=materials,
        max_order=9  # Maximum number of times a sound wave can reflect
    ) 
    return shoebox_object

def place_mics(shoebox: pra.Room, mic_pos: np.ndarray):
    """
    Place mics at the specified positions in the given Shoebox, oriented towards
    the center of the room's floor.
    
    Args:
        shoebox: The room into which the mics will be placed.
        mic_pos: array of shape (n_mics, 3) storing the Cartesian coordinates
            at which to place the mics.
    """
    shoebox_dims = shoebox.shoebox_dim
    if any([coord >= box_size for coord, box_size in zip(mic_pos.max(axis=0), shoebox_dims)]) \
        or any([coord <= 0 for coord in mic_pos.min(axis=0)]):
        raise ValueError("The microphones must be within the box. They cannot be located along an egde or face.")
    
    # Compute directivity of microphones
    # Assumes the microphones are all pointed toward the center of the room's floor
    # Can be simplified by just pointing them downward
    room_center = np.array(shoebox_dims) / 2
    room_center[2] = 0
    
    mic_directions = room_center[np.newaxis, ...] - mic_pos  # List of vectors pointing from each mic to room_center
    mic_directions /= np.sqrt((mic_directions**2).sum(axis=1, keepdims=True))  # Normalize the vectors
    
    # Convert the direction vectors into polar coordinates for the directivities used by PRA
    directivities = list()
    for direction in mic_directions:
        # Angle within the x-y plane. 0 radians is toward the positive x direction
        # In spherical coordinates, this is \theta
        azimuth = np.arctan2(direction[1], direction[0])
        # Same thing here, but between the z axis and the x-y plane
        xy_vec_size = np.sqrt((direction[:2] ** 2).sum())
        # Using arctan instead of arctan2 because I need the result to be within Q1 and Q4
        # This would be called \phi in spherical coordinates
        colatitude = np.pi/2 - np.arctan(direction[2] / xy_vec_size)
        pra_direction = DirectionVector(
            azimuth=azimuth,
            colatitude=colatitude,
            degrees=False
        )
        # For visualizations, see https://en.wikipedia.org/wiki/Microphone#Polar_patterns
        # For supported options, see https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.directivities.html?highlight=DirectivityPattern#pyroomacoustics.directivities.DirectivityPattern
        directivities.append(CardioidFamily(
            orientation=pra_direction,
            pattern_enum=DirectivityPattern.HYPERCARDIOID
        ))
    
    # transpose because shoebox expects positions in shape (3, n_mics)
    shoebox.add_microphone_array(mic_array=mic_pos.T, directivity=directivities)
    
def add_source(
    speaker_pos: Sequence,
    stimulus: np.ndarray,
    sr_stimulus: int,
    room: pra.Room
    ):
    """
    Add an audio source playing the given stimulus at location speaker_pos.
    
    If the sample rate of the stimulus doesn't match the sampling rate used in the
    room for simulation, resample audio to match.
    """
    if not isinstance(speaker_pos, list):
        speaker_pos = list(speaker_pos)
    
    if room.fs != sr_stimulus:
        stimulus_resampled = librosa.resample(
            stimulus.astype('float'),
            orig_sr=sr_stimulus,
            target_sr=room.fs
            )
        room.add_source(speaker_pos, signal=stimulus_resampled, delay=0)
        return room.fs, stimulus_resampled
        
    else:
        room.add_source(speaker_pos, signal=stimulus, delay=0)
        return sr_stimulus, stimulus

# def arbitrary_setup() -> pra.Room:
#     """
#     Generate arbitrary room setup. speaker_pos should be a 2-tuple storing
#     floats from 0 to 1 storing the desired x and y coordinates of the speaker
#     in the room, as a fraction of the room dimension.

#     To get audio playback from the returned room, access the room's sound source
#     list, and add your desired stimulus to the source `signal` attribute.
#     """
#     room = generate_room(X_DIM, Y_DIM, Z_DIM)
#     place_mics(room, MIC_POS)
#     return room

SimulatedSignal = namedtuple('SimulatedSignal', [
    'signal', 'sample_rate', 'speaker_position'
    ])

def simulate_signals(
    stimuli: Iterable[np.ndarray],
    sample_rates: Iterable[float],
    speaker_position: Tuple[float],
    wall_materials: Mapping[str, str] = None,
    ) -> List[Tuple]:
    """
    Given iterables of stimuli, their associated sample rates, and
    the desired location from which to simulate playing the stimuli,
    return a list of simulated signals.
    """
    signals = []
    for stimulus, sr in zip(stimuli, sample_rates):
        room = generate_room(X_DIM, Y_DIM, Z_DIM, wall_materials=wall_materials)
        place_mics(room, MIC_POS)
        # add the given stimulus to the room at the specified position
        add_source(speaker_position, stimulus, sr, room)
        room.simulate()
        result = SimulatedSignal(
            room.mic_array.signals,
            sr,
            speaker_position
        )
        signals.append(result)
    return signals

def generate_dataset(
    stimuli: Iterable[np.ndarray],
    sample_rates: Iterable[float],
    speaker_spacing: float,
    wall_materials: Mapping[str, str] = None,
    speaker_height=0.01,
    ):
    """
    Generate a dataset of simulated microphone data from the given
    stimuli.

    The microphone data is simulated in a room of dimensions specified
    in config.py, and each stimulus is simulated from a grid of locations
    on the arena floor. The spacing of those gridpoints is determined by
    the `speaker_spacing` parameter.
    """
    # set up points at which to simulate audio sources
    num_pts = int(1 / speaker_spacing)
    # add some offset so the points aren't directly on the pyroom wall
    offset = 0.1
    xcoords = np.linspace(offset, X_DIM-offset, num_pts)
    ycoords = np.linspace(offset, Y_DIM-offset, num_pts)

    results = []
    for x in xcoords:
        for y in ycoords:
            results += simulate_signals(
                stimuli, sample_rates, (x, y, speaker_height),
                wall_materials=wall_materials
                )
    return results