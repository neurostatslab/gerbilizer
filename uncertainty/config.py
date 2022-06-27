import numpy as np

SIGMA = 0.5  # std for gaussian smoothing

X_DIM = 1.83
Y_DIM = 2.44
Z_DIM = 0.76

offset = 0.1

# place mics in top corners of the room
MIC_POS = np.array([
        [X_DIM-offset, Y_DIM-offset, Z_DIM-offset],
        [offset, offset, Z_DIM-offset],
        [offset, Y_DIM-offset, Z_DIM-offset],
        [X_DIM-offset, offset, Z_DIM-offset]
    ])

F_LO = 0
F_HI = 125000
TEMP = 20

RESOLUTION = 0.025