import numpy as np


class Discretized:
    """Hold the discretized pitches for a given part of music. Can access the discretized
    pitches at any time by indexing the object.
        
    Parameters:
        sequence (ndarray): Sequence of note states to discretize
    """
    def __init__(self, sequence):
        timestamps = np.cumsum([float(state.duration) for state in sequence]+[np.inf])
        pitch_sequence = np.array([state.pitch for state in sequence]+["REST"])
        mask = np.concatenate((pitch_sequence[:-1] != pitch_sequence[1:], [True]))

        self.timestamps = timestamps[mask]          # The timestamps at which the corresponding pitch changes
        self.pitch_sequence = pitch_sequence[mask]  # The pitches at the corresponding timestamps

    def __getitem__(self, t):
        return self.pitch_sequence[np.searchsorted(self.timestamps, t, side="right")]
