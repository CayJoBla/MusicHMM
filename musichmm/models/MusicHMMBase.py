import numpy as np
import pickle


class MusicHMMBase:
    """Abstract class to represent a Hidden Markov Model for music. This class is meant to be a parent
    class for different MusicHMM model types.

    Subclasses should implement the fit, sample, and sequence_to_song methods.
    """
    def __init__(self):
        self.initialized = False

    def generate_song(self, num_notes, currstate=None, **kwargs):
        """Sample a sequence from the trained HMM and convert it to a Song object.
        Subclasses should implement the `sample` and `sequence_to_song` methods.

        Parameters:
            num_notes (int): Number of notes to sample for the new song
            currstate (NoteState): Current state to start the sampling from

        Returns:
            (Song): A new Song object generated from the HMM
        """
        X = self.sample(num_notes, currstate)
        return self.sequence_to_song(X, **kwargs)

    def save(self, filename):
        """Save the MusicHMMBase object to a file"""
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def from_file(cls, filename):
        """Load a MusicHMMBase object from a file"""
        with open(filename, "rb") as file:
            return pickle.load(file)


def check_state_initialization(func):
    """A decorator function to check if the states have been initialized"""
    def wrapper(self, *args, **kwargs):
        if not self.initialized:
            raise ValueError("States have not been initialized yet. "
                                "Fit the model or call initialize_states() first.")
        return func(self, *args, **kwargs)
    return wrapper