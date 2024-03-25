import numpy as np

class SongDataset:
    """A basic class to hold a list of Song objects and provide some basic methods for working with them.
    This class also contains methods for converting the songs into state sequences for use in the MusicHMM models.

    Attributes:
        songs (list): A list of Song objects.
    """
    def __init__(self, songs):
        self.songs = songs

    def to_states(self):
        """Convert the list of songs into a list of state sequences for use in the MusicHMM models."""
        state_sequences = np.array([song.to_states() for song in self.songs])   # (song, part, state sequence)
        num_parts = len(state_sequences[0])
        return np.array([state_sequences[:,i] for i in range(num_parts)])       # (part, song, state sequence)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, i):
        return self.songs[i]