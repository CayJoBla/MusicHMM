import numpy as np
from hmmlearn.hmm import CategoricalHMM
from musichmm.data.Song import Song
from musichmm.models.MusicHMMBase import MusicHMMBase, check_state_initialization
import warnings


class SimpleMusicHMM(MusicHMMBase):
    """Class to represent a Hidden Markov Model for music. This model only considers the
    first part of the music and uses a CategoricalHMM model representation.

    Attributes:
        hmm (CategoricalHMM): HMM model to train and sample from.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the SimpleMusicHMM class with the underlying CategoricalHMM"""
        super().__init__()
        self.hmm = CategoricalHMM(*args, **kwargs)

    def fit(self, dataset):
        """Train the HMM on the given dataset of songs. 
        
        Parameters:
            songs (SongDataset): A SongDataset object containing songs to train on
        """
        # Get sequences and sequence lengths to pass to hmm.fit()
        sequences, lengths = self.initialize_states(dataset) 
        self.hmm.fit(sequences, lengths=lengths)

        return self
        
    def initialize_states(self, dataset):
        """Returns the concatenated dataset as a sequence of indices that map to note states. 
        Called internally when fitting. Outputs can be passed directly to the hmm.fit() method

        Parameters:
            dataset (SongDataset): A dataset of Song objects to train on

        Returns:
            ndarray(int): A 2d sequence of state indices concatenated from all songs in the dataset
            ndarray(int): An array of sequence lengths for each song in the dataset
        """
        self.initialized = False    # Reset the initialized flag in case state initialization fails

        # Get NoteState sequences (only the first part)
        part_sequences = dataset.to_states()        # (part, song, state sequence)
        if len(part_sequences) > 1:
            warnings.warn("SimpleMusicHMM only supports training on Songs with a single part." 
                          "Only the first part of each song will be used.")
        part = part_sequences[0]                    # (song, state sequence); only consider the first part
        self.n_parts = 1

        # Extract the song lengths, and convert to a single sequence of state indices
        lengths = np.array([len(song) for song in part])
        unique_states, sequences = np.unique(np.concatenate(part), return_inverse=True)
        sequences = sequences.reshape(-1,1)

        # Store the unique states
        self.states = unique_states

        self.initialized = True     # Successfully initialized states
        
        return sequences, lengths

    def sample(self, num_notes, currstate=None):
        currstate = self.state_to_idx(currstate) if currstate is not None else currstate
        return self.hmm.sample(num_notes, currstate=currstate)[0]

    @check_state_initialization
    def state_to_idx(self, states):
        return np.searchsorted(self.states, states)

    @check_state_initialization
    def idx_to_state(self, indices):
        return self.states[indices.flatten()]
               
    def sequence_to_song(self, X):
        return Song.from_sequences([self.idx_to_state(X)])
