import numpy as np
from hmmlearn.hmm import CategoricalHMM
from musichmm.data.Song import Song

class SimpleMusicHMM:
    """Class to represent a Hidden Markov Model for music. This model only considers the
    first part of the music.

    Attributes:
        n_components (int): Number of components for the hidden states of the HMM
        hmm (CategoricalHMM): HMM model to train and sample from. Available after calling `fit()`
    """
    def __init__(self, n_components):
        """Initialize MusicHMM class with a given number of hidden components"""
        self.n_components = n_components
        
    def fit(self, data):
        """Train the HMM on the given dataset of songs
        
        Parameters:
            songs (SongDataset): A SongDataset object containing songs to train on
        """
        # Get observations by index and initialize unique states
        sequence, lengths = self._initialize_states(data)
        sequence = sequence.reshape(-1,1)
        print(sequence)
        
        # Train an HMM for each part
        self.hmm = CategoricalHMM(n_components=self.n_components).fit(sequence, lengths=lengths)

        return self
        
    def _initialize_states(self, dataset):
        """Returns the concatenated dataset, as a sequence of indices that map to note states. Also 
        initializes the `states_`, `state_to_idx_`, and `n_` hidden attributes. Called internally when fitting.

        Parameters:
            dataset (SongDataset): A dataset of Song objects to train on

        Returns:
            (tuple(ndarray(int),ndarray(int))): Tuple containing a note state index sequence concatenated
                from all songs in the dataset, and an array of sequence lengths for each song.
        """
        part_sequences = dataset.to_states()    # (part, song, state sequence)
        part = part_sequences[0]                # (song, state sequence); only consider the first part

        lengths = np.array([len(song) for song in part])
        unique_states, sequence = np.unique(np.concatenate(part), return_inverse=True)

        self.states_ = unique_states
        self.state_to_idx_ = {state:i for i, state in enumerate(unique_states)}
        self.n_ = 1
        
        return sequence, lengths
               
    def gen_song(self, num_notes=30, currstate=None):
        """Sample a new song from the HMM.
        
        Parameters:
            num_notes (int): Number of notes to sample for the new song
            currstate (int): Current state to start the sampling from

        Returns:
            (Song): A new Song object generated from the HMM
        """
        _, Z = self.hmm.sample(num_notes, currstate=currstate)
        return Song.from_sequences([self.states_[Z]])