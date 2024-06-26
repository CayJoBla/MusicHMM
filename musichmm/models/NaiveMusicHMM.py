import numpy as np
from hmmlearn.hmm import CategoricalHMM
from musichmm.data.Song import Song
from musichmm.models.MusicHMMBase import MusicHMMBase, check_state_initialization
import warnings

class NaiveMusicHMM(MusicHMMBase):
    """Class to represent a Hidden Markov Model for music. This model assumes that each 
    part of the music is independent of the others.

    Attributes:
        hmm (list(CategoricalHMM)): List of HMM models to train and sample from.
    """
    def __init__(self, n_parts, *args, **kwargs):
        """Initialize the NaiveMusicHMM class with the underlying CategoricalHMM models"""
        super().__init__()
        self.hmm = [CategoricalHMM(*args, **kwargs) for _ in range(n_parts)]
        self.n_parts = n_parts
        
    def fit(self, dataset):
        """Train the HMM on the given dataset of songs
        
        Parameters:
            songs (SongDataset): A SongDataset object containing songs to train on
        """
        # Get sequences and sequence lengths to pass to hmm.fit()
        sequences, lengths = self.initialize_states(dataset) 
        
        # Train an HMM for each part
        for i, hmm in enumerate(self.hmm):
            hmm.fit(sequences[i], lengths=lengths[i])

        return self
        
    def initialize_states(self, dataset):
        """Returns the concatenated dataset, divided by part, as a sequence of indices that map to 
        note states. Called internally when fitting.

        Parameters:
            dataset (SongDataset): A dataset of Song objects to train on

        Returns:
            list(ndarray(int)): A list of 2d sequences of state indices concatenated from all songs in the dataset. 
                                Each element in the list corresponds to a part.
            list(ndarray(int)): A list of arrays of sequence lengths for each song in the dataset. Each element in
                                the list corresponds to a part.
        """
        self.initialized = False    # Reset the initialized flag in case state initialization fails

        part_sequences = dataset.to_states()    # (part, song, state sequence)
        if len(part_sequences) > self.n_parts:
            warning.warn(f"NaiveMusicHMM was initialized with {self.n_parts} parts, but the dataset has "
                         f"{len(part_sequences)} parts. Only the first {self.n_parts} parts will be used.")
        elif len(part_sequences) < self.n_parts:
            raise ValueError(f"NaiveMusicHMM was initialized with {self.n_parts} parts, but the dataset has "
                             f"{len(part_sequences)} parts. Please provide a dataset containing songs with at "
                             f"least {self.n_parts} parts.")

        # Extract the song lengths, and convert to a single sequence of state indices
        lengths = [np.array([len(song) for song in part]) for part in part_sequences]
        states_list = [np.unique(np.concatenate(part), return_inverse=True) for part in part_sequences]
        unique_states, sequences = map(list, zip(*states_list))
        sequences = [seq.reshape(-1,1) for seq in sequences]    # Reshape to 2D

        # Store the unique states
        self.states = unique_states

        self.initialized = True     # Successfully initialized states
        
        return sequences, lengths

    def sample(self, num_notes, currstate=None):
        if currstate is not None:
            raise NotImplementedError("Specifying a current state has not been implemented for NaiveMusicHMM sampling")
        return [hmm.sample(num_notes, currstate=None)[0] for hmm in self.hmm]

    @check_state_initialization
    def state_to_idx(self, states): # list(list(NoteState)) - (part, states)
        return [np.searchsorted(self.states[i], states[i]) for i in range(self.n_parts)]

    @check_state_initialization
    def idx_to_state(self, indices):
        return [self.states[i][indices[i].flatten()] for i in range(self.n_parts)]
               
    def sequence_to_song(self, X, truncation=False, padding=False):
        """Return a Song object from a sequence of states. 
        
        If truncate is True, the song will be truncated to the duration of the shortest part in the sequence.
        If padding is True, the parts will be padded with rests to match the duration of the longest part.
        """
        state_sequence = self.idx_to_state(X)

        if truncation:  # Truncate the song to the duration of the shortest part
            cum_durations = [np.cumsum([state.duration for state in part]) for part in state_sequence]
            min_duration = min([part[-1] for part in cum_durations])
            indices = [np.searchsorted(part_durations, min_duration)+1 for part_durations in cum_durations]
            state_sequence = [part[:ind] for part, ind in zip(state_sequence, indices)]

        if padding:     # Pad the parts with rests to match the duration of the longest part
            durations = np.array([sum([state.duration for state in part]) for part in state_sequence])
            duration_diff = max(durations) - durations  # The rest of 0 duration will be dropped by the Song class
            state_sequence = [part + [NoteState("REST", duration)] for part, duration in zip(state_sequence, duration_diff)]

        return Song.from_sequences(state_sequence)
               
    