import numpy as np
from hmmlearn.hmm import CategoricalHMM
from musichmm.data.Song import Song
from musichmm.data.NoteState import NoteState
from musichmm.models.MusicHMMBase import MusicHMMBase, check_state_initialization
import warnings

class NaiveIndependentMusicHMM(MusicHMMBase):
    """Class to represent a Hidden Markov Model for music. This model assumes that each 
    part of the music is independent of the others, and that the pitches are independent
    of their respective note durations

    Attributes:
        hmm (list(CategoricalHMM)): List of HMM models to train and sample from.
    """
    def __init__(self, n_parts, *args, **kwargs):
        """Initialize the NaiveMusicHMM class with the underlying CategoricalHMM models"""
        super().__init__()
        self.hmm = {
            "pitch": [CategoricalHMM(*args, **kwargs) for _ in range(n_parts)],
            "duration": [CategoricalHMM(*args, **kwargs) for _ in range(n_parts)]
        }
        self.n_parts = n_parts
        
    def fit(self, dataset):
        """Train the HMM on the given dataset of songs
        
        Parameters:
            songs (SongDataset): A SongDataset object containing songs to train on
        """
        # Get sequences and sequence lengths to pass to hmm.fit()
        sequences, lengths = self.initialize_states(dataset) 
        
        # Train an HMM for each part
        for i, pitch_hmm in enumerate(self.hmm["pitch"]):
            pitch_hmm.fit(sequences[i][:, np.newaxis, 0], lengths=lengths[i])
        for i, duration_hmm in enumerate(self.hmm["duration"]):
            duration_hmm.fit(sequences[i][:, np.newaxis, 1], lengths=lengths[i])

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

        # Convert the state sequences to indices
        lengths = []
        sequences = []
        unique_pitches = []
        unique_durations = []
        for part in part_sequences:
            lengths.append(np.array([len(song) for song in part]))                              # Get song lengths
            state_sequence = np.array([state.as_tuple() for song in part for state in song])    # Sequence of (pitch, duration)
            pitches, pitch_sequences = np.unique(state_sequence[:,0], return_inverse=True)      # Get unique pitches and encode as indices
            unique_pitches.append(pitches)
            durations, duration_sequences = np.unique(state_sequence[:,1], return_inverse=True) # Get unique durations and encode as indices
            unique_durations.append(durations)
            sequences.append(np.vstack((pitch_sequences, duration_sequences)).T)                # Sequence of (pitch index, duration index)

        # Store the unique states and a define mappings between states to indices
        self.pitches = unique_pitches
        self.durations = unique_durations

        self.initialized = True     # Successfully initialized states
        
        return sequences, lengths

    def sample(self, num_notes, currstate=None):
        if currstate is not None:
            raise NotImplementedError("Specifying a current state has not been implemented for NaiveIndependentMusicHMM sampling")
        X_pitch = [hmm.sample(num_notes, currstate=None)[0] for hmm in self.hmm["pitch"]]
        X_duration = [hmm.sample(num_notes, currstate=None)[0] for hmm in self.hmm["duration"]]
        return [np.hstack((pitch, duration)) for pitch, duration in zip(X_pitch, X_duration)]

    @check_state_initialization
    def state_to_idx(self, states):
        """Convert a list of lists of NoteState objects to a list of lists of state indices"""
        state_sequence = [np.array([state.as_tuple() for state in part]) for part in states]
        return [np.vstack((np.searchsorted(pitches, seq[:,0]), np.searchsorted(durations, seq[:,1]))).T 
                for pitches, durations, seq in zip(self.pitches, self.durations, state_sequence)]

    @check_state_initialization
    def idx_to_state(self, indices):
        """Convert a list of lists of state indices to a list of lists of NoteState objects"""
        return [[NoteState(p, float(d)) for p, d in zip(pitches[ind[:,0]], durations[ind[:,1]])] 
                for pitches, durations, ind in zip(self.pitches, self.durations, indices)]
               
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
               
    