import numpy as np
from hmmlearn.hmm import CategoricalHMM
from musichmm.data.Song import Song
from musichmm.data.NoteState import NoteState
from musichmm.models.MusicHMMBase import MusicHMMBase, check_state_initialization
import warnings

class SimpleIndependentMusicHMM(MusicHMMBase):
    """Class to represent a Hidden Markov Model for music. This model only considers the
    first part of the music and uses a CategoricalHMM model representation.
    This model also assumes that the pitch and duration of notes are independent of each other.

    Attributes:
        hmm (dict(CategoricalHMM)): HMM models to train and sample from.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the underlying CategoricalHMM models for note pitch and duration"""
        super().__init__()
        self.hmm = {
            "pitch": CategoricalHMM(*args, **kwargs),
            "duration": CategoricalHMM(*args, **kwargs)
        }

    def fit(self, dataset):
        """Train the HMM on the given dataset of songs. 
        
        Parameters:
            songs (SongDataset): A SongDataset object containing songs to train on
        """
        # Get sequences and sequence lengths to pass to hmm.fit()
        sequences, lengths = self.initialize_states(dataset)
        self.hmm["pitch"].fit(sequences[:, np.newaxis, 0], lengths=lengths)
        self.hmm["duration"].fit(sequences[:, np.newaxis, 1], lengths=lengths)

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
            warnings.warn("SimpleIndependentMusicHMM only supports training on Songs with a single part." 
                          "Only the first part of each song will be used.")
        part = part_sequences[0]                    # (song, state sequence); only consider the first part
        self.n_parts = 1

        # Extract the song lengths, separate the pitches and durations, and convert to sequences of state indices
        lengths = np.array([len(song) for song in part])
        state_sequences = np.array([state.as_tuple() for song in part for state in song])
        unique_pitches, pitch_sequences = np.unique(state_sequences[:,0], return_inverse=True)
        unique_durations, duration_sequences = np.unique(state_sequences[:,1], return_inverse=True)
        sequences = np.hstack((pitch_sequences.reshape(-1,1), duration_sequences.reshape(-1,1)))

        # Store the unique states and a define mappings between states to indices
        self.pitches = unique_pitches
        self.durations = unique_durations

        self.initialized = True     # Successfully initialized states

        return sequences, lengths

    def sample(self, num_notes, currstate=None):
        curr_pitch, curr_duration = self.state_to_idx([currstate])[0] if currstate is not None else (None, None)
        X_pitch = self.hmm["pitch"].sample(num_notes, currstate=curr_pitch)[0]
        X_duration = self.hmm["duration"].sample(num_notes, currstate=curr_duration)[0]
        return np.hstack((X_pitch.reshape(-1,1), X_duration.reshape(-1,1)))

    @check_state_initialization
    def state_to_idx(self, states):
        """Convert a list of NoteState objects to a list of state indices"""
        state_sequence = np.array([state.as_tuple() for state in states])
        pitch_indices = np.searchsorted(self.pitches, state_sequence[:,0])
        duration_indices = np.searchsorted(self.durations, state_sequence[:,1])
        return np.hstack((pitch_indices.reshape(-1,1), duration_indices.reshape(-1,1)))

    @check_state_initialization
    def idx_to_state(self, indices):
        """Convert a list of state indices to a list of NoteState objects"""
        pitches = self.pitches[indices[:,0]]
        durations = self.durations[indices[:,1]]
        return [NoteState(p, float(d)) for p, d in zip(pitches, durations)]
               
    def sequence_to_song(self, X):
        return Song.from_sequences([self.idx_to_state(X)])