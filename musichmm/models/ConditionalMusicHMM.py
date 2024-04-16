import numpy as np
from hmmlearn.hmm import CategoricalHMM
from musichmm.data.Song import Song
from musichmm.data.NoteState import NoteState
from musichmm.models.MusicHMMBase import MusicHMMBase, check_state_initialization
from musichmm.utils import Discretized
import warnings

class ConditionalMusicHMM(MusicHMMBase):
    """Class to represent a Hidden Markov Model for music. This model takes into account 
    pitch dependencies between the parts of the a song. In order to reduce the complexity of
    the state space, two assumptions are made in this model:
        1. The duration of a note is independent of the pitch of the note (i.e. we model the
            pitch and duration of a note as two separate HMMs)
        2. The pitch of a note is only dependent on the pitch of the bass note (i.e. we do not
            account for dependencies between the pitches of the other parts of the song). 
            The bass part is assumed to be the last part in the song.

    Attributes:
        hmm (list(CategoricalHMM)): List of HMM models to train and sample from.
    """
    def __init__(self, n_parts, *args, **kwargs):
        """Initialize the ConditionalMusicHMM class with the underlying CategoricalHMM models"""
        super().__init__()
        self.hmm = {
            "pitch": [CategoricalHMM(*args, **kwargs) for _ in range(n_parts)],
            "duration": [CategoricalHMM(*args, **kwargs) for _ in range(n_parts)]
        }
        self.n_parts = n_parts
        
    def fit(self, dataset, **kwargs):
        """Train the HMM on the given dataset of songs
        
        Parameters:
            songs (SongDataset): A SongDataset object containing songs to train on
        """
        # Get sequences and sequence lengths to pass to hmm.fit()
        sequences, lengths = self.initialize_states(dataset, **kwargs) 
        
        # Train an HMM for each part (bass part is formatted the same, so no special treatment is needed)
        for i, pitch_hmm in enumerate(self.hmm["pitch"]):
            pitch_hmm.fit(sequences[i][:, np.newaxis, 0], lengths=lengths[i])
        for i, duration_hmm in enumerate(self.hmm["duration"]):
            duration_hmm.fit(sequences[i][:, np.newaxis, 1], lengths=lengths[i])

        return self
        
    def initialize_states(self, dataset, **kwargs):
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

        # Check that the dataset has the correct number of parts
        part_sequences = dataset.to_states()    # (part, song, state sequence)
        if len(part_sequences) > self.n_parts:
            warning.warn(f"NaiveMusicHMM was initialized with {self.n_parts} parts, but the dataset has "
                         f"{len(part_sequences)} parts. Only the first {self.n_parts} parts will be used.")
        elif len(part_sequences) < self.n_parts:
            raise ValueError(f"NaiveMusicHMM was initialized with {self.n_parts} parts, but the dataset has "
                             f"{len(part_sequences)} parts. Please provide a dataset containing songs with at "
                             f"least {self.n_parts} parts.")

        # Discretize the bass part to add conditional dependence for the other parts
        discretized_bass = Discretized(np.concatenate(part_sequences[-1]))

        # Convert the state sequences to indices
        lengths = []
        sequences = []
        unique_pitches = []
        unique_durations = []
        for i, part in enumerate(part_sequences):
            lengths.append(np.array([len(song) for song in part]))                              # Get song lengths
            state_sequence = np.array([state.as_tuple() for song in part for state in song])    # Sequence of (pitch, duration)
            pitch_sequence, duration_sequence = state_sequence.T                                # Separate pitch and duration sequences
            if i != self.n_parts-1:              # For non-bass parts, add bass pitch to the state
                timestamps = np.cumsum(duration_sequence.astype(float)).astype(int)             # Get times of each state
                timestamps = np.pad(timestamps, (1, 0))[:-1]                                    # Shift by one to get the correct timestamps
                bass_pitches = discretized_bass[timestamps]                                          # Corresponding bass pitches
                pitch_sequence = np.vstack((bass_pitches, pitch_sequence)).T                    # Augment states with bass pitches
            pitches, pitch_sequences = np.unique(pitch_sequence, axis=0, return_inverse=True)   # Get unique pitches and encode as indices
            pitches = pitches.reshape(-1,1) if pitches.ndim==1 else pitches                     # Ensure pitches is 2D
            unique_pitches.append(pitches)
            durations, duration_sequences = np.unique(duration_sequence, return_inverse=True)   # Get unique durations and encode as indices
            unique_durations.append(durations)
            sequences.append(np.vstack((pitch_sequences, duration_sequences)).T)                # Sequence of (pitch index, duration index)

        self.pitches = unique_pitches
        self.durations = unique_durations

        self.initialized = True     # Successfully initialized states
        
        return sequences, lengths

    def sample(self, num_notes, currstate=None):
        """Sample a new song from the HMMs with dependence on the bass part

        Parameters:
            num_notes (int): Number of states to sample from each part

        Returns:
            list(ndarray(int)): A list of 2d sequences of state indices. Each element in the list corresponds to a part.
        """
        if currstate is not None:
            raise NotImplementedError("Specifying a current state has not been implemented for ConditionalMusicHMM sampling")

        # Sample hidden states from the pitch HMMs (and observations from bass part)
        hidden_pitch_states = []
        for i, hmm in enumerate(self.hmm["pitch"]):
            X, Z = hmm.sample(num_notes, currstate=None)
            if i == self.n_parts-1:
                X_bass_pitch = X
            else:
                hidden_pitch_states.append(Z)

        # Sample observations from the duration HMMs
        X_duration = [hmm.sample(num_notes, currstate=None)[0] 
                        for hmm in self.hmm["duration"]]

        # Decode the bass observations
        bass_notes = self.decode_bass(X_bass_pitch, X_duration[-1])
        discretized_bass = Discretized(bass_notes)

        # Generate conditionally dependent observation states
        X_pitch = []
        for i, hmm in enumerate(self.hmm["pitch"][:-1]):    # For each non-bass part
            # Get the time indices for the bass notes
            duration_sequence = self.durations[i][X_duration[i]].astype(float)

            timestamps = np.pad(np.cumsum(duration_sequence).astype(int), (1,0))[:-1]   # Get times of each state with index shift
            
            # Find note state indices that satisfy the bass note conditions at each time
            conditional_indices = [np.argwhere(self.pitches[i][:,0]==pitch).flatten() 
                                    for pitch in discretized_bass[timestamps]]
            
            # Get the conditional probabilities for the states that satisfy the bass note conditions
            emission_probs = hmm.emissionprob_[hidden_pitch_states[i]]
            unnormalized_probs = [probs[indices] for probs, indices in zip(emission_probs, conditional_indices)]
            conditional_probs = [probs / np.sum(probs) for probs in unnormalized_probs]

            # Sample the next note state based on the conditional probabilities
            X = [indices[np.random.choice(np.arange(len(probs)), p=probs)] if len(indices) > 0 else -1
                 for indices, probs in zip(conditional_indices, conditional_probs)]

            X_pitch.append(np.reshape(X, (-1,1)))
        
        X_pitch.append(X_bass_pitch)
        return [np.hstack((pitch, duration)) for pitch, duration in zip(X_pitch, X_duration)]

    @check_state_initialization
    def decode_bass(self, X_pitch, X_duration):
        """Decode the sampled bass notes from the pitch and duration indices"""
        pitches = self.pitches[-1][X_pitch].flatten()
        durations = self.durations[-1][X_duration].flatten()
        return [NoteState(p,float(d)) for p, d in zip(pitches, durations)]

    @check_state_initialization
    def idx_to_state(self, indices):
        """Convert state indices to NoteState objects"""
        return [
            [NoteState(p,float(d)) for p,d in zip(self.pitches[i][part[:,0]][:,-1], self.durations[i][part[:,1]])]
            for i, part in enumerate(indices)
        ]

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
    
                
