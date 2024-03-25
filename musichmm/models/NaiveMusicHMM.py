import numpy as np
from hmmlearn.hmm import CategoricalHMM

class NaiveMusicHMM:
    """Class to represent a Hidden Markov Model for music. This model assumes that each 
    part of the music is independent of the others.

    Attributes:
        n_components (int): number of components for the HMM
        hmm (list(CategoricalHMM)): HMM model to train and sample from
        states (ndarray): Array of possible states
        state_to_idx (dict): Map from state to index

        # obs (list): Observed sequences of notes by index
        # obs_len (list): length of observation sequence for each song
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
        observations = self._initialize_states(data)
        
        # Train an HMM for each part
        self.HMMs = [CategoricalHMM(n_components=self.n_components).fit(
                        sequence, lengths=lengths
                    ) for sequence, lengths in observations]

        return self
        
    def _initialize_states(self, dataset):
        """Returns the concatenated dataset, divided by part, as a sequence of indices that map to 
        note states. Also initializes the `states_`, `state_to_idx_`, and `n_` hidden attributes.
        Called internally when fitting. States are given as NoteState objects as parsed in the Song 
        object class.

        Parameters:
            dataset (SongDataset): A dataset of Song objects to train on

        Returns:
            (list(tuple(ndarray(int),ndarray(int)))): List of arrays of note state indices and sequence
                lengths. The first tuple element is the sequence of note state indices for each part.
                Each tuple in the list corresponds to a part. Since note state sequences for all the 
                songs in a part are concatenated together, the second array in each tuple is an array of 
                sequence lengths corresponding to each song in the part.

                Example:
                    For a toy dataset with 2 parts and 3 songs, the output might look like:
                    [
                        (
                            np.array([0,2,3,1,4,3,5,2,6,7,1,3]),
                            np.array([3, 5, 4])
                        ),
                        (
                            np.array([0,3,4,2,5,3,6,4,7,8,2,4,5,6,1,3,2,5,6,7,8,1]),
                            np.array([7, 8, 7])
                        )
                    ]
                    Note that the note state indices are not shared between parts.
        """
        part_sequences = dataset.to_states()    # (part, song, state sequence)
        num_parts = len(part_sequences)

        state_to_idx = []   # List of dictionaries mapping unique note states to indices for each part
        sequences = []      # List of sequences of note state indices for each part

        for i, part in enumerate(part_sequences):
            lengths = np.array([len(song) for song in part])
            unique_states, obs = np.unique(np.concatenate(part), return_inverse=True)
            state_to_idx.append({state:i for i, state in enumerate(unique_states)})
            sequences.append((obs, lengths))

        self.states_ = unique_states
        self.state_to_idx_ = state_to_idx
        self.n_ = num_parts
        
        return sequences
               
    def gen_song(self, num_notes=30):
        """Sample a new song from the HMMs
        
        
        """
        # s = stream.Score()
        # for i in parts:
        #     ind = self.HMMs[i].sample(num_notes)[0].ravel()
        #     notes = self.notes[ind]
            
        #     p = stream.Part(id=i)
        #     for n in notes: 
        #         if n[0] == 0:
        #             p.append(note.Rest(quarterLength=n[1]))
        #         else:
        #             p.append(note.Note(n[0],quarterLength=n[1]))
        #     s.insert(0,p)
            
        # new_song = Song()
        # new_song.parse(s)
        # new_song.gen_stream()
        
        return new_song