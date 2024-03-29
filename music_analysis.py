"""
Contains objects and methods for creating and training HMMs and classifiers on the music21 database
"""

from music21 import *
import music21
import numpy as np
import os
import dill as pickle
from hmmlearn.hmm import MultinomialHMM

def load_clean_split(composer_name, train_prop=0.6, seed=None, accept=lambda p: len(p.parts)==4):
    """
    Loads the corpus, filter to songs with 4 parts, transposes to C, splits into major and minor, splits into train/test.
    
    Parameters:
        composer_name (str) - name of composer to use from music21 database.
        train_prop (float) - proportion to use in training set
        seed - random seed to use for RNG (settable for consistency)
        accept (lambda function) - function that determines whether a given piece in the corpus should be included, returning True if so and False otherwise
    
    Returns:
        (major, minor) - tuple of training data
        (major, minor) - tuple of test data
    """
    rng = np.random.RandomState(seed=seed)
    paths = corpus.getComposer(composer_name)
    transposed_major = []
    transposed_minor = []
    # Load songs
    for i, p_name in enumerate(paths):
        p = corpus.parse(p_name)
        if not accept(p):
            continue
        k = p.analyze('key')
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        if 'major' in k.name:
            transposed_major.append(p.transpose(i))
        elif 'minor' in k.name:
            transposed_minor.append(p.transpose(i))
    # Train/test split
    def split(l):
        """Splits the list into train and test portions"""
        l = np.array(l, dtype=object)
        num = int(len(l) * train_prop)
        mask = rng.choice(len(l), size=num, replace=False)
        return l[mask], l[~mask]
        
    maj_tr, maj_test = split(transposed_major)
    min_tr, min_test = split(transposed_minor)
    
    return (maj_tr, min_tr), (maj_test, min_test)

class Song:
    """
    Class for holding relevant processed information about a piece, with several methods to convert to other useful formats.
    """
    def __init__(self):
        """Initialize Song class.
        
        Attributes:
            score (Score): Song from music21 Score class
            key (str): Key of the song
            parts (ndarray(p,n)): Array of parts(p) and notes(n)
            n_parts (int): Number of parts in the song
            stream (Stream): music21 Stream object generated from parts
        """
        return
        
    def parse(self, score):
        """Parse the MusicXML score into a trainable format"""
        parts = []
        discrete = [[]] * len(score.parts)
        
        # Generate discretized notelist for each part
        for i, p in enumerate(score.parts):
            for n in p.recurse().notesAndRests:
                if n.isRest:
                    discrete[i] = discrete[i] + ([0] * int(n.quarterLength*12))
                else:
                    discrete[i] = discrete[i] + ([n.pitch] * int(n.quarterLength*12))
        #pad to make lengths the same
        max_len = max([len(part) for part in discrete])
        discrete = [
            part + [part[-1]] * (max_len - len(part)) for part in discrete
        ]
        
        discrete = np.array(discrete)
        
        # Generate states from the music
        for i, p in enumerate(score.parts):
            t = 0         # Time for dependence on other parts
            notes = []    # States (pitch[0,t], ..., pitch[n_parts,t], dur)
            for n in p.recurse().notesAndRests:
                pitches = [part[t] for part in discrete]
                notes.append(pitches + [n.quarterLength])
                t += int(n.quarterLength*12)
                
            # Add notes list for each part to part list
            parts.append(notes)  
            
        self.score = score
        self.key = score.analyze('key')
        self.parts = np.array(parts, dtype=object)
        self.n_parts = len(parts)  
        
        return self
        
    def gen_stream(self, parts=None):
        """Generate a song from the parts and notes parsed
        
        Parameters:
            parts (list): list of part indices to generate
        """
        # If None, generate all parts
        if parts is None:
            parts = np.arange(self.n_parts)
        parts = np.array(parts, dtype=int)
        
        # Generate the song for the parts indicated
        s = stream.Stream()
        for i, notes in enumerate(self.parts[parts]):
            p = stream.Part(id=i)
            for n in notes: 
                if n[i] == 0:
                    p.append(note.Rest(quarterLength=n[self.n_parts]))
                else:
                    p.append(note.Note(n[i],quarterLength=n[self.n_parts]))
            s.insert(0,p)
        self.stream = s
        
        return self
        
    def play(self):
        """Play the song generated by gen_stream"""
        self.stream.show('midi')  
        
    def save(self, filename):
        """Save the generated song as a midi file"""
        self.stream.write('midi')        # Not working

class MusicHMM:
    def __init__(self, n_components):
        """Initialize MusicHMM class
        
        Attributes:
            songs (list): List of Song class objects to train on
            n_components (int): number of components for the HMM
            HMMs (list(MultinomialHMM)): HMM objects for each part
            states (ndarray): Array of possible states
            state_ind (func): Map from note state to index
            obs (list): Observed sequences of states by index
            obs_len (list): length of observation sequence for each song
        """
        self.n_components = n_components
        
        
    def fit(self, songs, parts=[0,1,2,3]):
        """Train the HMM on the list of songs for part indices given
        
        Parameters:
            songs (list): List of parsed Song objects to train on
            parts (list): List of part indices from songs to use in training
        """
        self.songs = songs
        
        # Generate state space data and observations by index
        self.init_states(parts)
        self.init_obs_matrices(parts)
        
        # Train a HMM for each part
        n_parts = len(parts)
        HMMs = []
        for i in range(n_parts):
            obs = np.array(self.obs[i]).reshape(-1, 1)   # Reshape observations
            hmm = MultinomialHMM(n_components=self.n_components)
            hmm.fit(obs, lengths=self.obs_len[i])
            HMMs.append(hmm)
            
        self.HMMs = HMMs
        self.n_parts = len(parts)
        
    def init_states(self, parts):
        """Create and save a dictionary of unique note states"""
        states = {tuple(n) for song in self.songs for p in song.parts[parts] for n in p}
        
        # Create ways to get ids from states and vice versa
        self.states = list(states)
        self.states_dict = {n:i for i,n in enumerate(self.states)}
        # Add a observation state for when we don't recognize the input
        self.state_ind = lambda n: self.states_dict.get(tuple(n), self.missing_state_id)
        self.missing_state_id = len(self.states)
        self.states.append(self.states[0]) # placeholder for missing states
        self.states = np.array(self.states)
        
        self.n_parts = len(parts)
        
    def init_obs_matrices(self, parts):
        """Create the matrices of note indices observed"""
        # Initialize
        obs = [[]] * self.n_parts
        obs_len = [[]] * self.n_parts
        
        # Process the songs
        for i in range(self.n_parts):
            for song in self.songs:
                p = song.parts[parts[i]]
                seq = [self.state_ind(n) for n in p]
                obs[i] += seq
                obs_len[i].append(len(seq))
                
        self.obs = obs
        self.obs_len = obs_len 
                
               
    def gen_song(self, num_notes=40):
        """Sample a new song from the HMM
        
        Parameters:
            num_notes (int): Number of notes to sample from each part
            parts (list): List of part indices to generate in song
        """
        s = stream.Score()
        part_lengths = []
        for i in range(self.n_parts):
            hmm = self.HMMs[i]
            ind = hmm.sample(num_notes)[0].ravel()
            states = self.states[ind]
            T = 0
            
            p = stream.Part(id=i)
            for n in states: 
                if n[i] == 0:
                    p.append(note.Rest(quarterLength=n[self.n_parts]))
                else:
                    p.append(note.Note(n[i],quarterLength=n[self.n_parts]))
                T += n[self.n_parts]
            s.insert(0,p)
            part_lengths.append(T)
        
        max_len = max(part_lengths)    
        for i, p in enumerate(s.parts):
            if part_lengths[i] < max_len:
                p.append(note.Rest(quarterLength=max_len-part_lengths[i]))
            
        new_song = Song()
        new_song.parse(s)
        new_song.gen_stream()
        
        return new_song