from music21 import stream, note, pitch, interval, midi
import numpy as np

from .NoteState import NoteState#, StartState, EndState

class Song:
    """Class to represent a song from the music21 library
    
    Attributes:
        score (Score): Song from music21 Score class
        key (str): Key of the song
        parts (ndarray(p,n)): Array of parts(p) and notes(n)
    """
    def __init__(self, score):
        """Initialize a Song object from a music21 Score object."""
        self.score = score
        self.key = score.analyze('key')
        
    def to_states(self):
        """Parse the MusicXML score into a series of states for each part"""
        states = np.empty(len(self.score.parts), dtype=object)
        for i, part in enumerate(self.score.parts):
            # states[i] = [StartState()]
            states[i] = [NoteState.from_note(n) for n in part.recurse().notesAndRests if n.quarterLength > 0]
            # states[i] += [EndState(), StartState()]
        return states
        
    def stream(self):
        """Generate a Stream object from the parts and notes parsed
        
        Parameters:
            part_indices (array): Part indices to include in the generated stream
        """
        return self.score.flatten()

    def transpose(self, key='C'):
        """Transpose the song to the specified key
        
        Parameters:
            key (str): Key to transpose the song to
        """
        i = interval.Interval(self.key.tonic, pitch.Pitch(key))
        self.score = self.score.transpose(i)
        self.key = self.score.analyze('key')

    def part(self, i):
        """Return part of the song as a new Song object.
        
        Parameters:
            i (int, str): Index or name of the part to extract
        """
        return Song(stream.Score(self.score.parts[i]))

    @classmethod
    def from_sequences(cls, sequences):
        """Create a Song object from a list of part sequences of NoteState objects"""
        score = stream.Score()
        for i, part in enumerate(sequences):
            score.insert(0, stream.Part([state.note() for state in part if isinstance(state, NoteState)], id=i))
        return cls(score)

    def play(self):
        """Play the song using the music21 midi player"""
        self.stream().show('midi')

    def show(self):
        """Show the song using the music21 musicxml player"""
        self.score.show()

    def save(self, filename):
        """Save the generated song as a midi file"""
        self.stream().write('midi', fp=filename)



        