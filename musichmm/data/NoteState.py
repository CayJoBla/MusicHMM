from music21.note import Note, Rest
from music21.pitch import Pitch

class NoteState:
    """Class to represent a note state in the HMM.
    Main purpose is to compare notes and rests in the state space.
    
    Attributes:
        pitch (str): The string representation of the pitch of the note
        quarterLength (float): The duration of the note in quarter notes
    """
    def __init__(self, note):
        """Initialize a NoteState object from a music21 note or rest object"""
        self.isRest = note.isRest
        self.pitch = str(note.pitch) if not self.isRest else None
        self.quarterLength = note.quarterLength

    def note(self):
        """Return a music21 note object from the NoteState"""
        if self.isRest:
            return Rest(quarterLength=self.quarterLength)
        else:
            return Note(self.pitch, quarterLength=self.quarterLength)

    def __eq__(self, other):
        """Check if two NoteState objects are equal"""
        return (self.pitch == other.pitch) and (self.quarterLength == other.quarterLength)

    def __lt__(self, other):
        """Check if one NoteState is less than another"""
        if isinstance(other, SongStart):
            return False
        elif isinstance(other, SongEnd):
            return True
        elif self.isRest != other.isRest:             # One is a rest and the other is not
            return self.isRest
        elif not self.isRest and not other.isRest:  # Neither are rests
            if self.pitch != other.pitch:
                return Pitch(self.pitch) < Pitch(other.pitch)
            else:
                return self.quarterLength < other.quarterLength
        else:                                       # Both are rests
            return self.quarterLength < other.quarterLength
    
    def __le__(self, other):
        """Check if one NoteState is less than or equal to another"""
        return (self < other) or (self == other)
    
    def __gt__(self, other):
        """Check if one NoteState is greater than another"""
        return not (self < other)
    
    def __ge__(self, other):
        """Check if one NoteState is greater than or equal to another"""
        return (self > other) or (self == other)

    def __repr__(self):
        """Return a string representation of the NoteState"""
        return f"NoteState({self.pitch}, {self.quarterLength})"

    def __str__(self):
        """Return a string representation of the NoteState"""
        return f"{self.pitch} ({self.quarterLength})"

    # def __hash__(self):
    #     """Return a hash of the NoteState"""
    #     return hash((self.pitch, self.quarterLength))


class SequenceState(NoteState):
    """Abstract class for the SongStart and SongEnd states"""
    def __init__(self):
        self.isRest = False
        self.pitch = None
        self.quarterLength = 0.0
    
    def note(self):
        return None

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __lt__(self, other):
        # Should be defined in subclasses
        raise NotImplementedError("Cannot compare SequenceState objects")
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return self.__class__.__name__


class SongStart(SequenceState):
    def __lt__(self, other):
        if isinstance(other, NoteState):
            return True
        else:
            raise ValueError("Cannot compare SongStart with non-NoteState object")


class SongEnd(SequenceState):
    def __lt__(self, other):
        if isinstance(other, NoteState):
            return False
        else:
            raise ValueError("Cannot compare SongEnd with non-NoteState object")

