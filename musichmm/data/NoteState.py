from music21.note import Note, Rest
from music21.pitch import Pitch
from enum import Enum

class State:
    """Abstract class for different states extracted from a song sequence"""

    # Use enum to establish state ordering for comparisons
    StateType = Enum("StateType", ["START", "END", "NOTE", "REST"])

    def __init__(self, state_type):
        self._type = self.StateType[state_type.upper()]

    def note(self):
        return None

    def __getattribute__(self, attr):
        if attr == "type":
            return self._type.name
        return super().__getattribute__(attr)

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.__dict__ == other.__dict__

    def __lt__(self, other):
        return self._type.value < other._type.value
    
    def __le__(self, other):
        return (self < other) or (self == other)
    
    def __gt__(self, other):
        return not (self < other)
    
    def __ge__(self, other):
        return (self > other) or (self == other)

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __str__(self):
        return self.__repr__()


class NoteState(State):
    def __init__(self, pitch, duration):
        super().__init__(state_type="REST" if pitch=="REST" else "NOTE")
        self.pitch = str(pitch)
        self.duration = duration

    def note(self):
        """Return a music21 Note or Rest object from the NoteState"""
        if self.type == "REST":
            return Rest(quarterLength=self.duration)
        else:
            return Note(pitch=self.pitch, quarterLength=self.duration)
        
    def __lt__(self, other):
        if not isinstance(other, NoteState) or self.type != other.type:
            return super().__lt__(other)
        elif self.pitch != other.pitch:
            return Pitch(self.pitch) < Pitch(other.pitch)
        else:
            return self.duration < other.duration

    def __repr__(self):
        return f"NoteState({self.pitch}, {self.duration})"

    def as_tuple(self):
        return (self.pitch, self.duration)

    def __getitem__(self, key):
        return self.as_tuple()[key]

    @classmethod
    def from_note(cls, note):
        pitch = "REST" if note.isRest else str(note.pitch)
        return cls(pitch, note.quarterLength)
        

class StartState(State):
    def __init__(self):
        super().__init__(state_type="START")


class EndState(State):
    def __init__(self):
        super().__init__(state_type="END")
        

        


