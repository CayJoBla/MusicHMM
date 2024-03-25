import music21
import pickle
import os
from tqdm import tqdm

from .SongDataset import SongDataset
from .Song import Song

class BachDataset(SongDataset):
    composer = 'bach'

    def __init__(self, transpose_key='C', major=True):
        mode = 'major' if major else 'minor'
        filename = f'{self.composer}_{transpose_key}_{mode}.pkl'

        if os.path.exists(filename):    # Load the data from file
            print(f"Loading {self.composer} dataset from {filename}")
            with open(filename, 'rb') as file:
                songs = pickle.load(file)
        else:                           # Load the data from the music21 corpus
            song_paths = music21.corpus.getComposer(self.composer)
            songs = []
            for i, song_path in tqdm(enumerate(song_paths), total=len(song_paths), desc=f"Parsing {self.composer} dataset from music21"):
                score = music21.corpus.parse(song_path)
                song = Song(score)
                if len(score.parts) != 4 or song.key.mode != mode:          # Filter songs
                    continue
                if transpose_key is not None and transpose_key != song.key: # Transpose
                    song.transpose(transpose_key)
                songs.append(song)
            print(f"Extracted {len(songs)} songs from the music21 corpus")
            print(f"Saving {self.composer} dataset to {filename}")
            with open(filename, 'wb') as file:      # Save to file
                pickle.dump(songs, file)   

        super().__init__(songs)