# MusicHMM
This project contains multiple implementations using Hidden Markov Models (HMMs) for music generation. Included are 5 different models, each with different underlying assumptions about the music they attempt to model. 

## Data
For training, I use the Bach chorales corpus from the music21 python package. The corpus was then downsampled to only chorales in a major key, and only those chorales with 4 parts, resulting in a total of 182 chorales. These were then transposed to the key of C major to simplify the model.

## Models
Using 20 hidden states for each HMM, I trained 5 different models on the Bach chorales corpus. Each model makes different assumptions about the music it is attempting to model.  

### Simple MusicHMM
The model only takes a single part of the music into account, and models the music using a single Hidden Markov Model. The observation states are tuples of the pitch and duration of the notes.

### Simple Independent MusicHMM
This model again only takes a single part of the music into account. However, it models the music using two independent Hidden Markov Models, one for the pitch and one for the duration of the notes. This effectively assumes that the pitch and duration of the notes are independent of each other, which, while not necessarily the most useful assumption, drastically reduces the number of unique observation states in the underlying HMMs.

### Naive MusicHMM
This model takes all parts of the music into account, but makes the naive assumption that each part of the music is independent of all other parts. This is done by creating a separate Hidden Markov Model for each part of the music. A before, the observation states for each HMM are tuples of the pitch and duration of the notes.

### Naive Independent MusicHMM
This model is an extension of the Naive MusicHMM, but models the pitch and duration of the notes independently of each other as was done in the Simple Independent model. This is done by creating two separate Hidden Markov Models for each part of the music, one for the pitch and one for the duration of the notes. This results in a total of 8 different HMMs for the Bach chorales corpus.

### Conditional MusicHMM
This final model is the most complex of the models. It takes all parts of the music into account, as the naive models do; however, it attempts to introduce conditional dependencies into the model between the bass pitches and other parts of the music. This is done by including the concurrent bass pitch into the state of the other parts of the music. During generation, the model will sample the bass pitch first, and then sample the other parts of the music conditioned on the bass pitch. In order to reduce the number of unique observation states, the model again assumes that the pitch and duration of the notes are independent of each other, resulting in a total of 8 different HMMs for the Bach chorales corpus. 

## Results
Music sampled from each of the models can be found in the `generated_songs` directory. The music was generated by sampling 30 notes from each HMM, and then truncating each song to the length of its shortest part. The music was then converted to MIDI format using the music21 python package. 

Despite attempting to incorporate dependencies into the final model, the music generated by the Conditional MusicHMM still contains a lot of discordant notes. This is likely due to the fact that while the model captures dependencies between the bass and other parts of the music, it does not capture dependencies between the other parts of the music. This is a limitation of the model that could be addressed in future work.