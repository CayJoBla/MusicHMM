def dep_gen_song(self, measures=12, measure_len=4):
    """Sample a new song from the HMM with dependence between parts
    
    Parameters:
        num_notes (int): Number of notes to sample from each part
        parts (list): List of part indices to generate in song
    """
    def measureize(obs, measures, measure_len=4):
        """Return measure-ized sequence of notes"""
        k = 0       # Index for observation states
        max_time = 12 * measure_len
        for m in range(measures):
            T = 0
            while T != max_time:
                d = int(obs[k][-1] * 12)
                if T + d > max_time:
                    d = max_time - T
                    obs[k] = [obs[k][l] for l in range(self.n_parts)] + [float(d) / 12]
                T += d
                k += 1
                
        return obs[:k]
        
    
    # Generate states from trained models
    hidden_seq = []      # List of hidden states sampled
    for i in range(self.n_parts):
        hmm = self.HMMs[i]
        Z, X = hmm.sample(measures*measure_len*12)
        if i == (-1 % self.n_parts):
            bass_seq =  Z.ravel()
        else:
            hidden_seq.append(X)
        
    # Measure-ize bass
    bass = measureize(np.array(self.states[-1])[bass_seq], measures, measure_len)
        
    # Discretize bass states
    bass_disc = []
    for n in bass:
        bass_disc += [n[-2]] * int(n[-1] * 12)
        
    # Generate dependent states while measurizing
    note_seq = [[] for _ in range(self.n_parts-1)] + [bass.tolist()]
    conditional_notes = {}
    K = [0 for _ in range(self.n_parts-1)]
    max_time = 12 * measure_len
    for m in range(measures):
        for i in range(self.n_parts-1):
            part_states = self.states[i]
            x_states = hidden_seq[i]          # Generated hidden states
            hmm = self.HMMs[i]
            
            T = 0
            while T < max_time:
                cond = bass_disc[T]           # Find bass note condition

                # Find note state indices that satisfy the bass note condition
                if (i,cond) not in conditional_notes.keys():
                    cond_ind = [j for j in range(len(part_states)) 
                                if (part_states[j][-2]==cond)]
                    conditional_notes[(i,cond)] = cond_ind
                else:
                    cond_ind = conditional_notes[(i,cond)]
                cond_ind = np.array(cond_ind)
                
                # What if there are no notes conditionally?
                if len(cond_ind) > 0:
                    # Get conditional probabilities
                    prob = np.array(hmm.emissionprob_[x_states[K[i]], cond_ind])
                    cond_prob = prob / np.sum(prob)
                    
                    # Generate a note conditioned on the discretized bass
                    note_ind = cond_ind[np.random.choice(np.arange(len(prob)), 
                                                            size=1, p=cond_prob)]
                    n = part_states[note_ind[0]]
                else:
                    n = [0 for _ in range(self.n_parts-1)] + [cond, 1.0]
                
                # Adjust timing for measurization
                d = int(n[-1] * 12)
                if T + d > max_time:
                    d = max_time - T
                    n = [n[l] for l in range(self.n_parts)] + [float(d) / 12]
                T += d
                
                K[i] += 1
                note_seq[i].append(n)
                
    # Cheat - Create a final C chord to end the song
    chord = [pitch.Pitch('C4'), pitch.Pitch('E3'), 
                pitch.Pitch('G2'), pitch.Pitch('C2'), 4.0]
    for p in note_seq:
        p.append(chord)
    
    
            
    # Create the Song object
    s = stream.Score()
    for i in range(self.n_parts):
        p = stream.Part(id=i)
        for n in note_seq[i]:
            if n[i] == 0:
                p.append(note.Rest(quarterLength=n[-1]))
            else:
                p.append(note.Note(n[i],quarterLength=n[-1]))
        s.insert(0,p)
        
    new_song = Song()
    new_song.parse(s)
    new_song.gen_stream()
    
    return new_song