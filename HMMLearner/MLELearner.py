import numpy as np

import HiddenMarkovModel


class MLEBigramCorpusLearner:
    def __init__(self, possible_states, possible_observation_symbols, start_symbol, stop_symbol):
        self.state_counts = dict()
        self.state_bigram_counts = dict()
        self.state_observation_counts = dict()
        self.states = possible_states
        self.observation_symbols = possible_observation_symbols
        self.START = start_symbol
        self.STOP = stop_symbol

        self.a = np.empty(0)
        self.b = np.empty(0)
        self.initial_probabilities = np.empty(0)

    def make_counts(self, documents):
        for document in documents:
            self.state_counts[self.START] =\
                self.state_counts.get(self.START, 0) + 1
            first_observation_symbol, first_state = document[0]
            assert first_observation_symbol in self.observation_symbols
            assert first_state in self.states
            self.state_bigram_counts[(self.START, first_state)] =\
                self.state_bigram_counts.get((self.START, first_state), 0) + 1

            for i, (observation_symbol, state) in enumerate(document):
                assert observation_symbol in self.observation_symbols
                assert state in self.states
                try:
                    _, next_state = document[i+1]
                except IndexError:
                    next_state = self.STOP
                    self.state_counts[self.STOP] =\
                        self.state_counts.get(self.STOP, 0) + 1
                self.state_counts[state] = self.state_counts.get(state, 0) + 1
                self.state_bigram_counts[(state, next_state)] =\
                    self.state_bigram_counts.get((state, next_state), 0) + 1
                self.state_observation_counts[(state, observation_symbol)] =\
                    (self.state_observation_counts.get(
                            (
                                state, observation_symbol
                            ), 0
                        )
                        + 1
                     )
    
    def make_probabilities(self):
        self.a = np.zeros((len(self.states), len(self.states)+1))
        self.b = np.zeros((len(self.states)+1, len(self.observation_symbols)))
        for i, state1 in enumerate(self.states):
            for j, state2 in enumerate(self.states + [self.STOP]):
                self.a[i, j] = self.state_bigram_counts.get((state1, state2), 0) / self.state_counts[state1]
            for observation_symbol, j in self.observation_symbols.items():
                self.b[i, j] = self.state_observation_counts.get((state1, observation_symbol), 0) / self.state_counts[state1]

        self.initial_probabilities = np.asarray([
            self.state_bigram_counts.get((self.START, state), 0) for state
            in self.states
            ], dtype=float)
        self.initial_probabilities /= self.state_counts[self.START]
            

    def learn_hmm(self, documents):
        self.make_counts(documents)
        self.make_probabilities()

        return HiddenMarkovModel.HiddenMarkovModel(
            self.START, self.STOP,
            self.states, self.observation_symbols,
            self.initial_probabilities,
            self.a, self.b
            )
