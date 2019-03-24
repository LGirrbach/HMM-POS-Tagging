from nltk import bigrams

import TransitionMatrix
import HiddenMarkovModel


class MLEBigramCorpusLearner:
    def __init__(self, start_symbol, stop_symbol, unknown_observation_symbol, observation_count_threshold=3):
        self.state_counts = dict()
        self.observation_counts = dict()
        self.state_bigram_counts = dict()
        self.state_observation_counts = dict()
        self.state_state_transitions = dict()
        self.state_observation_transitions = dict()
        self.all_states = []
        self.all_observation_symbols = []
        self.START = start_symbol
        self.STOP = stop_symbol
        self.UNK = unknown_observation_symbol
        self.observation_count_threshold = observation_count_threshold

    def make_counts(self, documents):
        for document in documents:
            for observation_symbol, _ in document:
                observation = observation_symbol.lower()
                self.observation_counts[observation_symbol] =\
                    self.observation_counts.get(observation_symbol, 0) + 1

        for document in documents:
            self.state_counts[self.START] =\
                self.state_counts.get(self.START, 0) + 1
            first_observation_symbol, first_state = document[0]
            self.state_bigram_counts[(self.START, first_state)] =\
                self.state_bigram_counts.get((self.START, first_state), 0) + 1

            for i, (observation_symbol, state) in enumerate(document):
                observation_symbol = observation_symbol.lower()
                if self.observation_counts.get(observation_symbol, 0) <\
                        self.observation_count_threshold:
                    observation_symbol = self.UNK
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

    def make_transition_probabilities(self):
        for (state1, state2), count in self.state_bigram_counts.items():
            self.state_state_transitions[(state1, state2)] =\
                count/self.state_counts[state1]

        for (state, observation_symbol), count in self.state_observation_counts.items():
            self.state_observation_counts[(state, observation_symbol)] =\
                count/self.state_counts[state]

    def collect_states_and_observation_symbols(self):
        self.states = list(sorted(self.state_counts.keys()))
        assert self.START in self.states
        assert self.STOP in self.states
        self.observation_symbols = list(
            sorted(set([observation_symbol for _, observation_symbol
                        in self.state_observation_counts.keys()]))
            )
        assert self.UNK in self.observation_symbols
    
    def collect_transition_probabilities(self):
        self.state_state_transitions = [
            (state1, state2, transition_probability)
            for (state1, state2), transition_probability
            in self.state_state_transitions.items()
            ]
        self.state_observation_transitions = [
            (state, observation_symbol, transition_probability)
            for (state, observation_symbol), transition_probability
            in self.state_observation_counts.items()
            ]

    def learn_hmm(self, documents):
        self.make_counts(documents)
        self.make_transition_probabilities()
        self.collect_states_and_observation_symbols()
        self.collect_transition_probabilities()
        
        return HiddenMarkovModel.HiddenMarkovModel(
            self.START, self.STOP, self.UNK,
            TransitionMatrix.StateTransitionMatrix(
                self.states, self.state_state_transitions
                ),
            TransitionMatrix.TransitionMatrix(
                self.states, self.observation_symbols,
                self.state_observation_transitions
                )
            )
