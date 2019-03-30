import numpy as np

from HMMLearner.MLELearner import MLEBigramCorpusLearner as MLELearner


class KneserNeyMLELearner(MLELearner):
    def __init__(self, *args):
        super().__init__(*args)
        self.lambda_state_state = dict()
        self.lambda_state_observation = dict()
        self.p_continuation_state_state = dict()
        self.p_continuation_state_observation = dict()
        self.d = 0.75

    def make_probabilities(self):
        self.a = np.zeros((len(self.states), len(self.states)+1), dtype=float)
        self.b = np.zeros((len(self.states)+1, len(self.observation_symbols)), dtype=float)
        self.make_continuation_probability()
        self.make_lambdas()
        for i, state1 in enumerate(self.states):
            for j, state2 in enumerate(self.states + [self.STOP]):
                self.a[i, j] =\
                    max(self.state_bigram_counts.get((state1, state2), 0)-self.d, 0)
                self.a[i, j] /= self.state_counts[state1]
                self.a[i, j] += self.lambda_state_state[state1]*self.p_continuation_state_state[state2]
            for observation_symbol, j in self.observation_symbols.items():
                self.b[i, j] =\
                    max(self.state_observation_counts.get((state1, observation_symbol), 0)-self.d, 0)
                self.b[i, j] /= self.state_counts[state1]
                self.b[i, j] += self.lambda_state_observation[state1]*self.p_continuation_state_observation[observation_symbol]

        self.initial_probabilities = np.zeros(len(self.states))
        for i, state in enumerate(self.states):
            self.initial_probabilities[i] = max(self.state_bigram_counts.get((self.START, state), 0) - self.d, 0)
            self.initial_probabilities[i] /= self.state_counts[self.START]
            self.initial_probabilities[i] += self.lambda_state_state[self.START]*self.p_continuation_state_state[state]

    def make_continuation_probability(self):
        denominator_state_state = sum(self.state_bigram_counts.values())
        denominator_state_observation =\
            sum(self.state_observation_counts.values())
        for state2 in self.states + [self.STOP]:
            numerator_state_state = len(
                [state1 for state1 in self.states + [self.START, self.STOP]
                 if self.state_bigram_counts.get((state1, state2), 0) > 0
                 ]
                )
            self.p_continuation_state_state[state2] =\
                numerator_state_state/denominator_state_state
        for observation_symbol in self.observation_symbols:
            numerator_state_observation = len(
                [state for state in self.states 
                 if self.state_observation_counts.get((state, observation_symbol), 0) > 0
                 ]
                )
            self.p_continuation_state_observation[observation_symbol] =\
                numerator_state_observation/denominator_state_observation

    def make_lambdas(self):
        for state in self.states + [self.START, self.STOP]:
            num_bigrams_state_state = len(
                [state2 for state2 in self.states 
                 if self.state_bigram_counts.get((state, state2), 0) > 0
                 ]
                )
            num_bigrams_state_observation = len(
                [observation_symbol for observation_symbol 
                 in self.observation_symbols
                 if self.state_observation_counts.get((state, observation_symbol), 0) > 0
                 ]
                )
            self.lambda_state_state[state] = (self.d/self.state_counts[state])*num_bigrams_state_state
            self.lambda_state_observation[state] = (self.d/self.state_counts[state])*num_bigrams_state_observation
