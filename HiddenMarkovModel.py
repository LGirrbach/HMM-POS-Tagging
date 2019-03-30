import numpy as np


class HiddenMarkovModel:
    def __init__(self, start_state, final_state,
                 possible_states, observations, initial_probabilities, 
                 transition_matrix, emission_matrix):
        self.start_state = start_state
        self.final_state = final_state
        self.unknown_word_symbol = None
        self._states = tuple(possible_states)
        self.observation_to_index_mapping = observations
        self.state_to_index_mapping = {state: i for i, state
                                       in enumerate(self._states)}
        self.state_to_index_mapping[self.final_state] = len(self._states)
        self._transition_matrix = transition_matrix
        self._emission_matrix = emission_matrix
        self.initial_probabilities = initial_probabilities
        
        assert len(self._states) == len(self.initial_probabilities)
        assert isinstance(self._transition_matrix, np.ndarray)
        assert isinstance(self._emission_matrix, np.ndarray)
        assert self._transition_matrix.shape ==\
            (len(self._states), len(self._states) + 1)
        assert self._emission_matrix.shape ==\
            (len(self._states) + 1, len(self.observation_to_index_mapping))
        assert self.start_state not in self._states
        assert self.final_state not in self._states

    def __iter__(self):
        return enumerate(self._states)
    
    def set_unknown_word_symbol(self, symbol):
        assert symbol in self.observation_to_index_mapping
        self.unknown_word_symbol = symbol

    def get_transition_probability(self, current_state, next_state):
        if current_state == self.start_state:
            j = self.state_to_index_mapping[next_state]
            return self.initial_probabilities[j]
        i = self.state_to_index_mapping[current_state]
        j = self.state_to_index_mapping[next_state]
        return self._transition_matrix[i, j]

    def get_emission_probability(self, state, observation_symbol):
        if observation_symbol not in self. observation_to_index_mapping:
            print("Unknown symbol:", observation_symbol)
            if self.unknown_word_symbol is not None:
                observation_symbol = self.unknown_word_symbol
            else:
                raise RuntimeError("Unknown Word symbol not set")
        i = self.state_to_index_mapping[state]
        j = self.observation_to_index_mapping[observation_symbol]
        return self._emission_matrix[i, j]
    
    def get_observation_symbol_index(self, observation_symbol):
        try:
            return self.observation_to_index_mapping[observation_symbol]
        except KeyError:
            if self.unknown_word_symbol is not None:
                return self.observation_to_index_mapping[self.unknown_word_symbol]

    def get_state_index(self, state):
        return self.state_to_index_mapping[state]

    @property
    def states(self):
        return self._states
    
    @property
    def a(self):
        return self._transition_matrix
    
    @property
    def b(self):
        return self._emission_matrix
