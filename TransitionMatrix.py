import sys
import logging

import numpy as np


class TransitionMatrix:
    def __init__(self, preceding_states, succeeding_states, transitions):
        self.preceding_states = preceding_states
        self.succeeding_states = succeeding_states
        self.preceding_states_index = {state: i for i, state
                                       in enumerate(self.preceding_states)}
        self.succeeding_states_index = {state: i for i, state
                                        in enumerate(self.succeeding_states)}
        self.__transition_matrix = np.zeros((len(self.preceding_states), len(self.succeeding_states)))
        for preceding_state, succeding_state, probability in transitions:
            preceding_state_index =\
                self.preceding_states_index[preceding_state]
            succeeding_state_index =\
                self.succeeding_states_index[succeding_state]
            self.__transition_matrix[preceding_state_index][succeeding_state_index] = probability

    def transition_probability(self, state1, state2):
        if state1 not in self.preceding_states or state2 not in self.succeeding_states:
            # logging.error("Unknown states %s %s", state1, state2)
            raise AssertionError("Invalid states: " + state1 + " " + state2)
        preceding_state_index = self.preceding_states_index[state1]
        succeding_state_index = self.succeeding_states_index[state2]
        return self.__transition_matrix[preceding_state_index][succeding_state_index]


class StateTransitionMatrix(TransitionMatrix):
    def __init__(self, states, transitions):
        super().__init__(states, states, transitions)
    
    def get_states(self):
        return self.preceding_states[:]
