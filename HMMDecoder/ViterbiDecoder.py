import numpy as np

from math import log10
import random
import logging


def log(x):
    if x == 0.0:
        return -float('inf')
    try:
        return log10(x)
    except ValueError:
        print(x)
        raise 


class ViterbiDecoder:
    def __init__(self, hmm, start_symbol, stop_symbol):
        self.HMM = hmm
        self.state_to_index_mapping = {state: i for i, state in self.HMM}
        self.index_to_state_mapping = {i: state for i, state in self.HMM}
        self.START = start_symbol
        self.STOP = stop_symbol
        self.trellis_matrix = np.empty(0)
        self.backpointer_matrix = np.empty(0)
    
    def decode(self, observation):
        self.trellis_matrix = np.zeros(((len(self.HMM.states),
                                         len(observation) + 1)))
        self.backpointer_matrix = np.zeros(((len(self.HMM.states),
                                             len(observation) + 1)))
        observation = [
            self.HMM.get_observation_symbol_index(observation_symbol)
            for observation_symbol in observation
            ]
        for j, state in enumerate(self.HMM):
            self.trellis_matrix[j, 0] =\
                (
                    log(self.HMM.get_transition_probability(self.START, state))
                    + log(self.HMM.b[j, observation[0]])
                )
        for t, observation_symbol in enumerate(observation[1:], 1):
            # logging.info("Calculate column %s", t)
            for j, state2 in self.HMM:
                max_transition_value = -float('inf')
                max_state_index = -1
                for i, state1 in self.HMM:
                    transition_value = (
                        self.trellis_matrix[j][t-1]
                        + log(self.HMM.a[i, j])
                        + log(self.HMM.b[j, observation_symbol])
                        )
                    if transition_value > max_transition_value:
                        max_transition_value = transition_value
                        max_state_index = state1_index
                self.trellis_matrix[j][t] = max_transition_value
                self.backpointer_matrix[j][t] = max_state_index
        for i, state in self.HMM:
            self.trellis_matrix[i][-1] = self.trellis_matrix[i][-2] + log(self.HMM.get_transition_probability(state, self.STOP))
            self.backpointer_matrix[j][-1] = state_index
        
        state_sequence = []
        start_index = -1
        max_probability = -float('inf')
        for i, state in self.HMM:
            if self.trellis_matrix[state_index][-1] > max_probability:
                max_probability = self.trellis_matrix[i][-1]
                start_index = state_index
        if start_index == -1:
            start_index = random.choice(list(self.state_to_index_mapping.values()))
            logging.critical("Zero probability observation")
        state_sequence.append(start_index)
        for i in range(len(observation)-1, 0, -1):
            if self.backpointer_matrix[int(state_sequence[-1]), i] == -1:
                self.backpointer_matrix[int(state_sequence[-1]), i] = random.choice(list(self.state_to_index_mapping.values()))
                logging.critical("Zero probability observation")
            state_sequence.append(self.backpointer_matrix[int(state_sequence[-1]), i])
        
        state_sequence.reverse()
        return [self.index_to_state_mapping[state_index] for state_index in state_sequence]
        
        
