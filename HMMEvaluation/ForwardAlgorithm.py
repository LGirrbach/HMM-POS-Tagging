import numpy as np

from math import log, exp
import logging


class ForwardAlgorithm:
    def __init__(self, hmm, start_symbol, stop_symbol):
        self.HMM = hmm
        self.state_to_index_mapping = {state: i for i, state
                                       in enumerate(self.HMM)}
        self.START = start_symbol
        self.STOP = stop_symbol
        self.trellis_matrix = None

    def estimate_observation_likelihood(self, observation):
        # print(observation, len(observation))
        logging.info("Create Trellis Matrix")
        logging.info("HMM has %s states", len(self.HMM.get_states()))
        self.trellis_matrix = np.zeros(((len(self.HMM.get_states()),
                                         len(observation))))
        logging.info("Calculate First Column")
        for state in self.HMM:
            state_index = self.state_to_index_mapping[state]
            self.trellis_matrix[state_index][0] =\
                (self.HMM.get_transition_probability(self.START, state)
                 * self.HMM.get_emission_probability(state, observation[0])
                 )
            print(self.trellis_matrix[state_index][0])

        logging.info("Calculate following columns")
        for t, observation_symbol in enumerate(observation[1:], 1):
            logging.info("Calculate column %s", t)
            for state2 in self.HMM.get_states():
                if state2 == self.STOP or state2 == self.START:
                    continue
                for state1 in self.HMM.get_states():
                    if state1 == self.STOP or state1 == self.START:
                        continue
                    state1_index = self.state_to_index_mapping[state1]
                    state2_index = self.state_to_index_mapping[state2]
                    self.trellis_matrix[state2_index][t] +=\
                        (
                            self.trellis_matrix[state1_index][t-1]
                            * self.HMM.get_transition_probability(state1, state2)
                            * self.HMM.get_emission_probability(state2, observation_symbol)
                        )
                
            print(self.trellis_matrix)
                    
        result = 0
        for state in self.HMM:
            state_index = self.state_to_index_mapping[state]
            result += self.trellis_matrix[state_index][-1] * self.HMM.get_transition_probability(state, self.STOP)
        return result
                                              
            
            
        
