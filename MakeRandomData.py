import numpy as np


class MakeRandomData:
    def __init__(self, number_of_states, number_of_observation_symbols):
        self.number_of_states = number_of_states
        self.STOP = self.number_of_states - 1
        self.number_of_observation_symbols = number_of_observation_symbols
        self.a = np.ones((self.number_of_states, self.number_of_states+1))
        self.b = np.ones((self.number_of_states, self.number_of_observation_symbols))
        self.initial_probabilities = np.random.dirichlet(np.ones(self.number_of_states), 1)[0]
        for state in range(self.number_of_states):
            self.a[state] = np.random.dirichlet(self.a[state], 1)[0]
            self.b[state] = np.random.dirichlet(self.b[state], 1)[0]
        self.states = [state for state in range(self.number_of_states)]
        self.vocabulary = {i: i for i in range(self.number_of_observation_symbols)}
    
    def generate_sequence(self):
        state_sequence = []
        observation = []
        next_state = np.random.choice(self.number_of_states, p=self.initial_probabilities)
        while next_state != self.number_of_states:
            current_state = next_state
            state_sequence.append(current_state)
            observation.append(self.get_emission(current_state))
            next_state = self.get_next_state(current_state)
        return list(zip(observation, state_sequence))
    
    def get_next_state(self, state):
        return np.random.choice(self.number_of_states+1, p=self.a[state])
    
    def get_emission(self, state):
        return np.random.choice(self.number_of_observation_symbols, p=self.b[state])
    
    def get_tagged_training_data(self, n):
        training_data = []
        for _ in range(n):
            training_data.append(self.generate_sequence())
        return training_data
    
    def get_untagged_training_data(self, n):
        training_data = []
        for _ in range(n):
            observation = self.generate_sequence()
            training_data.append([observation_symbol for observation_symbol, state in observation])
        return training_data
    
    def get_test_data(self, n):
        return self.get_tagged_training_data(n)
    
