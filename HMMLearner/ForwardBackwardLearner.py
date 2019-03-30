from HiddenMarkovModel import HiddenMarkovModel as HMM

import numpy as np

class ForwardBackwardAlgorithm:
    def __init__(self, possible_states, possible_observation_symbols, start_state, final_state):
        self.states = tuple(possible_states)
        self.states_with_final_state = tuple(list(self.states) + [final_state])
        self.N = len(self.states)
        self.initial_probabilities = np.empty(len(possible_states))
        self.final_state = final_state
        self.final_state_index = self.states_with_final_state.index(self.final_state)
        self.start_state = start_state
        self.current_observation = None
        self.a = np.empty(0)
        self.b = np.empty(0)
        self.scaling_coefficients = np.empty(0)
        self.alpha = np.empty(0)
        self.beta = np.empty(0)
        self.zeta = np.empty(0)
        self.gamma = np.empty(0)
        self.all_observation_symbols = possible_observation_symbols
        self.initialised = False
    
    def show(self, k):
        print('\n'*5)
        print(self.states)
        print(self.all_observation_symbols)
        print(self.current_observation)
        print()
        print("Initial Probabilities")
        print(self.initial_probabilities)
        print()
        print("Transition Matrix A:")
        print(self.a)
        print()
        print("Emission Matrix B:")
        print(self.b)
        print()
        print("Forward Probabilities")
        print(self.alpha[k])
        print()
        print("Backward Probabilities")
        print(self.beta[k])
        print()
        print("Zeta Matrices")
        print(self.zeta[k])
        print()
        print("Gamma Matrix")
        print(self.gamma[k])

    def learn_hmm(self, observations):
        self.observations = observations
        self.initialise()
        for counter in range(10):
            self.e_step()
            self.m_step()
            print("\rIteration {} finished".format(counter+1), end='')
        print()
        #print(self.a)
        #print(self.b)
        
        #transition_matrix = np.empty((self.N+2, self.N+2))
        #transition_matrix[:-2, :-1] = self.a
        #transition_matrix[-2] = np.zeros(self.N+2)
        #transition_matrix[-1,:-2] = self.initial_probabilities
        #transition_matrix[:, -1] = np.zeros(self.N+2)
        #transition_matrix[-1, -2] = 0.0
        return HMM(
            self.start_state, self.final_state, self.states,
            self.all_observation_symbols, self.initial_probabilities, 
            self.a, self.b
            )

    def initialise(self):
        for k, observation in enumerate(self.observations):
            transformed_observation = []
            for observation_symbol in observation:
                transformed_observation.append(self.all_observation_symbols[observation_symbol])
            self.observations[k] = transformed_observation
        if not self.initialised:
            self.random_initialise_a_b()
        self.initialised = True

    def random_initialise_a_b(self):
        self.a = np.ones((self.N, self.N+1))
        for i, _ in enumerate(self.a):
            self.a[i] = np.random.dirichlet(self.a[i], 1)
        m = len(self.all_observation_symbols.keys())
        self.b = np.ones((self.N+1, m))
        for i, _ in enumerate(self.b):
            self.b[i] = np.random.dirichlet(self.b[i], 1)
        self.b[self.N] = np.zeros(m)
    
    def initialise_from_hmm(self, hmm):
        assert hmm.a.shape == (self.N, self.N+1)
        assert hmm.b.shape == (self.N+1, len(self.all_observation_symbols.keys()))
        assert len(hmm.initial_probabilities) == self.N
        self.a = hmm.a
        self.b = hmm.b
        self.initial_probabilities = hmm.initial_probabilities
        self.initialised = True
        
        
    
    def e_step(self):
        K = len(self.observations)
        self.scaling_coefficients = np.empty(K, dtype=object)
        self.alpha = np.empty(K, dtype=object)
        self.beta = np.empty(K, dtype=object)
        self.zeta = np.empty(K, dtype=object)
        self.gamma = np.empty(K, dtype=object)
        print()
        for k, observation in enumerate(self.observations):
            print("\rProcessing observation {} ({}%)".format(k, round(100*((k+1)/len(self.observations)), 2)), end='')
            self.T = len(observation)
            self.current_observation = observation
            self.scaling_coefficients[k] = np.zeros((self.T))
            self.alpha[k] = np.zeros((self.N, self.T))
            self.beta[k] =  np.zeros((self.N, self.T))
            self.zeta[k] = np.zeros((self.T, self.N, self.N+1))
            self.gamma[k] = np.zeros((self.T, self.N))
            self.calculate_forward_probabilities(k)
            #print("Forward done for {}".format(k))
            self.calculate_backward_probabilities(k)
            #print("Backward done for {}".format(k))
            self.calculate_zeta(k)
            #print("Zeta done for {}".format(k))
            self.calculate_gamma(k)
            #print("Gamma done for {}".format(k))
            #print()
            # self.show(k)
        print()
        #print(self.alpha)
        #print()
    
    
    def m_step(self):
        deltas = []
        # Updata a
        for i, state_i in enumerate(self.states):
            for j, state_j in enumerate(self.states_with_final_state):
                old_aij = self.a[i, j]
                self.a[i, j] = np.sum(zeta_k[:, i, j].sum() for zeta_k in self.zeta)/np.sum(gamma_k[:, i].sum() for gamma_k in self.gamma)
                assert not np.isnan(self.a[i, j])
                deltas.append(abs(old_aij-self.a[i, j]))
        
        # Update b
        numerator_table = {index: np.zeros(self.N) for index in self.all_observation_symbols.values()}
        for k, observation in enumerate(self.observations):
            for t, observation_symbol in enumerate(observation):
                numerator_table[observation_symbol] += self.gamma[k][t, :]
        for v_k in self.all_observation_symbols.values():
            self.b[:-1, v_k] = numerator_table[v_k]
        for j, state in enumerate(self.states):
            denominator = np.sum(gamma_k[:, j].sum() for gamma_k in self.gamma)
            self.b[j, :] /= denominator
            assert all(not np.isnan(value) for value in self.b[j, :])

    def calculate_zeta(self, k):
        observation_probability = self.alpha[k][:, -1].dot(self.a[:, -1])
        for t in range(self.T-1):
            for i, state_i in enumerate(self.states):
                for j, state_j in enumerate(self.states):
                    self.zeta[k][t, i, j] = self.alpha[k][i, t]*self.a[i, j]*self.b[j, self.current_observation[t+1]]*self.beta[k][j, t+1]
                    self.zeta[k][t, i, j] /= observation_probability

        for i, state_i in enumerate(self.states):
            j = self.final_state_index
            self.zeta[k][self.T-1, i, j] = self.alpha[k][i, self.T-1]*self.a[i, j]
            self.zeta[k][self.T-1, i, j] /= observation_probability

    
    def calculate_gamma(self, k):
        for t in range(self.T):
            for i, _ in enumerate(self.states):
                self.gamma[k][t, i] = self.zeta[k][t, i, :].sum()

    def calculate_forward_probabilities(self, k):
        # Initialisation
        for i, _ in enumerate(self.states):
            self.alpha[k][i, 0] = self.initial_probabilities[i] * self.b[i, self.current_observation[0]]
        self.scaling_coefficients[k][0] = self.alpha[k][:, 0].sum()
        self.alpha[k][:, 0] /= self.scaling_coefficients[k][0]
        
        # Recursion
        for t in range(1, self.T):
            for j, _ in enumerate(self.states):
                self.alpha[k][j, t] = (self.alpha[k][:, t-1].dot(self.a[:, j]))*self.b[j, self.current_observation[t]]
                #if self.alpha[k][j, t] == 0:
                    #print(self.alpha[k][:, t-1], self.a[:, j], self.b[j, self.current_observation[t]])
                    #print('\n'*3)
            self.scaling_coefficients[k][t] = self.alpha[k][:, t].sum()
            try:
                assert self.scaling_coefficients[k][t] > 0
            except AssertionError:
                print(self.alpha[k])
                print(t)
                raise AssertionError
            self.alpha[k][:, t] /= self.scaling_coefficients[k][t]
        
        # Termination step omitted

    def calculate_backward_probabilities(self, k):
        # Initialisation
        for i, state in enumerate(self.states):
            self.beta[k][i, self.T-1] = self.a[i, self.final_state_index]*self.b[i, self.current_observation[self.T-1]]
            self.beta[k][i, self.T-1] /= self.scaling_coefficients[k][self.T-1]
        
        # Recursion
        for t in range(self.T-2, -1, -1):
            for i, _ in enumerate(self.states):
                self.beta[k][i, t] = (self.beta[k][:, t+1]*self.a[i, :-1]*self.b[:-1, self.current_observation[t+1]]).sum()
                self.beta[k][i, t] /= self.scaling_coefficients[k][t]
        
        # Termination step omitted
