from hmmlearn import hmm
import numpy as np


class ExampleEMLearner:
    def __init__(self, states):
        self.states = states
        self.hmm = hmm.MultinomialHMM(n_components=len(self.states))
    
    def learn_hmm(self, data, iterations=20):
        data = tuple([np.asarray([[observation] for observation in sample]) for sample in data])
        lengths = np.asarray([len(sample) for sample in data])
        data = np.concatenate(data)
        self.hmm = self.hmm.fit(data, lengths=lengths)
    
    def decode(self, sample):
        sample = np.asarray([[observation] for observation in sample])
        prob, labels = self.hmm.decode(np.asarray(sample), algorithm='viterbi')
        return list(labels)
