import logging
import time
import pickle
import random

from nltk.data import load
from nltk.corpus import treebank as training_corpus
from nltk.corpus import conll2000 as test_corpus
from nltk.corpus import gutenberg as learn_corpus

import numpy as np

from HMMLearner import Learner
from HMMLearner.ForwardBackwardLearner import ForwardBackwardAlgorithm
from HMMLearner.KneserNeyMLELearner import KneserNeyMLELearner
from HMMEvaluation import Evaluator
from HMMDecoder import Decoder
from DecoderTester import Test
from MakeTaggingData import MakeTaggingData


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    logging.basicConfig(level=logging.INFO)
    training_sentences = training_corpus.tagged_sents()
    test_sentences = test_corpus.tagged_sents()
    START = "<s>"
    STOP = "</s>"
    UNK = "UNK"
    ALL_TAGS = tuple(sorted(load('help/tagsets/upenn_tagset.pickle').keys()))
    create_new_model = True
    lowercase = True
    
    logging.info("Preparing data")
    data = MakeTaggingData(3, UNK, lowercase=lowercase)
    training_sentences = data.get_tagged_training_data()

    logging.info("Learn Model")
    if create_new_model:
        # learner = ForwardBackwardAlgorithm(ALL_TAGS, np.random.dirichlet(np.ones(len(ALL_TAGS)), 1)[0], START, STOP)
        learner = KneserNeyMLELearner(data.states, data.vocabulary, START, STOP)
        my_hmm = learner.learn_hmm(training_sentences)
        with open("HMM_model.dat", "wb") as hmm_file:
            pickle.dump(my_hmm, hmm_file)
    else:
        with open("HMM_model.dat", "rb") as hmm_file:
            my_hmm = pickle.load(hmm_file)
    my_hmm.set_unknown_word_symbol(UNK)

    logging.info("Instantiate decoder")
    decoder = Decoder(my_hmm, START, STOP)
    logging.info("Check decoder functionality")
    print(decoder.decode(["i", "went", "to", "town", "."]))
    
    time.sleep(3)

    logging.info("Instantiate tester")
    tester = Test(data.get_test_data())
    logging.info("Run tests")
    result = tester.test_decoder(decoder, lowercase=lowercase)

    logging.info("Output results")
    print()
    print(result)
    print()
