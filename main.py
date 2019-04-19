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
from HMMLearner.ExampleLerner import ExampleEMLearner
from HMMEvaluation import Evaluator
from HMMDecoder import Decoder
from DecoderTester import Test
from MakeTaggingData import MakeTaggingData
from MakeRandomData import MakeRandomData


if __name__ == '__main__':
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    logging.basicConfig(level=logging.INFO)
    START = "<s>"
    STOP = "</s>"
    UNK = "UNK"
    ALL_TAGS = tuple(sorted(load('help/tagsets/upenn_tagset.pickle').keys()))
    create_new_model = True
    lowercase = False
    
    logging.info("Preparing data")
    # data = MakeTaggingData(3, UNK, lowercase=lowercase)
    data = MakeRandomData(6, 15)
    tagged_training_sentences = data.get_tagged_training_data(20)
    untagged_training_sentences = data.get_untagged_training_data(50000)
    

    logging.info("Learn Model")
    if create_new_model:
        mle_learner = KneserNeyMLELearner(data.states, data.vocabulary, START, STOP)
        baum_welch_learner = ForwardBackwardAlgorithm(data.states, data.vocabulary, START, STOP)
        example_learner = ExampleEMLearner(data.states)
        # my_hmm = mle_learner.learn_hmm(tagged_training_sentences)
        # baum_welch_learner.initialise_from_hmm(my_hmm)
        #my_hmm = baum_welch_learner.learn_hmm(untagged_training_sentences, iterations=20)
        my_hmm = None
        example_learner.learn_hmm(untagged_training_sentences)
        
        with open("HMM_model.dat", "wb") as hmm_file:
            pickle.dump(my_hmm, hmm_file)
    else:
        with open("HMM_model.dat", "rb") as hmm_file:
            my_hmm = pickle.load(hmm_file)
    # my_hmm.set_unknown_word_symbol(UNK)

    logging.info("Instantiate decoder")
    # decoder = Decoder(my_hmm, START, STOP)
    decoder = example_learner
    #logging.info("Check decoder functionality")
    #print(decoder.decode(["i", "went", "to", "town", "."]))
    
    # time.sleep(3)

    logging.info("Instantiate tester")
    tester = Test(data.get_test_data(1000))
    logging.info("Run tests")
    result = tester.test_decoder(decoder, lowercase=lowercase)

    logging.info("Output results")
    print()
    print(result)
    print()
