import logging
import time
import pickle

from nltk.corpus import treebank as training_corpus
from nltk.corpus import conll2000 as test_corpus
from nltk.corpus import gutenberg as learn_corpus
from nltk.data import load

import numpy as np

from HMMLearner import Learner
from HMMLearner.ForwardBackwardLearner import ForwardBackwardAlgorithm
from HMMEvaluation import Evaluator
from HMMDecoder import Decoder
from DecoderTester import Test


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    training_sentences = training_corpus.tagged_sents()
    test_sentences = test_corpus.tagged_sents()
    START = "<s>"
    STOP = "</s>"
    UNK = "UNK"
    ALL_TAGS = tuple(sorted(load('help/tagsets/upenn_tagset.pickle').keys()))
    create_new_model = True
    
    learn_corpus_sentences = [[word.lower() for word in sentence]
                              for sentence in learn_corpus.sents()]

    logging.info("Learn Model")
    if create_new_model:
        learner = ForwardBackwardAlgorithm(ALL_TAGS, np.random.dirichlet(np.ones(len(ALL_TAGS)), 1)[0], START, STOP)
        my_hmm = learner.learn_hmm(learn_corpus_sentences)
        with open("HMM_model.dat", "wb") as hmm_file:
            pickle.dump(my_hmm, hmm_file)
    else:
        with open("HMM_model.dat", "rb") as hmm_file:
            my_hmm = pickle.load(hmm_file)

    logging.info("Instantiate decoder")
    decoder = Decoder(my_hmm, START, STOP)
    logging.info("Check decoder functionality")
    print(decoder.decode(["some", "folks", "went", "to", "the", "big", "party", "."]))
    time.sleep(3)

    logging.info("Instantiate tester")
    tester = Test(test_sentences[:10])
    logging.info("Run tests")
    result = tester.test_decoder(decoder, lowercase=True)

    logging.info("Output results")
    print()
    print(result)
    print()
