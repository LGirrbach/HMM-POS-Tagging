from nltk.corpus import treebank
from nltk.corpus import gutenberg

from sklearn.model_selection import train_test_split

import random


class MakeTaggingData:
    def __init__(self, unknown_word_threshold, unknown_word_symbol, lowercase=True, shuffle=False):
        self.unknown_word_threshold = unknown_word_threshold
        self.unknown_word_symbol = unknown_word_symbol
        self.is_lowercase = lowercase
        self.shuffle = shuffle
        self.word_counts = dict()
        self.vocabulary = []
        self.states = []
        self.tagged_sentences = list(treebank.tagged_sents())
        self._tagged_sentences_train, self._tagged_sentences_test =\
            train_test_split(self.tagged_sentences)
        self._untagged_sentences = list(gutenberg.sents())
        if self.is_lowercase:
            self._tagged_sentences_train = [
                [(word.lower(), tag) for word, tag in sentence]
                for sentence in self._tagged_sentences_train
                ]
            self._tagged_sentences_test = [
                [(word.lower(), tag) for word, tag in sentence]
                for sentence in self._tagged_sentences_train
                ]
            self._untagged_sentences = [
                [word.lower() for word in sentence]
                for sentence in self._untagged_sentences
                ]
        self.data_prepared = False
        self.prepare_data()
    
    def prepare_data(self):
        self.word_counts.clear()
        all_words = []
        all_states = []
        #for sentence in self._untagged_sentences:
            #all_words += sentence
        for sentence in self._tagged_sentences_train + self._tagged_sentences_test:
            for word, state in sentence:
                all_words.append(word)
                all_states.append(state)
        for word in all_words:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1
        self.vocabulary = [
            word for word in all_words
            if self.unknown_word_threshold <= self.word_counts.get(word, 0)
            ]
        self.vocabulary.append(self.unknown_word_symbol)
        self.vocabulary = set(self.vocabulary)
        self.vocabulary = {word: i for i, word in enumerate(sorted(self.vocabulary))}
        self.states = list(sorted(set(all_states)))
        self._untagged_sentences = [
            [self.get_word_symbol(word) for word in sentence]
            for sentence in self._untagged_sentences
            ]
        self._tagged_sentences_train = [
            [(self.get_word_symbol(word), tag) for word, tag in sentence]
            for sentence in self._tagged_sentences_train
            ]
        self._tagged_sentences_test = [
            [(self.get_word_symbol(word), tag) for word, tag in sentence]
            for sentence in self._tagged_sentences_test
            ]
        if self.shuffle:
            random.shuffle(self._untagged_sentences)
            random.shuffle(self._tagged_sentences_train)
            random.shuffle(self._tagged_sentences_test)
        self.data_prepared = True
    
    def get_word_symbol(self, word):
        if self.unknown_word_threshold > self.word_counts.get(word, 0):
            return self.unknown_word_symbol
        else:
            return word
    
    def get_tagged_training_data(self, n=0):
        assert self.data_prepared
        if n > 0:
            return self._tagged_sentences_train[:n]
        return self._tagged_sentences_train
    
    def get_untagged_training_data(self, n=0):
        assert self.data_prepared
        if n > 0:
            return self._untagged_sentences[:n]
        return self._untagged_sentences
    
    def get_test_data(self, n=0):
        assert self.data_prepared
        if n > 0:
            return self._tagged_sentences_test[:n]
        return self._tagged_sentences_test
    
    
