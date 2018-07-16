import numpy as np
from utility.utility import create_vocabulary, encode_text_sets
import pickle

class LanguageEncoder(object):
    def __init__(self, maximum_vocabulary_size=10000, lower=True, forbidden_characters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        self._maximum_vocabulary_size = maximum_vocabulary_size
        self._forbidden_characters = forbidden_characters
        self._lower = lower

        translate_forbidding_characters_dict = dict((c, " ") for c in self._forbidden_characters)
        self._translate_forbidding_characters_map = str.maketrans(translate_forbidding_characters_dict)

    def _filter(self, text):
        if self._lower:
            return text.lower().translate(self._translate_forbidding_characters_map)
        else:
            return text.translate(self._translate_forbidding_characters_map)

    def _build_vocabulary(self, text_sets):
        self._word_index_map, self._index_word_map = create_vocabulary(self._maximum_vocabulary_size, text_sets)
        self._vocabulary_size = len(self._word_index_map)

    def fit(self, x):
        # x: list of lists containing the texts

        # clean texts
        for i, texts in enumerate(x):
            for j, text in enumerate(texts):
                x[i][j] = self._filter(text)

        self._build_vocabulary(x)

    def transform(self, x, oh_encode=False):
        # clean texts
        for i, texts in enumerate(x):
            for j, text in enumerate(texts):
                x[i][j] = self._filter(text)

        return encode_text_sets(x, self._word_index_map)

    def transform_word(self, word):
        if word in self._word_index_map:
            return self._word_index_map[word]
        else:
            raise ValueError("word not in the dictionary")

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)