import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer

from utilities.utils import *


class TokenizerClass():
    def __init__(self, stopwords=stopwords.words("german"), tokenizer=Tokenizer()) -> None:
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.num_unique_words = 0

    def run(self):
        pass

    def decode_sequence(self, sequence):
        return " ".join([self.tokenizer.reverse_word_index.get(idx, "?") for idx in sequence])

    def fit_tokenizer_on_train(self, train_sentence_list: list[str]) -> int:
        """Fits the tokenizer on a sentence list, that should be the training data.

        Args:
            train_sentence_list (list[str]): List of sentences

        Returns:
            int: Number of unique words
        """
        self.tokenizer = Tokenizer(num_words=self.num_unique_words)
        self.tokenizer.fit_on_texts(texts=train_sentence_list)
        return self.num_unique_words

    def set_unique_words(self, sentence_list: pd.Series) -> int:
        """Counts number of unique words over list of sentences, stores it to class variable 
        num_unique_words & returns it.

        Args:
            sentence_list (list[str]): _description_

        Returns:
            int: _description_
        """
        counter = counter_word(sentence_list)
        self.num_unique_words = len(counter)
        return self.num_unique_words

    def remove_stopwords_series(self, sentence_list: pd.Series) -> pd.Series:
        cleaned_sentence_list = sentence_list.map(remove_stopwords)
        return cleaned_sentence_list
