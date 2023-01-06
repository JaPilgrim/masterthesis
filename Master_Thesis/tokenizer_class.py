import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer

from utils import *


class TokenizerClass():
    def __init__(self, stopwords=stopwords.words("german"), tokenizer=Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.num_unique_words = 0

    def run(self):
        pass

    def decode_sequence(sequence):
        return " ".join([self.tokenizer.reverse_word_index.get(idx, "?") for idx in sequence])

    def fit_tokenizer_on_train(self, train_sentence_list: list(str)) -> int:
        self.tokenizer(num_words=self.num_unique_words)
        self.tokenizer.fit_on_texts(train_sentence_list)
        pass

    def set_unique_words(self, sentence_list: list(str)) -> int:
        counter = counter_word(sentence_list)
        self.num_unique_words = len(counter)

    def remove_stopwords_series(self, sentence_list: pd.Series) -> pd.Series:
        cleaned_sentence_list = sentence_list.text.map(remove_stopwords)
        return cleaned_sentence_list
