from typing import List

from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tokenizer_class import TokenizerClass
from utils import *


class SentenceClassifier():
    def __init__(self,
                 df,
                 tokenizer_class: TokenizerClass,
                 model=keras.model.Sequential(),
                 test_share=0.1,
                 text_column="text"):
        self.model = model
        self.unique_words = None
        self.tokenizer_class = tokenizer_class
        self.test_share = test_share
        self.whole_df = df
        self.unique_words = 0
        self.text_column = text_column

    def run_preprocessing(self, text_column="text"):
        self.tokenizer_class.set_unique_words(self.whole_df[text_column])
        train_df, val_df = self.split_val_train(self.whole_df)
        self.tokenizer_class.tokenizer.fit_tokenizer_on_train(train_df[text_column])
        train_sequences = self.tokenizer_class.tokenizer.texts_to_sequences(train_df[text_column])
        val_sequences = self.tokenizer_class.tokenizer.texts_to_sequences(val_df[text_column])
        #TODO Padding

    def run_training(self):
        pass

    def split_val_train(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, val_df = train_test_split(df, self.test_share)
        return train_df, val_df

    def tokenize_data(
        self, train_list: list(str), val_list: list(str)) -> tuple[list(int), list(int)]:
        pass
