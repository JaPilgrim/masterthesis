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
                 model: (keras.Model | None),
                 test_share=0.1,
                 text_column="text",
                 padding_length=32):
        if (model):
            self.model = model
        else:
            self.model = self.default_model()
        self.unique_words = None
        self.tokenizer_class = tokenizer_class
        self.test_share = test_share
        self.whole_df = df
        self.train_df = pd.DataFrame
        self.val_df = pd.DataFrame
        self.unique_words = 0
        self.text_column = text_column
        self.padding_length = padding_length
        self.train_array = []
        self.val_array = []

    def run_preprocessing(self, text_column="text"):
        self.whole_df[text_column] = self.tokenizer_class.remove_stopwords_series(
            self.whole_df[text_column])
        self.tokenizer_class.set_unique_words(self.whole_df[text_column])
        self.train_df, self.val_df = self.split_val_train(self.whole_df)
        self.tokenizer_class.fit_tokenizer_on_train(self.train_df[text_column].tolist())
        self.train_df["sequences"] = self.tokenizer_class.tokenizer.texts_to_sequences(
            self.train_df[text_column])
        self.val_df["sequences"] = self.tokenizer_class.tokenizer.texts_to_sequences(
            self.val_df[text_column])
        self.train_df["padded"] = pad_sequences(self.train_df["sequences"],
                                                maxlen=self.padding_length,
                                                padding="post",
                                                truncating="post")
        self.val_df["padded"] = pad_sequences(self.val_df["sequences"],
                                              maxlen=self.padding_length,
                                              padding="post",
                                              truncating="post")

        #TODO Option to pass specific padder

    def default_model(self):
        model = keras.models.Sequential()
        model.add(layers.Embedding(self.unique_words, 32, input_length=32))
        # The layer will take as input an integer matrix of size (batch, input_length),
        # and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
        # Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.
        model.add(layers.LSTM(64, dropout=0.1))
        model.add(layers.Dense(1, activation="sigmoid"))
        #TODO try out others than sigmoid
        #min 2nd layer or more
        model.summary()
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
        optim = keras.optimizers.Adam(learning_rate=0.001)
        metrics = ["accuracy"]
        model.compile(loss=loss, optimizer=optim, metrics=metrics)
        return model

    def train_model(self, kwargs={"epochs": 20}):
        self.model.summary()
        self.model.fit(self.train_df["padded"],
                       self.train_df["target"],
                       validation_data=(self.val_df["padded"], self.val_df["target"]),
                       **kwargs)
        pass

    def add_keras_layer(self):
        pass

    def run_training(self):
        pass

    def split_val_train(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, val_df = split_train_test(df, self.test_share)
        return train_df, val_df

    def tokenize_data(self, train_list: list[str],
                      val_list: list[str]) -> tuple[list[int], list[int]]:
        pass
