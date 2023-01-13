import datetime

from keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tokenizer_class import TokenizerClass
from utils import *

#TODO: Funnktion die sich join zwischen POS von claim & nonclaim anschaut und die matches aus nonclaim rauslöscht


class SentenceClassifier():
    def __init__(self,
                 df,
                 tokenizer_class: TokenizerClass,
                 model="Default",
                 test_share=0.1,
                 text_column="text",
                 padding_length=32):
        self.model = model
        self.unique_words = None
        self.tokenizer_class = tokenizer_class
        self.test_share = test_share
        self.whole_df = df
        self.train_df = pd.DataFrame({
            'text': pd.Series(dtype='str'),
            'target': pd.Series(dtype='bool'),
            'padded': pd.Series(dtype='object')
        })
        self.test_df = self.train_df.copy()
        self.val_df = self.train_df.copy()
        self.unique_words = 0
        self.text_column = text_column
        self.padding_length = padding_length
        self.train_array = []
        self.val_array = []

    def preprocess_train_val(self, text_column="text"):
        self.whole_df[text_column] = self.tokenizer_class.remove_stopwords_series(
            self.whole_df[text_column])
        self.tokenizer_class.set_unique_words(self.whole_df[text_column])
        self.train_df, self.val_df = self.split_val_train(self.whole_df)
        self.tokenizer_class.fit_tokenizer_on_train(self.train_df[text_column].tolist())
        self.train_padded = self.raw_text_to_padded_sequences(self.train_df[text_column])
        self.val_padded = self.raw_text_to_padded_sequences(self.val_df[text_column])
        return self.train_df, self.val_df

    def raw_text_to_padded_sequences(self, text_list) -> pd.Series:
        sequence_list = self.tokenizer_class.tokenizer.texts_to_sequences(text_list)
        padded_list = pad_sequences(sequence_list, maxlen=32, padding="post", truncating="post")
        return padded_list

    def predict_target(self, text_list) -> list[bool]:
        self.test_data = list(text_list)
        test_padded = self.raw_text_to_padded_sequences(self.test_data)
        self.test_prediction = list(self.model.predict(test_padded))

        return self.test_prediction

    def save_current_test_as_csv(self):
        now = datetime.datetime.now()
        df_to_store = pd.DataFrame({'text': self.test_data, 'prediction': self.test_prediction})
        df_to_store.to_csv("results_" + str(self.model) + str(now.hour) + str(now.minute))

    def default_model(self):

        model = keras.models.Sequential()
        model.add(layers.Embedding(self.tokenizer_class.num_unique_words, 32, input_length=32))
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
        if (self.model == "Default"):
            self.model = self.default_model()
        es = EarlyStopping('val_loss', mode='min', verbose=1, patience=2)
        self.model.summary()
        self.model.fit(self.train_padded,
                       self.train_df["target"],
                       validation_data=(self.val_padded, self.val_df["target"]),
                       callbacks=[es],
                       **kwargs)
        pass

    def split_val_train(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, val_df = split_train_test(df, self.test_share)
        return train_df, val_df
