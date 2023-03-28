import datetime
from ast import literal_eval

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from back_classes.tokenizer_class import TokenizerClass
from back_classes.utils import *


class LSTMDataset():
    def __init__(self,
                 val_share=0.2,
                 text_column="text",
                 label_column="label",
                 padding_length=32,
                 tokenizer_class=None):
        if isinstance(tokenizer_class, TokenizerClass):
            self.tokenizer_class = tokenizer_class
        else:
            self.tokenizer_class = TokenizerClass()
        self.val_share = val_share
        self.text_column = text_column
        self.label_column = label_column
        self.padding_length = padding_length
        self.train_padded = []
        self.val_padded = []

        self.whole_df = pd.DataFrame()
        self.loaded_df = pd.DataFrame()
        self.train_df = pd.DataFrame({
            'text': pd.Series(dtype='str'),
            'label': pd.Series(dtype='bool'),
            'padded': pd.Series(dtype='object')
        })
        self.transfer_df = self.train_df.copy()
        self.test_df = self.train_df.copy()
        self.val_df = self.train_df.copy()

    def split_off_testset_external(self, path, val_share=0.2, random_state=2):
        """Takes a portion of loaded_df and uses it as test_df. Updates whole_df accordingly

        Args:
            path (str): path to test df
            val_share (float, optional): Defaults to 0.2.
            random_state (int, optional): Defaults to 2.

        Returns:
            train_df,val_df,whole_df: DFs
        """
        test_df = pd.read_csv(path)
        self.add_test_data(test_df)
        train_df, val_df, whole_df = self.whole_df_to_preprocessed_train_val(
            val_share, random_state)
        return train_df, val_df, whole_df

    def split_off_testset_internal(self, test_split=0.2, val_split=0.2, random_state=2):
        """Runs preprocessing from loaded df to split train test val and preprocessed.

        Args:
            path (str): path to test df
            test_split (float, optional): Defaults to 0.2
            val_share (float, optional): Defaults to 0.2.
            random_state (int, optional): Defaults to 2.

        Returns:
            train_df,val_df,whole_df: DFs
        """
        self.whole_df, self.test_df = split_train_val(self.whole_df,
                                                      test_size=test_split,
                                                      random_state=random_state)
        val_share = val_split / (1 - test_split)
        return self.whole_df, self.test_df

    def whole_df_to_preprocessed_train_val(self,
                                           val_share=0.2,
                                           random_state=2
                                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates run from Whole df to preprocessed train & val dataset

        Args:
            val_share(float): Share of validation set 
            text_column (str, optional): Column name. Defaults to "text".
            label_column (str): Column name for label.
            random_state (Int): Random Seed.

        Returns:
            Train DF: Training Data Set DF
            Test DF: Test Data Set DF
            Whole DF: The Whole Data Set DF
        """
        
        self.whole_df[self.text_column] = self.tokenizer_class.remove_stopwords_series(
            sentence_list=self.whole_df[self.text_column])
        self.tokenizer_class.set_unique_words(sentence_list=self.whole_df[self.text_column])

        self.train_df, self.val_df = split_train_val(self.whole_df,
                                                     test_size=val_share,
                                                     random_state=random_state)

        self.tokenizer_class.fit_tokenizer_on_train(self.train_df[self.text_column].tolist())

        self.train_padded = self.raw_text_to_padded_sequences(self.train_df[self.text_column])
        self.val_padded = self.raw_text_to_padded_sequences(self.val_df[self.text_column])

        return self.train_df, self.val_df, self.whole_df

    def split_test_whole(self, test_size=None, random_state=2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the loaded class into test and whole. 
        
        Args:
            test_size (_type_, optional): Fraction of test. Defaults to None.
            random_state (int, optional): _description_. Defaults to 2.

        Returns:
            tuple[pd.DataFrame,pd.DataFrame]: whole_df and test_df
        """

        self.whole_df, self.test_df = split_train_val(self.loaded_df,
                                                      test_size=test_size,
                                                      random_state=random_state)
        return self.whole_df, self.test_df

    def add_test_data(self, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Adds test-dataframe and adds to test_df class variable.

        Args:
            path (str): Path to the dataframe

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: whole_df and test_df
        """
        self.whole_df = self.loaded_df
        self.test_df = test_df[[self.text_column, self.label_column]]
        return self.whole_df, self.test_df

    def load_datasets(
        self,
        dataset_path: str,
        transferset_path='data/data_files/final_annotated_jannis97.csv',
    ) -> tuple[pd.DataFrame]:
        """Loads a CSV as dataset into class variables. 

        Args:
            path (str): Path to file.
            transferset_path (str): path to annotated transfer dataset.defaults to 
            'data/data_files/final_annotation_106.csv'

        Returns:
            dataframe: loaded df. 
        """
        loaded_df = pd.read_csv(dataset_path, dtype={'text': 'str'})
        self.loaded_df = loaded_df[[self.text_column, self.label_column]]

        self.transfer_df = pd.read_csv(
            transferset_path,
            sep=',',
            dtype={'text': 'str'},
            converters={'pos_list': literal_eval},
        )

        self.test_df = self.transfer_df.copy()
        self.whole_df = self.loaded_df.copy()
        return self.loaded_df, self.transfer_df

    def downsample_classdist_loaded_df(self):
        self.loaded_df = downsample_dataframe_classdist(self.loaded_df)

    def downsample_whole_df(self, size: int, random_state=2):
        """Downsamples size of whole_df to input while equalizing class distribution. 

        Args:
            size (int): Desired sample size.

        Raises:
            ValueError: _description_
        """
        n = int(size / 2)
        if len(self.whole_df) < size:
            raise ValueError('Downsample-size is smaller than actual whole df length.')
        try:
            self.whole_df = self.whole_df.groupby('label').sample(n=n, random_state=2)
        except:
            raise ValueError(
                'One class doesnt hold enough examples for half of specified downsample')

    def raw_text_to_padded_sequences(self, text_list: pd.Series) -> np.array:
        """With tokenizer transforms list of raw text to list of padded sentences (list of ints)

        Args:
            text_list (_type_): List of sentences.

        Returns:
            list: List of tokens (numbers), filled up to padding length with zeroes.
        """
        text_list = [str(s) for s in text_list]
        sequence_list = self.tokenizer_class.tokenizer.texts_to_sequences(text_list)
        padded_list = pad_sequences(sequence_list,
                                    maxlen=self.padding_length,
                                    padding="post",
                                    truncating="post")
        return padded_list

    def evaluate_model(self, model: keras.Model):
        """Evaluates a model based on test & transfer data set performance

        Args:
            model (keras.Model): model to be tested

        Returns:
            float: accuracy
            float: AUC
        """
        test_accuracy, test_AUC,test_f1 = self._predict_and_compute_test(self.test_df, model)
        transfer_accuracy, transfer_AUC,transfer_f1 = self._predict_and_compute_test(self.transfer_df, model)
        return test_accuracy, test_AUC, test_f1, transfer_accuracy, transfer_AUC,transfer_f1

    def _predict_and_compute_test(self, df, model: keras.Model):

        if isinstance(df[self.text_column].iloc[0], str):
            padded_list = np.array(self.raw_text_to_padded_sequences(df[self.text_column]))
        else:
            padded_list = np.array(df[self.text_column])

        df['probabilities'] = np.array(model.predict(padded_list))
        
        accuracy, AUC, f1_score = compute_accuracy_AUC_f1(df[self.label_column],
                                                          df['probabilities'])
        return accuracy, AUC, f1_score

    # def predict_label(self, text_list: list[str]) -> list[bool]:
    #     """Predits the label column. Stores test data & predictino to self.

    #     Args:
    #         text_list (list[str]): List of sentences.

    #     Returns:
    #         list[bool]: List of predictions
    #     """
    #     self.test_data = list(text_list)
    #     test_padded = self.raw_text_to_padded_sequences(self.test_data)
    #     self.test_prediction = list(self.model.predict(test_padded))
    #     return self.test_prediction

    # def save_result_as_csv(self):
    #     """Saves the most recent test results as csv, name is model name + hour/minute.
    #     """
    #     now = datetime.datetime.now()
    #     df_to_store = pd.DataFrame({'text': self.test_data, 'prediction': self.test_prediction})
    #     df_to_store.to_csv("results_" + str(self.model) + str(now.hour) + str(now.minute))

    # def save_test_data_as_csv(self):
    #     now = datetime.datetime.now()

    # def default_model(self):
    #     """Creates & returns the default model from calssification task.

    #     Returns:
    #         keras.model: The compiled
    #     """
    #     model = keras.models.Sequential()
    #     model.add(layers.Embedding(self.tokenizer_class.num_unique_words, 32, input_length=32))
    #     # The layer will take as input an integer matrix of size (batch, input_length),
    #     # and the largest integer (i.e. word index) in the input should be no larger than num_words (vocabulary size).
    #     # Now model.output_shape is (None, input_length, 32), where `None` is the batch dimension.
    #     model.add(layers.LSTM(64, dropout=0.1))
    #     model.add(layers.Dense(1, activation="sigmoid"))
    #     #TODO try out others than sigmoid
    #     #min 2nd layer or more
    #     model.summary()
    #     loss = keras.losses.BinaryCrossentropy(from_logits=False)
    #     optim = keras.optimizers.Adam(learning_rate=0.001)
    #     metrics = ["accuracy"]
    #     model.compile(loss=loss, optimizer=optim, metrics=metrics)
    #     return model

    # def train_model(self, label_column="label", kwargs={"epochs": 20}):
    #     if (self.model == "Default"):
    #         self.model = self.default_model()
    #     es = EarlyStopping('val_loss', mode='min', verbose=1, patience=2)
    #     self.model.summary()

    #     self.model.fit(self.train_padded,
    #                    self.train_df[label_column],
    #                    validation_data=(self.val_padded, self.val_df[label_column]),
    #                    callbacks=[es],
    #                    **kwargs)
    #     pass
