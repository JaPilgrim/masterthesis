import hashlib
import json
import random as python_random
import numpy as np
import keras
import optuna
import pandas as pd
import tensorflow as tf
import yaml
from keras.callbacks import EarlyStopping
from numpy.random import seed
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from back_classes.sentence_classifier import LSTMDataset
from back_classes.tokenizer_class import TokenizerClass
from back_classes.utils import *

config = {
    'method': 'bayes',
    'name': 'sweep-first',
    'metric': {
        'goal': 'maximize',
        'name': 'AUC'
    },
    'parameters': {
        'compile_metric': 'accuracy',
        'dense_activation': 'softsign',
        'ES_metric': 'val_loss',
        'test_share': 0.2,
        'eval_share': 0.2,
        'LSTM_dropout': 0.05,
        'hidden_layer_size': 128,
        'epochs': 10,
        'learning_rate': 0.0005,
        'dataset': 'posfilter_linklabeled'
    },
}


def objective(trial):
    lstm_dataset = LSTMDataset()
    lstm_dataset.load_datasets(f"""data/data_files/final_datasets/{
            trial.suggest_categorical('dataset',
            ['linklabeled','posfilter_linklabeled','posfilter_standardlabeled'])}.csv"""
                              )
    lstm_dataset.downsample_classdist_loaded_df()
    lstm_dataset.downsample_whole_df(50000)
    lstm_dataset.split_off_testset_internal(trial.suggest_categorical('test_share', [0.1, 0.2]))
    lstm_dataset.whole_df_to_preprocessed_train_val(
        trial.suggest_categorical('val_share', [0.1, 0.2]) / (1 - trial.params['test_share']))

    model = keras.models.Sequential()
    model.add(layers.Embedding(lstm_dataset.tokenizer_class.num_unique_words,32,input_length=32))
    model.add(layers.LSTM(trial.suggest_categorical('hidden_layers', [56,128,256])))
    model.add(layers.Dense(1,activation='softsign'))
    model.summary()

    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=0.0005)

    metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    
    es = EarlyStopping("val_loss", mode='min', verbose=1, patience=2)

    history = model.fit(lstm_dataset.train_padded,
                        lstm_dataset.train_df['label'],
                        batch_size=32,
                        validation_data=(lstm_dataset.val_padded, lstm_dataset.val_df['label']),
                        callbacks=[es],
                        epochs=10)

    test_acc,test_AUC,transfer_acc,transfer_AUC = lstm_dataset.evaluate_model(model)

    return test_acc,test_AUC,transfer_acc,transfer_AUC

study = optuna.create_study(directions=['maximize','maximize','maximize','maximize'])
study.optimize(objective, n_trials=30, timeout=300)

print("Number of finished trials: ", len(study.trials))
