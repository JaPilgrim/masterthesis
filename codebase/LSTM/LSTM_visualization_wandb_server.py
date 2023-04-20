"""To test & compare all 48 datasets in small samples. To be run with wandb agent.
    """
import os
import sys
sys.path.append('/root/projects/jpthesis/keygens/masterthesis/codebase/')
import numpy as np
import pandas as pd
import wandb
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
from utilities.lstm_data_handler import LSTMDataHandler
from utilities.utils import *

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Plot learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Compute and display confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='coolwarm',
                xticklabels=labels,
                yticklabels=labels,
                cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sweep_configuration = {
    'method': 'grid',
    'name': 'realswipe',
    'metric': {
        'goal': 'maximize',
        'name': 'test_acc'
    },
    'parameters': {
        'fileindex': {'max' : 107,'min' :0},
    }
}



def main():
    folder = '../data/data_files/test_samples/7th_RNN25k/'
    # run = wandb.init()
    # fileindex = wandb.config.fileindex
    fileindex = 15
    file_names = os.listdir(folder)

    filename = file_names[fileindex]
    print((filename))
    lstm_dataset = LSTMDataHandler()

    lstm_dataset.load_datasets(f"{folder}{filename}",
                               transferset_path='../data/data_files/annotated_pos_test.csv')

    lstm_dataset.split_off_testset_internal()
    if filename[:3] + 'pos':
        lstm_dataset.transfer_df['text'] = lstm_dataset.transfer_df['pos_string']

    lstm_dataset.whole_df_to_preprocessed_train_val()

    model = keras.models.Sequential()
    model.add(layers.Embedding(lstm_dataset.tokenizer_class.num_unique_words, 32, input_length=32))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(1, activation='softsign'))
    model.summary()

    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=0.0005)

    metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    es = EarlyStopping("val_loss", mode='auto', verbose=1, patience=2)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(lstm_dataset.train_df['label']),
        y=lstm_dataset.train_df['label'],
    )
    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit(lstm_dataset.train_padded,
                        lstm_dataset.train_df['label'],
                        batch_size=32,
                        class_weight=class_weights_dict,
                        validation_data=(lstm_dataset.val_padded, lstm_dataset.val_df['label']),
                        callbacks=[es],
                        epochs=10)



    y_val = lstm_dataset.val_df['label']
    y_val_pred = model.predict(lstm_dataset.val_padded)
    y_val_pred = np.round(y_val_pred).flatten().astype(int)  # Convert probabilities to class labels

    plot_learning_curves(history)
    plot_confusion_matrix(y_val, y_val_pred, labels=[0, 1])  # Assuming your classes are 0 and 1

    test_accuracy, test_precision, test_recall, test_f1_value, transfer_accuracy, transfer_precision, transfer_recall, transfer_f1_value = lstm_dataset.evaluate_model(
        model)

    false_samples = len(lstm_dataset.train_df[lstm_dataset.train_df['label'] == False])
    true_samples = len(lstm_dataset.train_df[lstm_dataset.train_df['label'] == True])
    data_meta_info = filename.split('_')
    log = {
        'articles': data_meta_info[0],
        'pos': data_meta_info[1],
        'resolved': data_meta_info[2],
        'labeling': data_meta_info[3],
        'filter': data_meta_info[4],
        'test_acc': test_accuracy,
        'test_prec': test_precision,
        'test_rec' : test_recall,
        'test_f1': test_f1_value,
        'transfer_acc': transfer_accuracy,
        'transfer_prec': transfer_precision,
        'transfer_rec' : transfer_recall,
        'transfer_f1': transfer_f1_value,
        'class_false': false_samples,
        'class_true': true_samples}
    # wandb.log(log)
    print(log)


main()
