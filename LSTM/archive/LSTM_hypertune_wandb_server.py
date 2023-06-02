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
from tensorflow.keras.callbacks import TensorBoard
import datetime



# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
from utilities.lstm_data_handler import LSTMDataHandler
from utilities.utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep-first',
    'metric': {
        'goal': 'maximize',
        'name': 'AUC'
		},
    'parameters': {
    'eval_share':{'values':[32,64,128]},
    'LSTM_dropout':{'values':[32,64,128]},
    'batch_size':{'values':[32,64,128]},
    'hidden_layer_size':{max:512,min:16},
    'learning_rate':{'values':[0.0001,0.0005,0.001,0.005,0.01]},
    }
     }




def main():
    folder = '../data/data_files/test_samples/5th_try/'
    run = wandb.init()
    # fileindex = wandb.config.fileindex
    fileindex = 2
    file_names = os.listdir(folder)
    print(len(file_names))
    filename = file_names[fileindex]

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')


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

    es = EarlyStopping("val_loss", mode='min', verbose=1, patience=2)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(lstm_dataset.train_df['label']),
        y=lstm_dataset.train_df['label'],
    )

    batch_size = 32
    buffer_size = len(lstm_dataset.train_df)

    train_dataset = tf.data.Dataset.from_tensor_slices((lstm_dataset.train_padded, lstm_dataset.train_df['label'].values))
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((lstm_dataset.val_padded, lstm_dataset.val_df['label'].values))
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE

    history = model.fit(train_dataset,
                        batch_size=32,
                        validation_data=val_dataset,
                        callbacks=[es,tensorboard_callback],
                        epochs=10)


    test_acc, test_AUC, test_f1, transfer_acc, transfer_AUC, transfer_f1 = lstm_dataset.evaluate_model(
        model)

    false_samples = len(lstm_dataset.train_df[lstm_dataset.train_df['label'] == False])
    true_samples = len(lstm_dataset.train_df[lstm_dataset.train_df['label'] == True])
    data_meta_info = filename.split('_')
    wandb.log({
        'articles': data_meta_info[0],
        'pos': data_meta_info[1],
        'resolved': data_meta_info[2],
        'labeling': data_meta_info[3],
        'filter': data_meta_info[4],
        'test_acc': test_acc,
        'test_AUC': test_AUC,
        'test_f1': test_f1,
        'transfer_acc': transfer_acc,
        'transfer_AUC': transfer_AUC,
        'transfer_f1': transfer_f1,
        'class_false': false_samples,
        'class_true': true_samples
    })


main()
