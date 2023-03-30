"""To test & compare all 48 datasets in small samples. To be run with wandb agent.
    """

import numpy as np
import pandas as pd
import wandb
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import callbacks, layers

from utilities.lstm_data_handler import LSTMDataHandler
from utilities.utils import *

sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_same_rows',
    'metric': {
        'goal': 'maximize',
        'name': 'test_acc'
    },
    'parameters': {
        'filename': {
            'values': [
                'nopos_resolved_quot_filter_equal.csv',
                'nopos_resolved_quot_nofilter_equal.csv',
                'nopos_nonresolved_quot_filter_equal.csv',
                'nopos_nonresolved_quot_nofilter_equal.csv',
                'pos_nonresolved_quot_filter_equal.csv',
                'pos_nonresolved_quot_nofilter_equal.csv',
                'pos_resolved_quot_filter_equal.csv',
                'pos_resolved_quot_nofilter_equal.csv',
                'nopos_resolved_link_filter_equal.csv',
                'nopos_resolved_link_nofilter_equal.csv',
                'nopos_nonresolved_link_filter_equal.csv',
                'nopos_nonresolved_link_nofilter_equal.csv',
                'pos_nonresolved_link_filter_equal.csv',
                'pos_nonresolved_link_nofilter_equal.csv',
                'pos_resolved_link_filter_equal.csv',
                'pos_resolved_link_nofilter_equal.csv',
                'nopos_resolved_namelink_filter_equal.csv',
                'nopos_resolved_namelink_nofilter_equal.csv',
                'nopos_nonresolved_namelink_filter_equal.csv',
                'nopos_nonresolved_namelink_nofilter_equal.csv',
                'pos_nonresolved_namelink_filter_equal.csv',
                'pos_nonresolved_namelink_nofilter_equal.csv',
                'pos_resolved_namelink_filter_equal.csv',
                'pos_resolved_namelink_nofilter_equal.csv',
                'nopos_resolved_quot_filter_random.csv',
                'nopos_resolved_quot_nofilter_random.csv',
                'nopos_nonresolved_quot_filter_random.csv',
                'nopos_nonresolved_quot_nofilter_random.csv',
                'pos_nonresolved_quot_filter_random.csv',
                'pos_nonresolved_quot_nofilter_random.csv',
                'pos_resolved_quot_filter_random.csv',
                'pos_resolved_quot_nofilter_random.csv',
                'nopos_resolved_link_filter_random.csv',
                'nopos_resolved_link_nofilter_random.csv',
                'nopos_nonresolved_link_filter_random.csv',
                'nopos_nonresolved_link_nofilter_random.csv',
                'pos_nonresolved_link_filter_random.csv',
                'pos_nonresolved_link_nofilter_random.csv',
                'pos_resolved_link_filter_random.csv',
                'pos_resolved_link_nofilter_random.csv',
                'nopos_resolved_namelink_filter_random.csv',
                'nopos_resolved_namelink_nofilter_random.csv',
                'nopos_nonresolved_namelink_filter_random.csv',
                'nopos_nonresolved_namelink_nofilter_random.csv',
                'pos_nonresolved_namelink_filter_random.csv',
                'pos_nonresolved_namelink_nofilter_random.csv',
                'pos_resolved_namelink_filter_random.csv',
                'pos_resolved_namelink_nofilter_random.csv',
            ]
        }
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project='equal_rows_swipe_testsample')


def main():
    run = wandb.init()
    filename = wandb.config.filename
    lstm_dataset = LSTMDataHandler()
    lstm_dataset.load_datasets(
        f"../data/data_files/test_samples/3rd_test_dataset_samples/{filename}",
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
    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit(lstm_dataset.train_padded,
                        lstm_dataset.train_df['label'],
                        batch_size=32,
                        class_weight=class_weights_dict,
                        validation_data=(lstm_dataset.val_padded, lstm_dataset.val_df['label']),
                        callbacks=[es],
                        epochs=10)

    test_acc, test_AUC, test_f1, transfer_acc, transfer_AUC, transfer_f1 = lstm_dataset.evaluate_model(
        model)

    false_samples = len(lstm_dataset.train_df[lstm_dataset.train_df['label'] == False])
    true_samples = len(lstm_dataset.train_df[lstm_dataset.train_df['label'] == True])
    data_meta_info = filename.split('_')
    wandb.log({
        'pos': data_meta_info[0],
        'resolved': data_meta_info[1],
        'labeling': data_meta_info[2],
        'filter': data_meta_info[3],
        'sampling_method': data_meta_info[4],
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
