import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from evaluate import load
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
from transformers import AutoTokenizer, DataCollatorWithPadding

from utilities.lstm_data_handler import LSTMDataHandler
from utilities.tokenizer_class import TokenizerClass
from utilities.utils import *

trans_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-german-cased')
import os

os.environ['TOKENIZERS_PARALLELISM'] = '1'
import tensorflow as tf
import wandb
from transformers import TFAutoModelForSequenceClassification, create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras import  mixed_precision

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# wandb.init(
#     project='pretrain-transform-wiki',
#     config={
#     'class_size':2000,
#     'random_state':2,
#     'eval_metric':'accuracy',
#     'eval_share':0.25,
#     'batch_size':256, #256
#     'epochs':4,
#     'huggingface_model':'distilbert-base-german-cased',
#     'loss_function':None,
#     'learning_rate':0.00032,
#     }
# )
# config=wandb.config
def evaluate_transformer(df:pd.DataFrame(), model) -> tuple[float,float,float,float]:
    sentence_list = list(df['text'])
    ground_truth = list(df['label'])

    tokenized = trans_tokenizer(sentence_list, return_tensors='np', padding='longest')
    outputs = model(tokenized).logits

    predictions = np.argmax(outputs, axis=1)
    predictions_list = list(predictions)

    accuracy, precision, recall, f1_value = f1_score(ground_truth, predictions_list)

    return accuracy, precision, recall, f1_value



def preprocess_transformer_token(example):
    return trans_tokenizer(example['text'], truncation=True)


data_collator = DataCollatorWithPadding(tokenizer=trans_tokenizer)

accuracy = load('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

filename = 'excellent_nopos_resolved_namelink_nofilter.csv'
df = pd.read_csv(
    f'../data/data_files/test_samples/6th_smalltransformer/{filename}',
    nrows=250)
test_share = 0.2
eff_val_share = 0.2 / (1 - test_share)
wiki_dataset = Dataset.from_pandas(df, preserve_index=False)
tokenized_wiki = wiki_dataset.map(preprocess_transformer_token, batched=True)
test_split = tokenized_wiki.train_test_split(test_share)
test_set = test_split['test']
val_split = test_split['train'].train_test_split(eff_val_share)
print(val_split['test']['text'])
print(val_split['test']['label'])
batch_size = 32
num_epochs = 2
batches_per_epoch = len(val_split['train']) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=0.00032,
                                       num_warmup_steps=0,
                                       num_train_steps=total_train_steps)

model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-german-cased",
                                                             num_labels=2,
                                                             from_pt=True)

transfer_set = pd.read_csv('../data/data_files/annotated_pos_test.csv')

if filename[:3] + 'pos':
        transfer_set['text'] = transfer_set['pos_string']

tf_train_set = model.prepare_tf_dataset(
    val_split['train'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    val_split["test"],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)

model.compile(optimizer=optimizer, )

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
# push_to_hub_callback = PushToHubCallback(
#     output_dir="new_version123",
#     tokenizer=trans_tokenizer,
# )

# callbacks = [metric_callback, push_to_hub_callback,WandbMetricsLogger(log_freq=5),WandbModelCheckpoint('models')]
callbacks = [
    metric_callback,
]


model.summary()
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs)


test_accuracy, test_precision, test_recall, test_f1_value = evaluate_transformer(test_set,model,)
transfer_accuracy, transfer_precision, transfer_recall, transfer_f1_value = evaluate_transformer(transfer_set,model,)

log = {

        'test_acc': test_accuracy,
        'test_prec': test_precision,
        'test_rec' : test_recall,
        'test_f1': test_f1_value,
        'transfer_acc': transfer_accuracy,
        'transfer_prec': transfer_precision,
        'transfer_rec' : transfer_recall,
        'transfer_f1': transfer_f1_value,
}

print(log)


# softmaxed = tf.nn.softmax(outputs)
# softmaxed = softmaxed.numpy()
