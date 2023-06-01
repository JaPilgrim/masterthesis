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
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


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


def preprocess_transformer_token(example):
    return trans_tokenizer(example['text'], truncation=True)


data_collator = DataCollatorWithPadding(tokenizer=trans_tokenizer)

accuracy = load('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


df = pd.read_csv(
    '../data/data_files/test_samples/5th_try/excellent_pos_resolved_namelink_nofilter.csv',
    nrows=2500)
test_share = 0.2
eff_val_share = 0.2 / (1 - test_share)

wiki_dataset = Dataset.from_pandas(df, preserve_index=False)
tokenized_wiki = wiki_dataset.map(preprocess_transformer_token, batched=True)
test_split = tokenized_wiki.train_test_split(test_share)
test_set = test_split['test']
val_split = test_split['train'].train_test_split(eff_val_share)

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
push_to_hub_callback = PushToHubCallback(
    output_dir="new_version123",
    tokenizer=trans_tokenizer,
)

# callbacks = [metric_callback, push_to_hub_callback,WandbMetricsLogger(log_freq=5),WandbModelCheckpoint('models')]
callbacks = [
    metric_callback,
]

# element_train = next(iter(tf_train_set)
#                      )
# feature_train, label_train = element_train

# element_val = next(iter(tf_validation_set))
# feature_val, label_val = element_val

# feature_shape = feature_train.shape
# label_shape = label_train.shape

print(tf_train_set)
print(tf_validation_set)

# print(label_shape)
# # Check shapes of input tensors
# print("Input tensor shapes:")
# for idx in range(2):
#     print(f"Input {idx+1}: {tf_train_set.element_spec[idx]}")

# # Check shapes of output tensors
# print("Output tensor shapes:")
# print(tf_train_set.element_spec[2].shape)

# # Check shapes of input tensors for the validation set
# print("Validation input tensor shapes:")
# for idx in range(2):
#     print(f"Input {idx+1}: {tf_validation_set.element_spec[idx]}")

# Check shapes of output tensors for the validation set
# print("Validation output tensor shapes:")
# print(tf_validation_set.element_spec[2].shape)

model.summary()
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs)

sentence_list = list(test_set['text'])
tokenized = trans_tokenizer(sentence_list, return_tensors='np', padding='longest')

outputs = model(tokenized).logits
predictions = np.argmax(outputs, axis=1)
predictions_list = list(predictions)

ground_truth = list(test_set['label'])

accuracy = accuracy_score(ground_truth, predictions_list)
precision = precision_score(ground_truth, predictions_list)
recall = recall_score(ground_truth, predictions_list)
f1_value = f1_score(ground_truth, predictions_list)

print(accuracy)
print(precision)
print(recall)
print(f1_value)

# softmaxed = tf.nn.softmax(outputs)
# softmaxed = softmaxed.numpy()
