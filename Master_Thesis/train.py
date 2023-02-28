import hashlib
import json

import keras
import pandas as pd
import yaml
from keras.callbacks import EarlyStopping
from numpy.random import seed
from tensorflow import keras
from tensorflow.keras import callbacks, layers
import tensorflow as tf
import wandb
from back_classes.sentence_classifier import SentenceClassifier
from back_classes.tokenizer_class import TokenizerClass
from utils import *
import random as python_random
# sweep_configuration = {
#     'method': 'bayes',
#     'name': 'sweep-first',
#     'metric': {
#         'goal': 'maximize',
#         'name': 'AUC'
# 		},
#     'parameters': {
#     'compile_metric':['accuracy','binary_accuracy','f1-score','AUC'],
#     'dense_activation':['relu','tanh','relu','softmax','elu','selu','softplus','softsign','swish'],
#     'ES_metric':['loss','val_loss','accuracy','val_accuracy'],
#     'eval_share':{max:0.5,min:0.1},
#     'LSTM_dropout':{max:0.5,min:0.001},
#     'hidden_layer_size':{max:512,min:16},
#     'epochs':{max:20,min:2},
#     'learning_rate':{max:0.1,min:0.00001},
#     }
#      }


def fetch_sampled_wiki_sentences(sample_size=221352,
                                 random_state=2,
                                 path='/data/fulltext_wiki_protected.csv'):
    """From folder location, fetches sampled number of wiki-sentences.

    Args:
        sample_size (int, optional): Number of desired sentences. Defaults to 50000.
        random_state (int, optional): Random seed. Defaults to 2.
        path (str, optional): Path to csv-file. Defaults to '/data/fulltext_wiki_protected.csv'.

    Returns:
        pd.DataFrame: 
    """
    class_size = int(sample_size / 2)
    all_articles = pd.read_csv(path, sep=',')[['title', 'bytes', 'full_text']]
    i = 1
    df = preprocess_classify_wiki_text(all_articles['full_text'].iloc[0])
    while i < len(all_articles):
        working_df = preprocess_classify_wiki_text(all_articles['full_text'].iloc[i])
        df = pd.concat([df, working_df])
        i += 1
    df_true = df[df['label'] == True].sample(n=class_size, random_state=random_state)
    df_false = df[df['label'] == False].sample(n=class_size, random_state=random_state)
    df_concated = pd.concat([df_true, df_false])
    return df_concated
    
def depr_fetch_sampled_wiki_sentences(class_size=110000,
                                 random_state=2,
                                 path='/data/wiki_all_sentence.csv'):
    df=pd.read_csv(path, sep=',')[['text','label']]
    df_true = df[df['label'] == True].sample(n=class_size, random_state=random_state)
    df_false = df[df['label'] == False].sample(n=class_size, random_state=random_state)
    df_concated = pd.concat([df_true, df_false])
    return df_concated

def hash_fn_wrapper(loss_fn):
    hash_value = (str(loss_fn.__class__.__name__) + str(loss_fn.__dict__))
    return hash_value


def evaluate_model(model, claim_extract):
    annotated = pd.read_csv('/data/FullAnnotated1.csv', sep=';', dtype={'sentence': 'str'})
    annotated = (annotated[annotated['to_exclude'] == 0])
    annotated_text = list(annotated['sentence'])
    annotated_padded = claim_extract.raw_text_to_padded_sequences(list(annotated_text))
    annotated['predictions'] = list(model.predict(annotated_padded))
    plot, auc = plot_compute_AUC(annotated['is_claim'], annotated['predictions'])
    return plot, auc


def init_classes(df_train, eval_share):
    tokenizer = TokenizerClass()
    claim_extract = SentenceClassifier(tokenizer_class=tokenizer)
    claim_extract.preprocess_train_val(df_train, eval_share)
    return claim_extract


def main():
    random_state=2
    seed(random_state)
    python_random.seed(random_state)
    tf.random.set_seed(random_state)
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    df_train = fetch_sampled_wiki_sentences(221352, random_state=2)
    claim_extract = init_classes(df_train, 0.25)

    model = keras.models.Sequential()
    model.add(layers.Embedding(claim_extract.tokenizer_class.num_unique_words, 32, input_length=32,embeddings_initializer=tf.keras.initializers.RandomUniform(seed=random_state)),)
    model.add(layers.LSTM(wandb.config.hidden_layer_size, dropout=wandb.config.LSTM_dropout, recurrent_initializer=tf.keras.initializers.Orthogonal(seed=42)))
    model.add(layers.Dense(1, activation=wandb.config.dense_activation,kernel_initializer=tf.keras.initializers.RandomUniform(seed=42)))


    model.summary()
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)


    metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    es = EarlyStopping("val_accuracy", mode='min', verbose=1, patience=2)


    history = model.fit(claim_extract.train_padded,
                        claim_extract.train_df['label'],batch_size=wandb.config.batch_size,
                        validation_data=(claim_extract.val_padded, claim_extract.val_df['label']),
                        callbacks=[es],
                        epochs=10)

    history_hash = hashlib.sha256(str(history.history).encode('utf-8')).hexdigest()
    plot, auc = evaluate_model(model, claim_extract)
    wandb.log({
        'AUC': auc,
        'whole_size': len(claim_extract.whole_df),
        'train_size': len(claim_extract.train_df)
    })


main()
