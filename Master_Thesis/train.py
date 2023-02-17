import keras
import pandas as pd
import wandb
import yaml
from keras.callbacks import EarlyStopping
from sentence_classifier import SentenceClassifier
from sklearn import metrics
from tokenizer_class import TokenizerClass
from utils import *

from tensorflow import keras
from tensorflow.keras import callbacks, layers

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


def fetch_prep_wiki_dataset(path='fulltext_wiki_protected.csv'):
    all_articles = pd.read_csv(path, sep=',')[['title', 'bytes', 'full_text']]
    i = 1
    df = preprocess_classify_wiki_text(all_articles['full_text'].iloc[0])
    while i < len(all_articles):
        working_df = preprocess_classify_wiki_text(all_articles['full_text'].iloc[i])
        df = pd.concat([df, working_df])
        i += 1
    return df


def evaluate_model(model, claim_extract):
    annotated = pd.read_csv('FullAnnotated1.csv', sep=';', dtype={'sentence': 'str'})
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
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    df = fetch_prep_wiki_dataset()
    df_train = df.sample(50000, random_state=2)
    claim_extract = init_classes(df_train, 0.25)
    model = keras.models.Sequential()
    model.add(layers.Embedding(claim_extract.tokenizer_class.num_unique_words, 32, input_length=32))
    model.add(layers.LSTM(wandb.config.hidden_layer_size, dropout=wandb.config.LSTM_dropout))
    model.add(layers.Dense(1, activation=wandb.config.dense_activation))
    model.summary()
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    optim = keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
    metrics = ['accuracy']
    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    es = EarlyStopping("val_accuracy", mode='min', verbose=1, patience=2)
    history = model.fit(claim_extract.train_padded,
                        claim_extract.train_df['target'],
                        validation_data=(claim_extract.val_padded, claim_extract.val_df['target']),
                        callbacks=[es],
                        epochs=10)
    plot, auc = evaluate_model(model, claim_extract)
    wandb.log({
        'AUC': auc,
        'whole_size': len(claim_extract.whole_df),
        'train_size': len(claim_extract.train_df)
    })


main()
