import os
import sys
import ast
sys.path.append("..")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
from utils import *
import string

import spacy


import importlib.metadata as importlib_metadata

importlib_metadata.version('tqdm')
import spacy

nlp = spacy.load("de_core_news_sm")

df = pd.read_csv('../../data/all_articles_resolved_truth_cleaned_new.csv')

df['sentence_list'] = df['sentence_list'].apply(
    ast.literal_eval)
df['resolved_sentence_list'] = df['resolved_sentence_list'].apply(
    ast.literal_eval)

article_sentence_list = df.sentence_list
pos_sentence_list = []
for i, sentence_list in enumerate(article_sentence_list):
    words_rem = [remove_stopwords(sentence) for sentence in sentence_list]
    punct_rem = [
        sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in words_rem
    ]
    pos_list = []
    for sentence in punct_rem:
        sentence_pos = get_sentence_pos_str(sentence, nlp)
        pos_list.append(sentence_pos)
    pos_sentence_list.append(pos_list)
    if (len(pos_list) != len(sentence_list)):
        print('<< Fault Here ! >>')
        print('index: '+ str(i))
        print('<< Fault Here ! >>')
    if len(pos_sentence_list)%100==0:
        print(len(pos_sentence_list))

df['pos_sentence_list'] = pos_sentence_list
print('!!!!non-resolved-done!!!!!!')
print('!!!!non-resolved-done!!!!!!')
print('!!!!non-resolved-done!!!!!!')

article_resolved_sentence_list = df.resolved_sentence_list
pos_resolved_sentence_list = []
for i, sentence_list in enumerate(article_resolved_sentence_list):
    words_rem = [remove_stopwords(sentence) for sentence in sentence_list]
    punct_rem = [
        sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in words_rem
    ]
    pos_list = []
    for sentence in punct_rem:
        sentence_pos = get_sentence_pos_str(sentence, nlp)
        pos_list.append(sentence_pos)
    pos_resolved_sentence_list.append(pos_list)
    if (len(pos_list) != len(sentence_list)):
        print('<< Fault Here ! >>')
        print('index: '+ str(i))
        print('<< Fault Here ! >>')
    if len(pos_resolved_sentence_list)%100==0:
        print(len(pos_resolved_sentence_list))


df['pos_resolved_sentence_list'] = pos_resolved_sentence_list

df.to_csv('../../data/4_pos_resolved_truth_cleaned.csv',index=False)