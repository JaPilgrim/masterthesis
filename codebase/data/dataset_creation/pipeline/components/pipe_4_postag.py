"""Gets the POS-sequence for every sentence.
    1. For resolved- & non-resolved sentences seperately
    2. Loop through all articles>sentences
    3. Get POS-sequence (as string per sentence)
    4. Check if sentence count still matches, otherwise drop (didnt happen)
    5. Store with 2 separate text columns still row=article
        -pos_resolved_sentence_list
        -pos_sentence_list
    """
import ast
import os
import sys

sys.path.append("..")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import importlib.metadata as importlib_metadata
import string

import pandas as pd
import spacy

from utilities.utils import *

importlib_metadata.version('tqdm')
import spacy

nlp = spacy.load("de_core_news_sm")

def main(file_path='../../../data_files/',file_name='intermediate_df'):
    df = pd.read_csv(f'{file_path}3_{file_name}.csv')

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

    df.to_csv(f'{file_path}4_{file_name}.csv', index=False)
