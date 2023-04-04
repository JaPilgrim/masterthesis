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


def get_pos_sequences(nlp, article_sentence_list):
    whole_pos_list = []
    for i, sentence_list in enumerate(article_sentence_list):
        words_removed = [remove_stopwords(sentence) for sentence in sentence_list]
        punct_removed = [
            sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in words_removed
        ]

        docs = list(nlp.pipe(punct_removed))

        article_pos_list = []
        for doc in docs:
            sentence_pos = [str(token.pos_) for token in doc if token.pos_ != "SPACE"]
            sentence_pos_str = " ".join(sentence_pos)
            article_pos_list.append(sentence_pos_str)

        whole_pos_list.append(article_pos_list)

        if (len(article_pos_list) != len(sentence_list)):
            print('<< Fault Here ! >>')
            print('index: ' + str(i))
            print('<< Fault Here ! >>')
        if len(whole_pos_list) % 100 == 0:
            print(len(whole_pos_list))
    return whole_pos_list


def main(folder: str, suffix=''):
    input_path = f"{folder}3_articles_resolved_labeled_cleaned{suffix}.csv"
    output_path = f"{folder}4_pos_resolved_truth_cleaned{suffix}.csv"
    nlp = spacy.load("de_core_news_sm", disable=['ner','parser'])

    # folder = '../../../data_files/pipeline_steps/excellent_articles/'
    # input_path = f"{folder}3_articles_resolved_labeled_cleaned.csv"
    # output_path = f"{folder}4_pos_resolved_truth_cleaned.csv"

    df = pd.read_csv(input_path)
    df['sentence_list'] = df['sentence_list'].apply(
        ast.literal_eval)
    df['resolved_sentence_list'] = df['resolved_sentence_list'].apply(
        ast.literal_eval)

    article_sentence_list = df.sentence_list
    pos_sentence_list = get_pos_sequences(nlp, article_sentence_list)
    df['pos_sentence_list'] = pos_sentence_list

    print('!!!!non-resolved-done!!!!!!')
    print('!!!!non-resolved-done!!!!!!')
    print('!!!!non-resolved-done!!!!!!')

    article_resolved_sentence_list = df.resolved_sentence_list
    pos_resolved_sentence_list = get_pos_sequences(nlp, article_resolved_sentence_list)
    df['pos_resolved_sentence_list'] = pos_resolved_sentence_list

    # article_resolved_sentence_list = df.resolved_sentence_list
    # pos_resolved_sentence_list = []
    # for i, sentence_list in enumerate(article_resolved_sentence_list):
    #     words_rem = [remove_stopwords(sentence) for sentence in sentence_list]
    #     punct_rem = [
    #         sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in words_rem
    #     ]
    #     pos_list = []
    #     for sentence in punct_rem:
    #         sentence_pos_list = get_sentence_pos_list(sentence, nlp)
    #         sentence_pos_list = [x for x in sentence_pos_list if x != "SPACE"]
    #         sentence_pos = " ".join(sentence_pos_list)
    #         pos_list.append(sentence_pos)
    #     pos_resolved_sentence_list.append(pos_list)
    #     if (len(pos_list) != len(sentence_list)):
    #         print('<< Fault Here ! >>')
    #         print('index: '+ str(i))
    #         print('<< Fault Here ! >>')
    #     if len(pos_resolved_sentence_list)%100==0:
    #         print(len(pos_resolved_sentence_list))


    # df['pos_resolved_sentence_list'] = pos_resolved_sentence_list

    df.to_csv(output_path, index=False)
