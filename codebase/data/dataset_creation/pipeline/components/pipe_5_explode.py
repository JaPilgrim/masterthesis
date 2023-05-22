"""Explodes dataframe to a sentence=row basis
    1. Remove full articles that:
        - Have less than 400 characters
        - Have 'Liste' in title
    2. Separate article-info from sentence-based
    3. Explode only sentence-based (+article ID)
        - rename columns
    4. Store 2 DFs, one article & one sentence rows
"""
import ast

import pandas as pd

from utilities.utils import *

folder = '../../../data_files/pipeline_steps/excellent_articles/'
input_path = f"{folder}4_pos_resolved_truth_cleaned.csv"
output_path_sentences = f"{folder}5.1_sentences_exploded.csv"
output_path_articles = f"{folder}5.2_article_info.csv"


def main(folder: str, suffix=''):
    input_path = f"{folder}4_pos_resolved_truth_cleaned{suffix}.csv"
    output_path_sentences = f"{folder}5.1_sentences_exploded{suffix}.csv"
    output_path_articles = f"{folder}5.2_article_info{suffix}.csv"
    df = pd.read_csv(input_path)

    rename = {
        'resolved_sentence_list': 'nopos_resolved_text',
        'sentence_list': 'nopos_nonresolved_text',
        'quot_truth_list': 'quot_label',
        'link_truth_list': 'link_label',
        'linkname_truth_list': 'namelink_label',
        'pos_sentence_list': 'pos_nonresolved_text',
        'pos_resolved_sentence_list': 'pos_resolved_text',
    }

    list_cols = [
        'resolved_sentence_list', 'sentence_list', 'quot_truth_list', 'link_truth_list',
        'linkname_truth_list', 'pos_resolved_sentence_list', 'pos_sentence_list'
    ]

    for col in list_cols:
        df[col] = df[col].apply(ast.literal_eval)

    drop_indices = []
    text_list = df['cleaned_article_text']
    for i, text in enumerate(text_list):
        if len(str(text)) < 400 or 'Liste' in df['title'].iloc[i]:
            drop_indices.append(i)
    df = df.drop(drop_indices)
    df = df.reset_index(drop=True)

    print('Dropping ' + str(len(drop_indices)) + " articles:")
    print(drop_indices)

    df['article_index'] = df.index
    df_articles = df.copy()
    df_articles = df_articles[[
        'article_index',
        'title',
        'bytes',
        'resolved_text_list',
        'cleaned_article_text',
        'sub_texts',
    ]]

    df_sentences = df.copy()
    df_sentences = df_sentences[[
        'article_index',
        'resolved_sentence_list',
        'sentence_list',
        'pos_sentence_list',
        'pos_resolved_sentence_list',
        'quot_truth_list',
        'link_truth_list',
        'linkname_truth_list',
    ]]
    df_sentences = df_sentences.explode(list_cols)
    df_sentences = df_sentences.rename(columns=rename)
    df_sentences = df_sentences.reset_index(drop=True)

    pos_sentences = df_sentences['pos_nonresolved_text']
    sentence_drop_indices = []
    for i, pos in enumerate(pos_sentences):
        if 'VERB' not in str(pos) or 'nan' in str(pos) or len(str(pos).split()) <4:
            sentence_drop_indices.append(i)
    
    print('Dropping ' + str(len(sentence_drop_indices)) + " sentences:")
    # print(sentence_drop_indices)

    print(len(df_sentences))
    df_sentences = df_sentences.drop(sentence_drop_indices)
    df_sentences = df_sentences.reset_index(drop=True)
    print(len(df_sentences))
    print(df_sentences.columns)


    df_sentences.to_csv(output_path_sentences)
    df_articles.to_csv(output_path_articles)
