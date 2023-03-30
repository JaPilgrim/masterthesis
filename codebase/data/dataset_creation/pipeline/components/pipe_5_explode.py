
import ast

import pandas as pd

from utilities.utils import *


def main(file_path='../../../data_files/test_pipeline/', file_name='intermediate_df'):
    """Explodes dataframe to a sentence=row basis

        1. Separate article-info from sentence-based
        2. Explode only sentence-based (+article ID)
            - rename columns
        3. Store 2 DFs, one article & one sentence rows

    Args:
        file_path (str, optional): Path to store (intermediate) results. 
                                    Defaults to '../../../data_files/test_pipeline/'.
        file_name (str, optional): Name for intermediate csv. Defaults to 'intermediate_df'.
    """
    df = pd.read_csv(f'{file_path}4_{file_name}.csv')

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

    df['article_index'] = df.index
    df_articles = df.copy()
    df_articles = df_articles[[
        'article_index',
        'title',
        'bytes',
        'no_sen',
        'res_no_sen',
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

    # df_sentences['pos_sentence_strings'] = df_sentences['pos_sentence_list']
    # df_sentences['pos_resolved_sentence_strings'] = df_sentences['pos_resolved_sentence_list']

    # pos_sentences_list = df_sentences['pos_sentence_list'].str.split().tolist()
    # pos_resolved_sentences_list = df_sentences['pos_resolved_sentence_list'].str.split().tolist()

    # df_sentences['pos_sentence_list'] = pos_sentences_list
    # df_sentences['pos_resolved_sentence_list'] = pos_resolved_sentences_list
    print(df_sentences.columns)
    df_sentences.to_csv(f'{file_path}5.1_{file_name}.csv',index=False)
    df_articles.to_csv(f'{file_path}5.2_{file_name}.csv',index=False)
