"""Fetches all articles from a list of articles names (Wiki). Uses HTML-parser to identify and 
    mark sentences:
    1. With @@ that have a citation or citation immediately following
    2. With == if they contain a (blue-font) link
    3. With ++ if they contain a word that has been used as link in that article before or is name 
        of a linked article

    Stores as csv, where each row represents an article.
"""

import pandas as pd
import csv
from utilities.utils import *
import sys
sys.path.append('....')

def main(article_name_path:str,file_path='../../../data_files/test_pipeline/',file_name='intermediate_df',nrows=None):
    """Fetches all articles from a list of articles names (Wiki). Uses HTML-parser to identify and 
    mark sentences:

            1. With @@ that have a citation or citation immediately following
            2. With == if they contain a (blue-font) link
            3. With ++ if they contain a word that has been used as link in that article before or is name 
                of a linked article

    Stores as csv, where each row represents an article.

    Args:
        article_name_path (str): Path to the list of article names.
        file_path (str, optional): Path to store (intermediate) results. 
                                    Defaults to '../../../data_files/test_pipeline/'.
        file_name (str, optional): Name for intermediate csv. Defaults to 'intermediate_df'.
    """
    article_name_list = pd.read_csv(article_name_path,nrows=nrows)
    try:
        fulltext_list = []
        titles = article_name_list.title
        for name in titles:
            successfull = False
            while not successfull:
                try:
                    text = fetch_wiki_fulltext_linklabeled(name)
                    fulltext_list.append(text)
                    successfull = True
                except Exception as e:
                    print(e)
            print(len(fulltext_list))
        article_name_list['sub_texts'] = fulltext_list
        fetched_articles_df = article_name_list[['title', 'bytes', 'sub_texts']]
        fetched_articles_df.to_csv(f'{file_path}1_{file_name}.csv', index=False)
    except KeyboardInterrupt:
        store_df = article_name_list.copy()
        store_df['sub_texts'].to_csv(f'sub_texts_list_{len(store_df)}.csv')
