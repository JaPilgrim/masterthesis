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

article_name_list = pd.read_csv('../../../data_files/pipeline_steps/all_protected_wiki_list.csv')
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
    fetched_articles_df.to_csv('../../../data_files/1_all_articles_fetched.csv',index=False)
except KeyboardInterrupt:
    store_df = pd.DataFrame()
    store_df['sub_texts'].to_csv(f'sub_texts_list_{len(store_df)}.csv')
