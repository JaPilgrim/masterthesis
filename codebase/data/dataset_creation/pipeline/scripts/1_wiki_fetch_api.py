"""Fetches all articles from a list of articles names (Wiki). Uses HTML-parser to identify and 
    mark sentences:
    1. With @@ that have a citation or citation immediately following
    2. With == if they contain a (blue-font) link
    3. With ++ if they contain a word that has been used as link in that article before or is name 
        of a linked article

    Stores as csv, where each row represents an article.
"""

import concurrent.futures
import csv
from random import randint
from time import sleep

import pandas as pd

from utilities.utils import *


def fetch_wiki_text(name):
    successfull = False
    while not successfull:
        try:
            text = fetch_wiki_fulltext_linklabeled(name)
            successfull = True
            return text
        except Exception as e:
            print(e)
            sleep(randint(1, 10))


titles_path = ('../../../data_files/pipeline_steps/excellent_articles/0_article_namelist_excellent.csv')
output_path = (
    '../../../data_files/pipeline_steps/excellent_articles/1_all_articles_fetched.csv')
backup_path = ('../../../data_files/pipeline_steps/excellent_articles')
article_name_list = pd.read_csv(titles_path)
titles = article_name_list.title
fulltext_list = []

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    future_to_title = {executor.submit(fetch_wiki_text, title): title for title in titles}

    for i, future in enumerate(concurrent.futures.as_completed(future_to_title)):
        title = future_to_title[future]
        try:
            text = future.result()
            article_name_list.loc[article_name_list.title == title, 'sub_texts'] = text
        except Exception as e:
            print(f"Error fetching text for {title}: {e}")

        if i % 50 == 0:
            print(i)
        #     df_save = pd.DataFrame()
        #     df_save[''] pd.Series(fulltext_list)
        #     fetched_articles_df = article_name_list[['title', 'bytes', 'sub_texts']]
        #     df_save.to_csv(f"{backup_path}/backUp_{str(i)}.csv", index=False)



article_name_list.to_csv(output_path, index=False)
