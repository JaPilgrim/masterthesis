import pandas as pd

from utils import *

article_name_list = pd.read_csv('../data/all_protected_wiki_list.csv')

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

print(article_name_list)
fetched_articles_df = article_name_list[['title', 'bytes', 'sub_texts']]
fetched_articles_df.to_csv('../data/all_articles_list.csv')