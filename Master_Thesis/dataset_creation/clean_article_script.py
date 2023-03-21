"""Reads a csv that contains a list of all articles with their fulltext.
1. Cleans the texts.
2. Splits into sentences.
3. Uses the uses marks to determine truth status by dataset type. 
    - Quotation labeled
    - linklabeled
    - linknamelabeled
4. Joins split sentences for clean text version per article.
5. Merges everything into df and stores as .csv
"""
import pandas as pd

from utils import *

all_article_list = pd.read_csv('../data/all_articles_list.csv')

text_list = all_article_list.sub_texts
cleaned_text_list = []

articles_sen_list = []

articles_quot_truth_list = []
articles_link_truth_list = []
articles_linkname_truth_list = []

for text in text_list:
    curr_sen_list = []

    curr_quot_truth = []
    curr_link_truth = []
    curr_linkname_truth = []
    str_cast_text = str(text)
    no_abbr_text = replace_abbreviations(str_cast_text)
    no_spec_text = clean_special_characters(no_abbr_text)

    curr_sen_list = no_spec_text.split('. ')

    for i, text in enumerate(curr_sen_list):
        if (' @@ ' in curr_sen_list[i]):
            curr_sen_list[i] = curr_sen_list[i].replace(' @@ ', '')
            curr_sen_list[i] = curr_sen_list[i].replace(' == ', '')
            curr_sen_list[i] = curr_sen_list[i].replace('==', '')
            curr_sen_list[i] = curr_sen_list[i].replace(' ++ ', '')

            curr_quot_truth.append(True)
            curr_link_truth.append(True)
            curr_linkname_truth.append(True)
        elif ('==' in curr_sen_list[i]):
            curr_sen_list[i] = curr_sen_list[i].replace(' == ', '')
            curr_sen_list[i] = curr_sen_list[i].replace('==', '')
            curr_sen_list[i] = curr_sen_list[i].replace(' ++ ', '')

            curr_quot_truth.append(False)
            curr_link_truth.append(True)
            curr_linkname_truth.append(True)
        elif (' ++ ' in curr_sen_list[i]):
            curr_sen_list[i] = curr_sen_list[i].replace(' ++ ', '')

            curr_quot_truth.append(False)
            curr_link_truth.append(False)
            curr_linkname_truth.append(True)
        else:
            curr_quot_truth.append(False)
            curr_link_truth.append(False)
            curr_linkname_truth.append(False)
        curr_sen_list[i] = curr_sen_list[i].replace(' @@ ', '')
        curr_sen_list[i] = curr_sen_list[i].replace(' == ', '')
        curr_sen_list[i] = curr_sen_list[i].replace('==', '')
        curr_sen_list[i] = curr_sen_list[i].replace(' ++ ', '')
    curr_clean_joined_text = ". ".join(curr_sen_list)
    cleaned_text_list.append(curr_clean_joined_text)

    articles_sen_list.append(curr_sen_list)

    articles_quot_truth_list.append(curr_quot_truth)
    articles_link_truth_list.append(curr_link_truth)
    articles_linkname_truth_list.append(curr_linkname_truth)
    print(len(articles_quot_truth_list))

all_article_list['cleaned_article_text'] = cleaned_text_list

all_article_list['sentence_list'] = articles_sen_list

all_article_list['quot_truth_list'] = articles_quot_truth_list
all_article_list['quot_truth_list'] = articles_quot_truth_list
all_article_list['quot_truth_list'] = articles_quot_truth_list

all_article_list.to_csv('../data/all_articles_truth_cleaned.csv')
