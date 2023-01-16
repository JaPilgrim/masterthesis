import json
import os
import random
import re
from random import sample

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from collections import Counter

from nltk.corpus import stopwords


def fetch_rawtext_from_wiki(subject='Maschinelles Lernen') -> str:
    """Fetches Raw Text from specified Wiki article

    Args:
        subject (str, optional): Name of Wiki-Article. Defaults to 'Maschinelles Lernen'.

    Returns:
        str: Raw text from wiki article (including citations)
    """
    url = 'https://de.wikipedia.org/w/api.php'
    params = {'action': 'parse', 'page': subject, 'format': 'json', 'prop': 'text', 'redirects': ''}

    response = requests.get(url, params=params)
    data = response.json()

    raw_html = data['parse']['text']['*']
    soup = BeautifulSoup(raw_html, 'html.parser')
    soup.find_all('p')
    text = ''

    for p in soup.find_all('p'):
        text += p.text
    return text


def preprocess_classify_wiki_text(wiki_raw_text: str) -> pd.DataFrame:
    """Transforms raw_text wiki article into df with sentences and 
    their is_claim label, based on citation

    Args:
        wiki_raw_text (str): Wiki article from API including citations "[1]"

    Returns:
        pd.DataFrame: df
    """
    newtext = re.sub("\[.*?\]", "_", wiki_raw_text)

    i = 0
    newtext = newtext.replace("._", "_.")
    while i < 1:
        newtext = newtext.replace("._", ".")
        i += 1
    sen_text = split_text(newtext)
    is_claim = []
    for i in sen_text:
        if "_" in i:
            is_claim.append(True)
        else:
            is_claim.append(False)
    df = pd.DataFrame()
    df["text"] = sen_text
    df["target"] = is_claim
    return df


def split_val_train(df: pd.DataFrame, test_share=0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = split_train_test(df, test_share)
    return train_df, val_df


def label_wiki_sentences(sen_text: list[str]) -> pd.DataFrame:
    """Labels preprocessed sentences (wiki article) as claim or non-claim, based on 
    if they contain an underscore

    Args:
        sen_text (list): list of preprocessed sentences of wiki articles 

    Returns:
        pd.DataFrame: Sentences ["text"] with label ["target"]
    """
    is_claim = []
    for i in sen_text:
        if "_" in i:
            is_claim.append(True)
        else:
            is_claim.append(False)
    df = pd.DataFrame()
    df["text"] = sen_text
    df["target"] = is_claim

    return df


def split_text(text: str) -> list:
    """Splits text into sentences. Does so by removing all points next to numbers, 
    than splitting on full-stops

    Args:
        text (str): Raw text of any kind

    Returns:
        list: list of sentences.
    """
    new_text = re.sub("[0-9]\.", "", text)
    sen_text = new_text.split(".")
    return sen_text


def split_train_test(df, test_size=.10) -> pd.DataFrame:
    """Splits df into train-test-dfs. Default 10% test.

    Args:
        df (_type_): dataframe of any kind
        test_size (float, optional): Fraction of Test Set. Defaults to .10.

    Returns:
        _type_: train & test df of incoming type
    """
    train_df, test_df = train_test_split(df, test_size=test_size)
    return train_df, test_df


def remove_stopwords(text, stopwords=stopwords.words("german")):
    stop = set(stopwords)
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


def fetch_from_fangcovid_local(how_many_articles: int, seed=5) -> str:
    """Returns a text, consisting of random FANG Covid nnews articles. 

    Args:
        how_many_articles (int): Number of articles to fetch
        seed (int, optional): Random Seed. Defaults to 5.

    Raises:
        FileExistsError: Fang Covid articles folder not found

    Returns:
        str: Raw text of concatennated FANG Covid articles
    """
    random.seed(seed)
    concat_article_text = ""
    save = os.getcwd()
    try:
        os.chdir("/Users/jannis/Desktop/fang-covid-main/articles/")
        json_list = sample(range(1, 40000), how_many_articles)
        for i in json_list:
            try:
                f = open(str(i) + ".json")
                df = json.load(f)
                f.close()
            except:
                raise FileExistsError
            concat_article_text = concat_article_text + df["article"]
    finally:
        os.chdir(save)
    return concat_article_text


# def fetch_from_fangcovid_remote(number: int):
#     json_list = sample(range(1, 40000), number)
#     df = pd.read_csv(
#         "https://github.com/justusmattern/fang-covid/blob/main/articles/0.json?raw=true")
#     # for i in json_list:
#     #     url="https://github.com/justusmattern/fang-covid/tree/main/articles"
#     return df
# vectorize a text corpus by turning each text into a sequence of integers
# fit only to training