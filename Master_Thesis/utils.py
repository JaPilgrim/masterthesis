import json
import math
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


def create_abbreviation_dict() -> dict:
    """Initialises a default abbreviation dict.

    Returns:
        dict: Keys: Abbreviation - Values: Full Text
    """
    abbr_to_text = {
        'a. D.': '',
        'a. d.': 'an der',
        'Abb.': 'Abbildung',
        'bzw.': 'beziehungsweise',
        'ca.': 'circa',
        'cf.': 'confer',
        'd. h.': 'das heisst',
        'Dr.': 'Doktor',
        'etc.': 'et cetera',
        'ggf.': 'gegebenenfalls',
        'i.d.R.': 'in der Regel',
        'i.A.': 'im Allgemeinen',
        'Ph.D.': 'Doctor of Philosophy',
        'Prof.': 'Professor',
        'prof.': 'Professor',
        'usw.': 'und so weiter',
        'v.a.': 'vor allem',
        'z. B.': 'zum Beispiel',
        'z. T.': 'zum Teil',
    }
    return abbr_to_text


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
    new_text = clean_special_characters(newtext)
    sen_text = new_text.split(". ")
    is_claim = label_wiki_sentences(sen_text)
    df = pd.DataFrame()
    df["text"] = sen_text
    df["target"] = is_claim
    return df


def split_val_train(df: pd.DataFrame, test_share=0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into train & test/val df

    Args:
        df (pd.DataFrame): DF
        test_share (float, optional): Share of val set. Defaults to 0.1.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: train set, test set
    """
    train_df, val_df = split_train_test(df, test_share)
    return train_df, val_df


def label_wiki_sentences(sen_text: list[str]) -> list[str]:
    """Labels preprocessed sentence list (wiki article) as claim or non-claim, based on 
    if they contain an underscore. Removes underscore afterwards

    Args:
        sen_text (list): list of preprocessed sentences of wiki articles 

    Returns:
        pd.DataFrame: Sentences ["text"] with label ["target"]
    """
    is_claim = []
    for i in range(len(sen_text)):
        if "_" in sen_text[i]:
            sen_text[i] = sen_text[i].replace('_', '')
            is_claim.append(True)
        else:
            is_claim.append(False)

    return is_claim


def clean_special_characters(text: str) -> str:
    """Returns cleaned text. Does so by:

    1. removing all points next to numbers,  
    2. Making ! to . full stop
    3. Making ? to ?.
    4. Strip()ing leading/trailing whitespaces
    5. Deleting \r
    6. Replacing \n with space

    Args:
        text (str): Raw text of any kind

    Returns:
        str: cleaned text.
    """
    new_text = re.sub("[0-9]\.", "", text)
    new_text = re.sub('\!', '\.', new_text)
    new_text = re.sub('\?', '\?.', new_text)
    new_text = new_text.strip()
    new_text = new_text.replace('\r', '')
    new_text = new_text.replace('\n', ' ')

    return new_text


def replace_abbreviations(text, replace_dict=create_abbreviation_dict()):
    """Replace abbreviations (dictionary keys) with full words (dictionary values). Default is a 
    short abbreviation list.

    Args:
        text (str): Text where Abbrv. should be removed
        replace_dict (_type_, optional): Dictionary to be used. Defaults create_abbreviation_dict()

    Returns:
        _type_: Text with replaced Abbreviations.
    """
    words = text.split()
    new_words = [replace_dict.get(word, word) for word in words]
    return " ".join(new_words)


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


def remove_stopwords(text: str, stopwords=stopwords.words("german")) -> str:
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


def draw_proportional_randomsample_from_FANG_df(df_full, total_number_sentences=100):
    """Draws ~ the input number of sentence from all "Verlage", equaling their share on total dataset.
    Each verlag is at least represented once.
    Args:
        df_full (_type_): whole fang dataset
        total_number_sentences (int, optional): The ~number of sentences to be drawn. Output will be 
        a bit higher due to up-rounding. Defaults to 100.

    Returns:
        list: List of sentences from all "Verlage"
    """
    grouped = df_full.groupby('source')
    verlage_aggr = grouped.aggregate('count')
    total_sentence_list = []
    sentence_fraction = total_number_sentences / len(df_full)
    for verlag in verlage_aggr.index:
        total_sentence_list = total_sentence_list + get_random_sentences_verlag(
            df_full, verlage_aggr, sentence_fraction, verlag)
    return total_sentence_list


def get_random_sentences_verlag(df_full, verlage_aggr, sentence_fraction, verlag):
    """Gets a fraction (specified in params) of random sentences from all articles of a verlag. Therefore iterates
    over all articles.


    Args:
        df_full (_type_): _description_
        verlage_aggr (_type_): _description_
        sentence_fraction (_type_): _description_
        verlag (_type_): _description_

    Returns:
        _type_: _description_
    """
    verlag_total_sentence_list = []
    print(verlag)
    verlag_number_sentences = math.ceil(verlage_aggr.loc[verlag, 'label'] * sentence_fraction)
    print(verlag_number_sentences)
    verlag_article_list = df_full[df_full.source == verlag]['article'].reset_index(drop=True)
    for article in verlag_article_list:
        verlag_total_sentence_list = verlag_total_sentence_list + get_sentences_article(article)
    verlag_random_selection = random.sample(verlag_total_sentence_list, verlag_number_sentences)
    return verlag_random_selection


def get_sentences_article(article: str) -> list:
    """Gets all cleaned sentences from article raw-text.

    Args:
        article (str): Raw-Text of an article

    Returns:
        list: List of all sentences in an article.
    """
    articles_sentence_list = []
    article = replace_abbreviations(article)
    cleaned_article = clean_special_characters(article)
    articles_sentence_list = cleaned_article.split('. ')
    return articles_sentence_list


def fetch_full_FANG_dataset() -> pd.DataFrame:
    """Fetches all FANG data into one DF. Depends on hardcoded local location!

    Returns:
        df: Dataframe inclduding the whole FANG dataset. 
    """
    df_list = []
    save = os.getcwd()
    os.chdir('/Users/jannis/Desktop/GitRepos/Master/masterthesis/Master_Thesis/data/fang-covid-main/articles')
    json_list = (range(1, 41241))
    try:
        for i in json_list:
            try:
                json_name = (str(i) + '.json')
                df = pd.read_json(json_name, typ='series')
            except FileExistsError:
                print(json_name)
                pass
            df_list.append(df)
    except:
        FileExistsError
    finally:
        os.chdir(save)
    df_full = pd.DataFrame(df_list)
    return df_full


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
        os.chdir('/Users/jannis/Desktop/fang-covid-main/articles/')
        json_list = sample(range(1, 40000), how_many_articles)
        for i in json_list:
            try:
                f = open(str(i) + ".json")
                single_fang = json.load(f)
                f.close()
            except FileExistsError:
                print("FileExistsError")
                print(str(i))
                pass

            concat_article_text = concat_article_text + single_fang["article"]
    finally:
        os.chdir(save)
    return concat_article_text
