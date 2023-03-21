import json
import math
import os
import random
import re
from random import sample
from typing import List

import nltk
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from collections import Counter

import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from spacy import Language


def get_sentence_pos_str(sentence: str, nlp: Language) -> str:
    """Returns the part-of-speech tags of a given sentence as concatenated strings.

    Args:
        sentence (str): A sentence, preferably cleaned of stopwords & punctuation.
        nlp (Language): NLTK language object.

    Returns:
        str: String of POS tag sequence.
    """
    doc = nlp(sentence)
    sentence_pos = ""
    for token in doc:
        sentence_pos = sentence_pos + " " + token.pos_
    return sentence_pos


def prefix_on_substring(word_list: List, free_text: str,prefix=" ++ ") -> str:
    """Adds a prefix in front o every occurence of an element of word_list in free_text.

    Args:
        word_list (list[str]): List of words to be searched for.
        free_text (str): Free text to be looked in.
        prefix (str): Desired prefix. Defaults to " ?? "

    Returns:
        linked_text: Free text but with prefix added.
    """
    for word in word_list:
        if word in free_text:
            free_text = free_text.replace(word, prefix + word)
    return free_text


def get_sentence_pos_list(sentence, nlp: Language) -> list[str]:
    """Returns the part-of-speech tags of a given sentence as list.

    Args:
        sentence (str): A sentence, preferably cleaned of stopwords & punctuation.
        nlp (Language): NLTK language object.

    Returns:
        list[str]: List of String POS tags.
    """

    doc = nlp(sentence)
    sentence_pos = []
    for token in doc:
        sentence_pos.append(str(token.pos_))
    return sentence_pos


def compute_accuracy_AUC(probabilities: pd.Series, prediction: pd.Series) -> tuple:
    """Computes and returns AUC & accuracy. 
    Args:
        ground_truth (list[bool]): Ground truth bool
        predicition (list[bool]): Prediction bool

    Returns:
        float: accuracy
        float: AUC
    """
    ground_truth = np.where(probabilities >= 0.5, 1, 0)

    accuracy = accuracy_score(ground_truth, prediction)
    auc = metrics.roc_auc_score(ground_truth, prediction)
    return accuracy, auc


def plot_compute_AUC(ground_truth: pd.Series, prediction: pd.Series) -> tuple:
    """Gets ground truth & prediction. Computes AUC & plots. Returns plot & figure.

    Args:
        ground_truth (list[bool]): Ground truth bool
        predicition (list[bool]): Prediction bool

    Returns:
        plot: plot file
        float: area under the curve
    """
    fpr, tpr, _ = metrics.roc_curve(ground_truth, prediction)
    auc = metrics.roc_auc_score(ground_truth, prediction)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    return plt, auc


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
        'engl.':'englisch',
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


def fetch_wiki_fulltext_linklabeled(subject='Maschinelles Lernen'):
    """Fetches Raw Text from specified Wiki article and captures info about linkages by writing
    ' @@ ' behind that location when it comes from citation, ' == ' when it comes from direct link
    and ' ++ ' when it comes from the linktext appearing in the text.
    Args:
        subject (str, optional): Name of Wiki-Article. Defaults to 'Maschinelles Lernen'.

    Returns:
        str: Raw text from wiki article (including citations) and including _ after a link
    """
    parse_url = 'https://de.wikipedia.org/w/api.php'
    params_parse = {
        'action': 'parse',
        'page': subject,
        'format': 'json',
        'prop': 'text|links|linkshere',
        'redirects': ''
    }

    # Make the API request
    response_parse = requests.get(parse_url, params=params_parse).json()
    processed_text = ''
    try:
        raw_html = response_parse['parse']['text']['*']
        soup = BeautifulSoup(raw_html, 'html.parser')
        soup.find_all('p')
        text = ''
        for p in soup.find_all('p'):
            text += p.text

        p_elements = soup.find_all('p')

        linked_words =[]
        for p in p_elements:
            # Iterate over each child element of the p element
            for child in p.children:
                if child.name  in (None,'i','b') :
                    # If tag is from free-text (none, italic or bold), add @@ behind linked words
                    # and append
                    linked_text = prefix_on_substring(linked_words,child.text,' ++ ')
                    processed_text += linked_text
                elif child.name == 'sup':
                    # If tag is citation
                    # Only add the @@
                    if processed_text[-1]=='.':
                        processed_text= processed_text[:-1] + " @@ " + '.'
                    else:
                        processed_text += " @@ "
                elif child.name == "a":
                    # If the child element is an "a" tag (link here), add a "@@" after it
                    # Also add title of linked page, and link-text with leading space
                    # and trailing space/comma/fullstop to linked_words list
                    processed_text += child.text + " == "
                    linked_words.append(" " + child.get("title") + " ")
                    linked_words.append(" " + child.get("title") + ",")
                    linked_words.append(" " + child.get("title") + ".")
                    linked_words.append(" " + child.text + " ")
                    linked_words.append(" " + child.text + ",")
                    linked_words.append(" " + child.text + ".")
                else:
                    pass
    finally:
        return processed_text
    

def split_classify_wiki_text(wiki_raw_text: str,
                             text_column='text',
                             label_column='label') -> pd.DataFrame:
    """Transforms raw_text wiki article into df with sentences and 
    their is_claim label, based on citation
    
    Args:
        wiki_raw_text (str): Wiki article from API including citations "[1]"
        text_column (str): Text column name. Defaults to 'text'
        label_column (str): Label column name. Defaults to 'label'
    Raises:
        TypeError: 
    Returns:
        pd.DataFrame: DF with sentence in text_column and label in label_column
    """
    wiki_raw_text = str(wiki_raw_text)
    if not isinstance(wiki_raw_text, str):
        raise TypeError("No raw text type!")
    no_cit_text = re.sub("\[.*?\]", "@@", wiki_raw_text)
    ##Replaced all parts of citation [1],[2] etc. with _
    i = 0
    no_cit_text = no_cit_text.replace(".@@", "@@.")
    ##Moved underscore from citation behind sentence in front of fullstop
    while i < 5:
        no_cit_text = no_cit_text.replace(".@@", ".")
        ##iterates in case there of multiple _ behind fullstop.
        i += 1
    cleaned_text = clean_special_characters(no_cit_text)
    sentence_list = cleaned_text.split(". ")
    is_claim = label_wiki_sentences(sentence_list)
    df = pd.DataFrame()
    df[text_column] = sentence_list
    df[label_column] = is_claim
    return df


def label_wiki_sentences(sen_text: list[str]) -> list[bool]:
    """Labels preprocessed sentence list (wiki article) as claim or non-claim, based on 
    if they contain '@@'. Returns list of bools. Removes underscore afterwards.

    Args:
        sen_text (list): list of preprocessed sentences of wiki articles 

    Returns:
        list[bool]: List of bools, analogues to sentence list and if they contain underscore or not.
    """
    is_claim = []
    for i, text in enumerate(sen_text):
        if ((" @@ " in sen_text[i]) or (" == " in sen_text[i]) or (" ++ " in sen_text[i])):
            sen_text[i] = text.replace(' @@ ', '')
            sen_text[i] = text.replace(' == ', '')
            sen_text[i] = text.replace(' ++ ', '')
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
    new_text = new_text.replace('\n', ' ')
    new_text= new_text.replace("[", "").replace("{", "").replace("*", "").replace("]","").replace('*',"")
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


def downsample_dataframe_classdist(df: pd.DataFrame, label_column='label', random_state=2):
    """Downsample a DF with binary label column to the size of the smaller class.

    Args:
        df (DataFrame): Df to be downsampled.
        label_column (str, optional): Namen of label column. Defaults to 'label'.
        random_state (int, optional): Random seed. Defaults to 2.

    Returns:
        pd.DataFrame: downsampled DF
    """
    df_true = df[df[label_column] == True]
    df_false = df[df[label_column] == False]
    if len(df_true) == len(df_false):
        return df
    min_length = min(len(df_true), len(df_false))
    df_true = df_true.sample(min_length, random_state=random_state)
    df_false = df_false.sample(min_length, random_state=random_state)
    df = pd.concat([df_true, df_false], ignore_index=True)
    return df


# def upsample_dataframe_classdist(df: pd.DataFrame, label_column='label', random_state=2):
#     """Upsample a DF with binary label column to the size of the smaller class.

#     Args:
#         df (DataFrame): Df to be upsampled.
#         label_column (str, optional): Name of label column. Defaults to 'label'.
#         random_state (int, optional): Random seed. Defaults to 2.

#     Returns:
#         pd.DataFrame: downsampled DF
#     """
#     df_true = df[df[label_column] == True]
#     df_false = df[df[label_column] == False]
#     if len(df_true) == len(df_false):
#         return df
#     max_length = max(len(df_true), len(df_false))
#     df_true = df_true.sample(max_length, random_state=random_state)
#     df_false = df_false.sample(max_length, random_state=random_state)
#     df = pd.concat([df_true, df_false], ignore_index=True)
#     return df


def split_train_val(df,
                    test_size=.10,
                    random_state=2,
                    label_column='label') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits df into train-test-dfs. Default 10% test and preserves distribution of label_column.

    Args:
        df (_type_): dataframe of any kind
        test_size (float, optional): Fraction of Test Set. Defaults to .10.
        random_state (int): Random seed. Defaults to 2.
        label_column (str): Name of the labeling column whose distribution is supposed to be 
        preserved. Defaults to 'label'.

    Returns:
        df: train & test df of incoming type
    """
    train_df, test_df = train_test_split(df,
                                         test_size=test_size,
                                         random_state=random_state,
                                         stratify=df[label_column])
    return train_df, test_df


def remove_stopwords(text: str, stopwords=stopwords.words("german")) -> str:
    """Removes stopwords from a text.

    Args:
        text (str): Input text
        stopwords (_type_, optional): List of words. Defaults to stopwords.words("german").

    Returns:
        str: _description_
    """
    stop = set(stopwords)
    filtered_words = ''
    if isinstance(text, str):
        filtered_words = [
            word.lower() for word in text.split()
            if (isinstance(word, str) & (word.lower() not in stop))
        ]
    return " ".join(filtered_words)


def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


# def decode(sequence):
#     return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])


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
    os.chdir(
        '/Users/jannis/Desktop/GitRepos/Master/masterthesis/Master_Thesis/data/fang-covid-main/articles'
    )
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
