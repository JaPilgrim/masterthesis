"""Adds reference resolution to the data.
1. Loops through all articles
2. Builds pipeline with spacy coreferee
3. Transforms an article full-text to a spacy-doc object.
3. Replaces refernces with strongest representation
    -Set up index list of tokens
    -loops throgh chain and chooses strongest chain member
    -If neither new nor old ref contains full stop, replace index in token index list
    -Set up new doc with index list
    -Get text of that doc
    -Split again at". "
    -If sentence number doesnt match, drop whole row (only twice)
4. Store as df row=articles
    -added resolved_sentence_list and resolved_text_list
"""
import pandas as pd
import spacy
from utilities.utils import *
import spacy
import coreferee
import de_core_news_sm
import ast
from itertools import zip_longest
import Levenshtein
import time
#TODO Function that drops all rows
start_time = time.time()
nlp = spacy.load('de_core_news_sm',disable=['ner','parser'])
nlp.add_pipe('coreferee')


folder = '../../../data_files/pipeline_steps/excellent_articles/'
input_path = f"{folder}2_articles_labeled_cleaned.csv"
output_path = f"{folder}3_articles_resolved_labeled_cleaned)_300.csv"

df = pd.read_csv(input_path,nrows=300)
df['sentence_list'] = df['sentence_list'].apply(
    ast.literal_eval)

text_list = df.cleaned_article_text.astype(str)
resolved_text_list = []

articles_pos_list = []

articles_sentence_list = df.sentence_list

articles_stripped_sentence_list = []
articles_resolved_sentence_list = []

indices_to_drop=[]

docs = list(nlp.pipe((text_list)))

for i,doc in enumerate(docs):
    # doc = nlp(str(text))
    sorted_tokens = sorted(doc, key=lambda token: token.i)
    for chain in doc._.coref_chains:
        for mention in chain:
            for index in mention:
                curr_token = doc[index]
                if doc._.coref_chains.resolve(doc[index]) is None:
                    continue
                if "."  in doc._.coref_chains.resolve(doc[index])[0].text:
                    continue
                if '.' in sorted_tokens[index].text:
                    continue
                sorted_tokens[index] = doc._.coref_chains.resolve(doc[index])[0]
    resolved_doc = spacy.tokens.Doc(doc.vocab, words=[token.text for token in sorted_tokens])

    resolved_text = resolved_doc.text
    resolved_sentence_list = resolved_text.split('. ')
    resolved_sentence_list = [s for s in resolved_sentence_list if s.strip()]

    sentence_list = articles_sentence_list[i]
    # striped_sentence_list = [s for s in sentence_list if s.strip()]

    resolved_text_list.append(resolved_text)
    # sentence_list = [item for item in sentence_list if item != ""]
    resolved_sentence_list = [item for item in resolved_sentence_list if item != ""]

    articles_resolved_sentence_list.append(resolved_sentence_list)
    # articles_stripped_sentence_list.append(striped_sentence_list)

    if len(resolved_sentence_list) != len(sentence_list):
        print('index: ' + str(i) + ' resolved_length: ' + str(len(resolved_sentence_list)))
        print('striped (original) length: ' + str(len(sentence_list)))
        print(df['title'].iloc[i])
        print(resolved_sentence_list[-2:])
        print(sentence_list[-2:])
        indices_to_drop.append(i)
        # store_text1=''
        # store_text2=''
        # for item1, item2 in zip_longest(resolved_sentence_list, resolved_sentence_list):
        #     if (Levenshtein.ratio(item1,item2)<0.95):
        #         i+=1
        #         if (Levenshtein.ratio(item1, item2) < 0.5):
        #             print(store_text1)
        #             print(store_text2)
        #             print(" << To Here >>")
        #             print("Resolved: " + str(item1))
        #             print("Original: " + str(item2))
        #             print("<< Here >> ")
        #             print( " ")
        #             break
        #     store_text1 = ("Resolved: " + item1)
        #     store_text2 = ("Original: " + item2)

    if len(resolved_text_list) % 50 ==0:
        print(len(resolved_text_list))

df['resolved_text_list'] = resolved_text_list
df['resolved_sentence_list'] = articles_resolved_sentence_list
print('dropping no. indices:' + str(len(indices_to_drop)))
for i in indices_to_drop:
    print(i)
df=df.drop(indices_to_drop)
df=df.reset_index(drop=True)
df_to_store = df[[
    'title', 'bytes', 'resolved_text_list', 'cleaned_article_text', 'sub_texts', 'resolved_sentence_list','sentence_list',
    'quot_truth_list', 'link_truth_list', 'linkname_truth_list'
]]
df_to_store.to_csv(output_path,index=False)
end_time = time.time()

took_time = end_time - start_time
print(f' took {took_time} time')