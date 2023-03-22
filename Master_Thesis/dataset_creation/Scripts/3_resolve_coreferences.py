import pandas as pd
import spacy
from utils import *
import spacy
import coreferee
import de_core_news_sm
import ast
from itertools import zip_longest
import Levenshtein
#TODO Function that drops all rows 
nlp = spacy.load('de_core_news_sm')
nlp.add_pipe('coreferee')

all_article_list_full = pd.read_csv('../../data/2_articles_labeled_cleaned.csv')
all_article_list_full['sentence_list'] = all_article_list_full['sentence_list'].apply(
    ast.literal_eval)

text_list = all_article_list_full.cleaned_article_text
resolved_text_list = []

articles_pos_list = []

articles_sentence_list = all_article_list_full.sentence_list

articles_stripped_sentence_list = []
articles_resolved_sentence_list = []

indices_to_drop=[]

for i,text in enumerate(text_list):
    doc = nlp(str(text))
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
    striped_sentence_list = [s for s in sentence_list if s.strip()]

    resolved_text_list.append(resolved_text)
    articles_resolved_sentence_list.append(resolved_sentence_list)
    articles_stripped_sentence_list.append(striped_sentence_list)

    if len(resolved_sentence_list) != len(sentence_list):
        print('index: ' + str(i) + ' resolved_length: ' + str(len(resolved_sentence_list)))
        print('striped (original) length: ' + str(len(sentence_list)))
        print(all_article_list_full['title'].iloc[i])
        indices_to_drop.append(i)
        store_text1=''
        store_text2=''
        for item1, item2 in zip_longest(resolved_sentence_list,striped_sentence_list):
            if (Levenshtein.ratio(item1,item2)<0.95):
                i+=1

                if (Levenshtein.ratio(item1, item2) < 0.5):
                    print(store_text1)
                    print(store_text2)
                    print(" << To Here >>")
                    print("Resolved: " + str(item1))
                    print("Original: " + str(item2))
                    print("<< Here >> ")
                    print( " ")
                    break
            store_text1 = ("Resolved: " + item1)
            store_text2 = ("Original: " + item2)

    if len(resolved_text_list) % 100 ==0:
        print(len(resolved_text_list))

all_article_list_full['resolved_text_list'] = resolved_text_list
all_article_list_full['resolved_sentence_list'] = articles_resolved_sentence_list
print('dropping no. indices:' + str(len(indices_to_drop)))
for i in indices_to_drop:
    print(i)
all_article_list_full.drop(indices_to_drop)
all_article_list_full.reset_index(drop=True)
df_to_store = all_article_list_full[[
    'title', 'bytes', 'resolved_text_list', 'cleaned_article_text', 'sub_texts', 'resolved_sentence_list','sentence_list',
    'quot_truth_list', 'link_truth_list', 'linkname_truth_list'
]]
df_to_store.to_csv('../../data/2_articles_labeled_cleaned.csv',index=False)
