"""Reads a csv that contains a list of all articles with their fulltext.
1. Cleans the texts.
    -replaces abbreviations
    -cleans special characters (except @@ == ++)

2. Splits into sentences
    -after ". "
    -removes all remaining full stops
    - in df stored as list of lists

3. Uses the  marks to determine truth status by dataset type. 
    @@ Quotation labeled
    == linklabeled
    ++ linknamelabeled
    
    adds following 
 
4. Joins split sentences for clean text version per article.
5. Merges everything into df and stores as .csv where articles = rows
"""
import pandas as pd

from utilities.utils import *


folder = '../../../data_files/pipeline_steps/excellent_articles/'
input_path = f"{folder}1_all_articles_fetched.csv"
output_path = f"{folder}2_articles_labeled_cleaned.csv"

df = pd.read_csv(input_path)

drop_indices = []
text_list = df['sub_texts']
for i, text in enumerate(text_list):
    if len(str(text)) < 400 or 'Liste' in df['title'].iloc[i]:
        drop_indices.append(i)

print(f"Dropping {len(drop_indices)} indices due to insufficient length or 'Liste' :")
print(drop_indices)
df = df.drop(drop_indices)
df = df.reset_index(drop=True)
drop_indices =[]
text_list = df.sub_texts
cleaned_text_list = []

articles_sen_list = []

articles_quot_truth_list = []
articles_link_truth_list = []
articles_linkname_truth_list = []

for article_index,text in enumerate(text_list):
    curr_sen_list = []

    curr_quot_truth = []
    curr_link_truth = []
    curr_linkname_truth = []
    str_cast_text = str(text)
    no_abbr_text = replace_abbreviations(str_cast_text)
    no_spec_text = clean_special_characters(no_abbr_text)

    curr_split_list = no_spec_text.split('. ')
    curr_split_list = [s.replace('.','') for s in curr_split_list]
    curr_split_list = [s for s in curr_split_list if s.strip()]
    curr_sen_list = curr_split_list
    for i, text in enumerate(curr_sen_list):
        if ('@@' in curr_sen_list[i]):
            curr_sen_list[i] = curr_sen_list[i].replace('@@', '')
            curr_sen_list[i] = curr_sen_list[i].replace('==', '')
            curr_sen_list[i] = curr_sen_list[i].replace('==', '')
            curr_sen_list[i] = curr_sen_list[i].replace('++', '')

            curr_quot_truth.append(True)
            curr_link_truth.append(True)
            curr_linkname_truth.append(True)
        elif ('==' in curr_sen_list[i]):
            curr_sen_list[i] = curr_sen_list[i].replace('==', '')
            curr_sen_list[i] = curr_sen_list[i].replace('==', '')
            curr_sen_list[i] = curr_sen_list[i].replace('++', '')

            curr_quot_truth.append(False)
            curr_link_truth.append(True)
            curr_linkname_truth.append(True)
        elif ('++' in curr_sen_list[i]):
            curr_sen_list[i] = curr_sen_list[i].replace('++', '')

            curr_quot_truth.append(False)
            curr_link_truth.append(False)
            curr_linkname_truth.append(True)
        else:
            curr_quot_truth.append(False)
            curr_link_truth.append(False)
            curr_linkname_truth.append(False)
        curr_sen_list[i] = curr_sen_list[i].replace('@@', '')
        curr_sen_list[i] = curr_sen_list[i].replace('==', '')
        curr_sen_list[i] = curr_sen_list[i].replace('==', '')
        curr_sen_list[i] = curr_sen_list[i].replace('++', '')
        if curr_sen_list[i] == '':
            print("Dropped empty sentence :")
            print(i)
            print(df['title'].iloc[article_index])
            del curr_sen_list[i]
            del curr_quot_truth[i]
            del curr_link_truth[i]
            del curr_linkname_truth[i]


            continue
    curr_clean_joined_text = ". ".join(curr_sen_list)

    cleaned_text_list.append(curr_clean_joined_text)
    articles_sen_list.append(curr_sen_list)
    if(len(curr_sen_list) != len(curr_quot_truth)):
        print("Labels dont match here:")
        print(df['title'].iloc[article_index])
        drop_indices.append(article_index)
    articles_quot_truth_list.append(curr_quot_truth)
    articles_link_truth_list.append(curr_link_truth)
    articles_linkname_truth_list.append(curr_linkname_truth)

    if len(articles_quot_truth_list) % 100 == 0:
        print(len(articles_quot_truth_list))



df['cleaned_article_text'] = cleaned_text_list

df['sentence_list'] = articles_sen_list

df['quot_truth_list'] = articles_quot_truth_list
df['link_truth_list'] = articles_link_truth_list
df['linkname_truth_list'] = articles_linkname_truth_list

print(f"Dropping {len(drop_indices)} indices due to non-matching Label & Sentence lengt :")
print(drop_indices)

df = df.drop(drop_indices)
df = df.reset_index(drop=True)

df.to_csv(output_path, index=False)
